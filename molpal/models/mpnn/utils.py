from typing import Optional

from torch import clamp, nn


def get_loss_func(dataset_type: str, uncertainty_method: Optional[str] = None) -> nn.Module:
    """Get the loss function corresponding to a given dataset type

    Parameters
    ----------
    dataset_type : str
        the type of dataset
    uncertainty_method : Optional[str]
        the uncertainty method being used

    Returns
    -------
    loss_function : nn.Module
        a PyTorch loss function

    Raises
    ------
    ValueError
        if is dataset_type is neither "classification" nor "regression"
    """
    if dataset_type == "classification":
        print("using BCEWithLogitsLoss")
        return nn.BCEWithLogitsLoss(reduction="none")
    if dataset_type == "classification_imbalanced":
        # BinaryMCCLoss
        print("using BinaryMCCLoss")
        return BinaryMCCLoss()

    elif dataset_type == "regression":
        if uncertainty_method == "mve":
            return negative_log_likelihood

        return nn.MSELoss(reduction="none")

    raise ValueError(f'Unsupported dataset type: "{dataset_type}."')


def negative_log_likelihood(means, variances, targets):
    """The NLL loss function as defined in:
    Nix, D.; Weigend, A. ICNN’94. 1994; pp 55–60 vol.1"""
    variances = clamp(variances, min=1e-5)
    return (variances.log() + (means - targets) ** 2 / variances) / 2

class BinaryMCCLoss(nn.Module):
    """Binary Matthews Correlation Coefficient loss for imbalanced classification.
    
    This loss function optimizes the Matthews Correlation Coefficient, which is
    particularly useful for imbalanced datasets.
    """
    def __init__(self, epsilon: float = 1e-8):
        """
        Parameters
        ----------
        epsilon : float, default=1e-8
            Small value to avoid division by zero
        """
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, preds, targets):
        """Calculate the MCC-based loss
        
        Parameters
        ----------
        preds : torch.Tensor
            Predicted values (logits)
        targets : torch.Tensor
            Target values (0 or 1)
            
        Returns
        -------
        torch.Tensor
            The loss value
        """
        import torch
        # Apply sigmoid if inputs are logits
        if not (0 <= preds.min() and preds.max() <= 1):
            preds = torch.sigmoid(preds)
        
        # Calculate confusion matrix elements
        TP = (targets * preds).sum(0)
        FP = ((1 - targets) * preds).sum(0)
        TN = ((1 - targets) * (1 - preds)).sum(0)
        FN = (targets * (1 - preds)).sum(0)
        
        # Calculate MCC
        numerator = TP * TN - FP * FN
        denominator = torch.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + self.epsilon)
        
        # Convert to loss (1 - MCC)
        mcc = numerator / denominator
        loss = 1 - mcc
        
        return loss