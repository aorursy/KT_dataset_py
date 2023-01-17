import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def kappa_loss(p, y, n_classes=5, eps=1e-10):
    """
    QWK loss function as described in https://arxiv.org/pdf/1612.00775.pdf
    
    Arguments:
        p: a tensor with probability predictions, [batch_size, n_classes],
        y, a tensor with one-hot encoded class labels, [batch_size, n_classes]
    Returns:
        QWK loss
    """
    
    W = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            W[i,j] = (i-j)**2
    
    W = torch.from_numpy(W.astype(np.float32)).to(device)
    
    O = torch.matmul(y.t(), p)
    E = torch.matmul(y.sum(dim=0).view(-1,1), p.sum(dim=0).view(1,-1)) / O.sum()
    
    return (W*O).sum() / ((W*E).sum() + eps)