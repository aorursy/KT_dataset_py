import torch
def quadratic_kappa_coefficient(output, target):

    n_classes = target.shape[-1]

    weights = torch.arange(0, n_classes, dtype=torch.float32, device=output.device) / (n_classes - 1)

    weights = (weights - torch.unsqueeze(weights, -1)) ** 2



    C = (output.t() @ target).t()  # confusion matrix



    hist_true = torch.sum(target, dim=0).unsqueeze(-1)

    hist_pred = torch.sum(output, dim=0).unsqueeze(-1)



    E = hist_true @ hist_pred.t()  # Outer product of histograms

    E = E / C.sum() # Normalize to the sum of C.



    num = weights * C

    den = weights * E



    QWK = 1 - torch.sum(num) / torch.sum(den)

    return QWK
target = torch.tensor([2,2,2,3,4,5,5,5,5,5]) - 1

output = torch.tensor([2,2,2,3,2,1,1,1,1,3]) - 1



output.shape, target.shape
import torch.nn.functional as F



target_onehot = F.one_hot(target, 5)

output_onehot = F.one_hot(output, 5)



output_onehot.shape, target_onehot.shape
quadratic_kappa_coefficient(output_onehot.type(torch.float32), target_onehot.type(torch.float32))
def quadratic_kappa_loss(output, target, scale=2.0):

    QWK = quadratic_kappa_coefficient(output, target)

    loss = -torch.log(torch.sigmoid(scale * QWK))

    return loss



class QWKLoss(torch.nn.Module):

    def __init__(self, scale=2.0):

        super().__init__()

        self.scale = scale



    def forward(self, output, target):

        # Keep trace of output dtype for half precision training

        target = F.one_hot(target.squeeze(), num_classes=6).to(target.device).type(output.dtype)

        output = torch.softmax(output, dim=1)

        return quadratic_kappa_loss(output, target, self.scale)
class QWKMetric(torch.nn.Module):

    def __init__(self, binned=False):

        super().__init__()

        self.binned = binned



    def forward(self, output, target):

        # Keep trace of dtype for half precision training

        dtype = output.dtype

        target = F.one_hot(target.squeeze(), num_classes=6).to(target.device).type(dtype)

        if self.binned:

            output = torch.sigmoid(output).sum(1).round().long()

            output = F.one_hot(output.squeeze(), num_classes=6).to(output.device).type(dtype)

        else:

            output = torch.softmax(output, dim=1)

        return quadratic_kappa_coefficient(output, target)
target = torch.randint(0, 6, (10, 1)).squeeze()

print("target: ", target)  # target class coming directly from the isup grades



output = torch.rand(10, 6)  # Logits from network, trained with not binned target

print("output: ", output)
nb_loss = QWKLoss()

b_metric = QWKMetric(binned=True)

nbl = nb_loss(output, target)

bl = b_metric(output, target)

print("not binned loss: ", nbl.item())

print("binned metric: ", bl.item())