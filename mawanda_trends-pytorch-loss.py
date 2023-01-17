import torch



DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class TReNDSLoss(torch.nn.Module):

    def __init__(self):

        super().__init__()

        self.weights = torch.tensor([.3, .175, .175, .175, .175], dtype=torch.float32, device=DEVICE)



    def __loss(self, output, target):

        nom = torch.sum(torch.abs(output-target), dim=0)

        denom = torch.sum(target, dim=0)

        return torch.sum(self.weights * nom / denom)



    def forward(self, output: torch.Tensor, target: torch.Tensor):

        return self.__loss(output, target)

output = torch.rand(64, 5)  # 64 is the batch dimension

target = torch.rand(64, 5)

loss_fn = TReNDSLoss()

loss_fn(output, target).item()