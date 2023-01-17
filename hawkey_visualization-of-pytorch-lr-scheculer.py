import torch

import torch.nn as nn

from torch.optim import lr_scheduler

import matplotlib.pyplot as plt
class NullModule(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc = nn.Linear(1,1)

    



model = NullModule()

optimizer = torch.optim.Adam(model.parameters())



def plot_lr(scheduler, step=100):

    lrs = []

    for i in range(step):

        lr = optimizer.param_groups[0]['lr']

        scheduler.step()

        lrs.append(lr)



    plt.plot(lrs)

    plt.show()
MLT = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda x:0.9)



plot_lr(MLT)
STP = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)



plot_lr(STP)
EXP = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)



plot_lr(EXP)
CA = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)



plot_lr(CA)
Cyc = lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1, cycle_momentum=False)



plot_lr(Cyc, 10000)
Cyc = lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, total_steps=10000, cycle_momentum=False)



plot_lr(Cyc, 10000)
CAWR = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0)



plot_lr(CAWR, 100)
CAWR = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2, eta_min=0)



plot_lr(CAWR, 1000)