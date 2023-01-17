import torch

import matplotlib.pyplot as plt

from torch import optim
model = torch.nn.Linear(1, 1)

optimizer = torch.optim.SGD(model.parameters(), lr=1)
scheduler = optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.5, last_epoch=-1)
EPOCHS = 10

NUM_BATCHES = 100

lr_ls = []

for e in range(EPOCHS):

    for i in range(NUM_BATCHES):

        lr = scheduler.get_last_lr()[0]

        lr_ls.append(lr)

    scheduler.step()
plt.plot(lr_ls)