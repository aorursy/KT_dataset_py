import numpy as np

import pandas as pd

import torch as tr

import torch.nn as nn

from torch.autograd import Variable
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

y = train.target

train.drop(["target", "id"], axis=1, inplace=True)

test.drop(["id"], axis=1, inplace=True)

x = tr.Tensor(train.append(test).values)

x = Variable(x).cuda()

del train, test
class autoencoder(nn.Module):

    def __init__(self):

        super(autoencoder, self).__init__()

        indim = 300

        layers = []

        for nunits in [400, 350, 300, 295, 290]:

            layers.append(nn.Linear(indim, nunits))

            layers.append(nn.ELU(True))

            layers.append(nn.BatchNorm1d(nunits))

            indim = nunits

        layers.pop()

        self.encoder = nn.Sequential(*layers)

        layers = []

        for nunits in [290, 293, 295, 297, 300]:

            layers.append(nn.Linear(indim, nunits))

            layers.append(nn.ELU(True))

            layers.append(nn.BatchNorm1d(nunits))

            indim = nunits

        layers.pop()

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):

        x = self.encoder(x)

        x = self.decoder(x)

        return x
model = autoencoder().cuda()

criterion = nn.MSELoss()

optimizer = tr.optim.Adadelta(model.parameters())

mse = []

last_improvement = 0

best_loss = float("Inf")

while last_improvement<100:

    loss = criterion(model(x), x)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    mse.append(loss.data.item())

    if best_loss > mse[-1]: 

        last_improvement=0

        best_loss = mse[-1]

    else: last_improvement+=1

    # print("MSE: ", mse[-1], "\tLast improvement: ", last_improvement, end="\r")

print("best MSE achieved: ", best_loss)
criterion = nn.L1Loss()

mae = criterion(model(x), x).data.item()

print("MAE: ", mae)
tr.save(model.state_dict(), './autoencoder.pth')

model.eval()

z = model.encoder(x).cpu().data.numpy()

df = pd.DataFrame(z).join(pd.DataFrame({"target" : y}))

df.to_csv("./encoded_features.csv")