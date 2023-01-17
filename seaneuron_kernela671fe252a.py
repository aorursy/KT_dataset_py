import numpy as np 

import pandas as pd 

import os

print(os.listdir("../input"))

import torch

from torch.autograd import Variable
df= pd.read_csv("../input/heart.csv")

df.columns
df.groupby(by='target').agg("mean")
from sklearn.preprocessing import normalize as norm

x_data = Variable(torch.Tensor(norm(df['oldpeak'].values.reshape(-1, 1))))

y_data = Variable(torch.Tensor(norm(df['slope'].values.reshape(-1, 1))))
from scipy.stats import pearsonr

from itertools import combinations



for a, b in combinations(df.columns, 2):

    i,p = pearsonr(df[a].values, df[b].values)

    if np.absolute(i) >= .5:

        print(a,b,i,p)
class Model(torch.nn.Module):



    def __init__(self):

        """

        In the constructor we instantiate two nn.Linear module

        """

        super(Model, self).__init__()

        self.linear = torch.nn.Linear(1, 1)  # One in and one out



    def forward(self, x):

        """

        In the forward function we accept a Variable of input data and we must return

        a Variable of output data. We can use Modules defined in the constructor as

        well as arbitrary operators on Variables.

        """

        y_pred = self.linear(x)

        return y_pred



# our model

model = Model()
criterion = torch.nn.MSELoss(size_average=False)

optimizer = torch.optim.SGD(model.parameters(), lr=0.000001)



# Training loop

for epoch in range(500):

        # Forward pass: Compute predicted y by passing x to the model

    y_pred = model(x_data)



    # Compute and print loss

    loss = criterion(y_pred, y_data)

    print(epoch, loss.data)



    # Zero gradients, perform a backward pass, and update the weights.

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()