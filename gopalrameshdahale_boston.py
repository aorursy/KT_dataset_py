import numpy as np

import pandas as pd

import torch

import torch.nn as nn

from torch.utils.data import TensorDataset

from torch.utils.data import DataLoader

import torch.nn.functional as F

import matplotlib.pyplot as plt
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']



df = pd.read_csv('../input/boston-house-prices/housing.csv',header=None, delimiter=r"\s+", names=column_names)

df.head()
df.shape
target = df['MEDV']

df.drop('MEDV',axis = 1,inplace =True)
# convert inputs and target to tensor

inputs = torch.tensor(np.array(df),dtype=torch.float)

target = torch.tensor(np.array(target),dtype=torch.float).view(-1,1)
# define dataset

train_ds = TensorDataset(inputs,target)
# define data loader

batch_size = 64

train_dl=  DataLoader(train_ds,batch_size)
for xb, yb in train_dl:

    print(xb.shape)

    print(yb.shape)

    break
# define model

model = nn.Linear(df.shape[1],1)

print(model.weight.shape)

print(model.bias.shape)
# loss function



# Define loss function

loss_fn = F.mse_loss
# Define optimizer

opt = torch.optim.SGD(model.parameters(), lr=1e-6)
# Utility function to train the model



def fit(num_epochs, model, loss_fn, opt, train_dl):

    losses = []

    # Repeat for given number of epochs

    for epoch in range(num_epochs):

        

        # Train with batches of data

        for xb,yb in train_dl:

            

            # 1. Generate predictions

            pred = model(xb)

            

            # 2. Calculate loss

            loss = loss_fn(pred, yb)

    

            # 3. Compute gradients

            loss.backward()

        

            # 4. Update parameters using gradients

            opt.step()

            

            # 5. Reset the gradients to zero

            opt.zero_grad()

        

        # Print the progress and add to losses

        if (epoch+1) % 100 == 0:

            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

            losses.append(loss.item())

    

    # plot losses

    plt.plot(np.arange(1,len(losses)+1,1),losses)
fit(10000, model, loss_fn, opt,train_dl)