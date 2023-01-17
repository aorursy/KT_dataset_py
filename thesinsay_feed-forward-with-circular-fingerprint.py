import os

__print__ = print

def print(string, end = '', flush = True):

    os.system(f'echo \"{string}\"')

    __print__(string, end = end, flush = flush)
import pandas as pd

import numpy as np



from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
import torch

import torch.nn.functional as F

from torch.optim import Adam

from torch import nn



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#df = pd.read_csv('/kaggle/input/1jhn-dataset-2/1JHN_dataset 2.csv')
! ls /kaggle/input/1jhn-dataset-2/1JHN_dataset 2.csv
df = pd.read_csv('/kaggle/input/1jhn-dataset-2/1JHN_dataset 2.csv')
df.head()
df = df.drop(['molecule_index', 'Unnamed: 0'], axis = 1)
df = df.dropna()
y = df['scalar_coupling_constant']
X = df.drop(['scalar_coupling_constant'], axis = 1)
X = X.drop(['atom_index_{}'.format(i) for i in range(7)], axis = 1)
X = X.drop(['atomic_{}'.format(i) for i in range(7)], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
X_train.to_csv('X_train')

X_test.to_csv('X_test')

y_train.to_csv('y_train')

y_test.to_csv('y_test')
Scaler = StandardScaler()
X_train = Scaler.fit_transform(X_train)
X_train.shape
def criterion(predict, truth):

    predict = predict.view(-1)

    truth   = truth.view(-1)

    assert(predict.shape==truth.shape)



    loss = torch.abs(predict-truth)

    loss = loss.mean()

    loss = torch.log(loss)

    return loss
class Feedforward(torch.nn.Module):

        def __init__(self):

            super(Feedforward, self).__init__()

            

            self.predict = nn.Sequential(

            nn.Linear(158, 1024),  #node_hidden_dim

            nn.ReLU(inplace=True),

            nn.Linear(1024, 512),

            nn.ReLU(inplace=True),

            nn.Linear(512, 1),

            )

            

        def forward(self, x):

            

            x = self.predict(x)

            

            return x
X_test = Scaler.transform(X_test)
X_test = torch.FloatTensor(X_test)

y_test = torch.FloatTensor(y_test.as_matrix())
X_train = torch.FloatTensor(X_train)

y_train = torch.FloatTensor(y_train.as_matrix())
model = Feedforward()

model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
X_train = X_train.to(device)

y_train = y_train.to(device)

X_test = X_test.to(device)

y_test = y_test.to(device)
epochs = 100000



for epoch in range(epochs):

    optimizer.zero_grad()

    # Forward pass

    y_pred = model(X_train)

    # Compute Loss

    loss = criterion(y_pred.squeeze(), y_train)

    

    #loss = l(y_pred.squeeze(), y_train)

   

    #print('Epoch {}: train loss: {}'.format(epoch, loss.item()))



    # Backward pass

    loss.backward()

    optimizer.step()

    

    y_test_pred = model(X_test)

    

    MAE = F.l1_loss(y_test_pred.squeeze(), y_test, reduction='mean')

    

    #print('Epoch {}: MAE: {}'.format(epoch, loss.item()))

    if epoch % 100 == 0:

        print('Epoch {}: MAE: {}'.format(epoch, MAE.item()))
torch.save(model.state_dict(), 'model_1JHN')