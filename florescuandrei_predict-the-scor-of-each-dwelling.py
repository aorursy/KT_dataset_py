from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import mean_squared_error
nRowsRead = 1000

df1 = pd.read_csv('/kaggle/input/Bucharest_HousePriceDataset.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = 'Bucharest_HousePriceDataset.csv'
from functools import partial

from IPython.display import HTML

import math

import matplotlib.pyplot as plt

from matplotlib import animation, rc

import numpy as np

from sklearn.datasets import make_regression

from sklearn.linear_model import LinearRegression

import torch

import torch.nn as nn

import sklearn

import torch.nn.functional as F

import tensorflow as tf

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import StandardScaler

from sklearn import preprocessing

from sklearn.model_selection import train_test_split
class GDLinearRegression(nn.Module):

  """A simple Linear Regression model"""



  def __init__(self):

    super().__init__()

    # We're initializing our model with random weights

    self.w = nn.Parameter(torch.randn(6, requires_grad = True))

    self.b = nn.Parameter(torch.randn(1, requires_grad = True))



  def __call__(self, x: torch.Tensor) -> torch.Tensor:

    x = torch.Tensor(x)

    result = x @ self.w+self.b

    return result



  # PyTorch is accumulating gradients

  # After each Gradient Descent step we should reset the gradients

  def zero_grad(self):

    self.w.grad.zero_()

    self.b.grad.zero_()
class MSE():

  """The Mean Squared Error loss"""

  

  def __call__(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

    x = torch.Tensor(x)

    target = torch.Tensor(target)

    #mse = (abs(x-target)).sum()

    mse = ((x-target)**2).sum().sqrt().mean()

    return mse
class GD():

  """Gradient Descent optimizer"""



  def __init__(self, params: torch.Tensor, lr: int):

    self.w, self.b = list(params)

    self.lr = lr





  def step(self):

   # print(f'lr:{self.lr} w.grad*lr: {self.w.grad} w: {self.w}')

    self.w -= self.lr*self.w.grad # Todo

    #print(f'w nou: {self.w} gradient nou {self.w.grad}')

    self.b -= self.lr*self.b.grad # Todo
def train(model: GDLinearRegression, data: torch.Tensor, 

          labels: torch.Tensor, optim: GD, criterion: MSE):

  """Linear Regression train routine"""



  predictions = model(data) # Todo

  loss = criterion(labels,predictions) # Todo

  #loss_history.append(loss.item())

  loss.backward() # Todo

  #print(f'loss{loss}')

  

  with torch.no_grad():

    optim.step() # Todo

    model.zero_grad()

  

  return model
X=df1.drop(columns='Scor').values



y=df1[['Scor']].values.ravel()

x_train, x_valid, y_train, y_valid = train_test_split(X,y, train_size = 0.8)



std_scale = preprocessing.StandardScaler().fit(x_train)

x_train = std_scale.transform(x_train)

x_train = torch.tensor(x_train).float()



#Folosim aceeasi deviatie standard de la training pentru test

x_valid = std_scale.transform(x_valid)

x_valid = torch.tensor(x_valid).float()



mse = MSE()
best_loss=10**1000

best_lr = 0

total_steps = 100

for idx in np.linspace(0.001,20,100):

    lr = idx

    total_steps = 100



    model = GDLinearRegression()

    optimizer = GD(model.parameters(), lr=lr)

    criterion = MSE()



    for i in range(total_steps):

        train(model, x_train, y_train, optimizer, criterion)



    with torch.no_grad():

        y_pred = model(x_train)

    if best_loss > mse(y_pred,y_train).item():

        best_loss = mse(y_pred,y_train).item()

        best_lr = idx
lr = best_lr

total_steps = 500

model = GDLinearRegression()

optimizer = GD(model.parameters(), lr=lr)

criterion = MSE()



for i in range(total_steps):

    train(model, x_train, y_train, optimizer, criterion)



with torch.no_grad():

    y_pred = model(x_train)
with torch.no_grad():

    y_pred = model(x_train)

predicted = torch.round(y_pred).numpy()

accuracy = (predicted==y_train).sum()/predicted.shape[0]

print(accuracy)
with torch.no_grad():

    y_pred = model(x_valid)

predicted = torch.round(y_pred).numpy()

accuracy = (predicted==y_valid).sum()/predicted.shape[0]

print(accuracy)
regression_matrix = confusion_matrix(predicted,y_valid)

regression_mse = mse(predicted,y_valid)

print(regression_matrix)

print('MSE:', regression_mse)
X=df1.drop(columns='Nr Camere').values



y=df1[['Nr Camere']].values.ravel()

x_train, x_valid, y_train, y_valid = train_test_split(X,y, train_size = 0.8)



std_scale = preprocessing.StandardScaler().fit(x_train)

x_train = std_scale.transform(x_train)

x_train = torch.tensor(x_train).float()



#Folosim aceeasi deviatie standard de la training pentru test

x_valid = std_scale.transform(x_valid)

x_valid = torch.tensor(x_valid).float()



mse = MSE()
lr = best_lr

total_steps = 500

model = GDLinearRegression()

optimizer = GD(model.parameters(), lr=lr)

criterion = MSE()



for i in range(total_steps):

    train(model, x_train, y_train, optimizer, criterion)



with torch.no_grad():

    y_pred = model(x_train)
with torch.no_grad():

    y_pred = model(x_train)

predicted = torch.round(y_pred).numpy()

accuracy = (predicted==y_train).sum()/predicted.shape[0]

print(accuracy)
with torch.no_grad():

    y_pred = model(x_valid)

predicted = torch.round(y_pred).numpy()

accuracy = (predicted==y_valid).sum()/predicted.shape[0]

print(accuracy)
regression_matrix1 = confusion_matrix(predicted,y_valid)

regression_mse1 = mse(predicted,y_valid)

print(regression_matrix1)

print('MSE:', regression_mse1)
X=df1.drop(columns='Scor').values



y=df1[['Scor']].values

y=y-1

x_train, x_valid, y_train, y_valid = train_test_split(X,y, train_size = 0.8)



std_scale = preprocessing.StandardScaler().fit(x_train)

x_train = std_scale.transform(x_train)

x_train = torch.tensor(x_train).float()



#Folosim aceeasi deviatie standard de la training pentru test

x_valid = std_scale.transform(x_valid)

x_valid = torch.tensor(x_valid).float()
class TwoLayer(nn.Module):

    def __init__(self, in_size: int, hidden_size: int, out_size: int):

        super().__init__()

        self._layer1 = nn.Linear(in_size, hidden_size)

        self._layer2 = nn.Linear(hidden_size, out_size)

    def forward(self, x):

        x = self._layer1(x)

        x = torch.relu(x)

    

        x = self._layer2(x)

        return x
model = TwoLayer(6,5000,5)

NUM_EPOCHS = 2000



optim = torch.optim.SGD(model.parameters(), lr=2.5)

for i in range(NUM_EPOCHS):

    model.train()

    optim.zero_grad()

    output = model(x_train)

    criterion = nn.CrossEntropyLoss()

    target = torch.tensor(y_train).long().squeeze(1)

    loss = criterion(output, target)

    loss.backward()

    optim.step()
predicted = np.array(torch.argmax(model(x_train), dim=-1))

accuracy = (predicted==y_train.ravel()).sum()/predicted.shape[0]

print(accuracy)
predicted = np.array(torch.argmax(model(x_valid), dim=-1))

accuracy = (predicted==y_valid.ravel()).sum()/predicted.shape[0]

print(accuracy)
class_matrix = confusion_matrix(predicted,y_valid)

class_mse = mse(predicted,y_valid)
print('Regresie scor\n\n',regression_matrix)

print('\nClasificare scor\n\n', class_matrix)

print('\nRegresie Nr. Camere\n\n',regression_matrix1)

print('Regresie scor\n\n',regression_mse)

print('\nClasificare scor\n\n', class_mse)

print('\nRegresie Nr. Camere\n\n',regression_mse1)
