# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# importing all the relevant libraries which will be used 

import numpy as np

import math

import matplotlib.pyplot as plt

import matplotlib.colors

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, mean_squared_error, log_loss

from tqdm import tqdm_notebook

import seaborn as sns

import time

from IPython.display import HTML

import warnings

warnings.filterwarnings('ignore')



from sklearn.preprocessing import OneHotEncoder

from sklearn.datasets import make_blobs



import torch
# Since we are using random function to create our dataset, setting seed so that results are reproducable. 

torch.manual_seed(0)
# creating a color map beforhand so that it is easy to visualise data with various colors.



my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("",["red", "yellow", "green"])
# Using make_blobs function to create data with 4 centres(4 classes) with 2 features(columns) only with 1000 input rows

data, labels = make_blobs(n_samples= 1000, centers= 4, n_features= 2, random_state= 0)

print(data.shape,labels.shape)
# Visualising data that has been created

plt.scatter(data[:,0], data[:,1], c= labels, cmap= my_cmap)

plt.show()
# Splitting the data into train and test set so that data can be trained on 1 part and tested on the other.

#The propostion by default is 75% train and 25% test data set

X_train, X_val, Y_train, Y_val = train_test_split(data,labels, stratify = labels, random_state = 0)



# Visualising both train and test dataset

print(X_train.shape, X_val.shape, Y_train.shape, Y_val.shape, labels.shape)
# Since we will be using pytorch to create deep neural network, converting arrays to tensors.

# using map function to apply same function on list of datasets.



X_train, X_val, Y_train, Y_val = map(torch.tensor,(X_train, X_val, Y_train, Y_val))
# defining our model. a1 and a2 are activation functions of 1st and 2nd layer respectively.

# h1 and h2 are hidden layers of activation function to bring non linearity using sigmoid



def model(x):

  a1 = torch.matmul(x,weights1) + bias1 # (N,2) x (2,2) -> (N,2)

  h1 = a1.sigmoid() # (N,2)

  a2 = torch.matmul(h1,weights2) + bias2 # (N,2) x (2,4) -> (N,4)

  h2 = a2.exp()/a2.exp().sum(-1).unsqueeze(-1)

  return h2
# defining cross entropy loss function

def loss_fn(y_hat,y):

  return -(y_hat[range(y.shape[0]),y].log()).mean()
# defining accuracy function

def accuracy(y_hat,y):

  pred = torch.argmax(y_hat,dim =1)

  return (pred== y).float().mean()
torch.manual_seed(0)

weights1 = torch.randn(2,2)/ math.sqrt(2)

weights1.requires_grad_()

bias1 = torch.zeros(2, requires_grad= True)



weights2 = torch.randn(2,4) / math.sqrt(2)

weights2.requires_grad_()

bias2 = torch.zeros(4, requires_grad= True)



learning_rate = 0.2

epochs = 1000



X_train = X_train.float()

Y_train = Y_train.long()



loss_arr = []

acc_arr = []



for epoch in range(epochs) :

  y_hat = model(X_train)

  loss = loss_fn(y_hat, Y_train)

  loss.backward()

  loss_arr.append(loss.item())

  acc_arr.append(accuracy(y_hat, Y_train))



  with torch.no_grad():

    weights1 -= weights1.grad * learning_rate

    bias1 -= bias1.grad * learning_rate

    weights2 -= weights2.grad * learning_rate

    bias2 -= bias2.grad * learning_rate

    weights1.grad.zero_()

    bias1.grad.zero_()

    weights2.grad.zero_()

    bias2.grad.zero_()



plt.plot(loss_arr,'r-')

plt.plot(acc_arr,'b-')

plt.show()

print("loss before training", loss_arr[0])

print("loss after training", loss_arr[-1])
import torch.nn.functional as F

import torch.nn as nn

from torch import optim
# Defining basic class to define the depth and nodes in neural network



class FirstNetwork(nn.Module):



  def __init__(self):

    super().__init__()

    torch.manual_seed(0)

    self.net = nn.Sequential(

        nn.Linear(2,2),

        nn.Sigmoid(),

        nn.Linear(2,4),

        nn.Softmax()

    )

    



  def forward(self,x):

    return self.net(x)
# Defining a fit function separately 

def fit_v2(x,y,model,opt,loss_fn,epochs = 1000):

  for epoch in range(epochs):

    loss = F.cross_entropy(model(x), y)



    loss.backward()



    opt.step()

    opt.zero_grad()

  return loss.item()
# calling neural network class and invoking cross entropy function. 

#optim is being used to optimise the parameters defined in our class



fn = FirstNetwork()

loss_fn = F.cross_entropy

opt = optim.SGD(fn.parameters(), lr =1)

fit_v2(X_train, Y_train, fn, opt, loss_fn)