from pathlib import Path

from IPython.core.debugger import set_trace

from fastai import datasets

from torch import tensor

import pickle, gzip, math, torch, matplotlib as mpl

import time

MNIST_URL = "http://deeplearning.net/data/mnist/mnist.pkl"
path = datasets.download_data(MNIST_URL , ext = ".gz"); path

with gzip.open(path, 'rb') as f:

    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding = 'latin-1')

x_train, y_train, x_valid, y_valid = map(tensor, (x_train, y_train, x_valid, y_valid)) # Converting to Tensors
n,m = x_train.shape

c = y_train.max()+1

nh = 50

n,m,c,nh # nh is number of hidden units, we are building a neural network with one hidden layer here.
import torch.nn.functional as F

from torch import nn

import torch.optim as optim
model_linear = nn.Sequential(nn.Linear(m,nh), nn.ReLU(), nn.Linear(nh,10))

opt = optim.SGD(model_linear.parameters(), lr=0.01)

loss_func = F.cross_entropy

epochs = 1

bs = 10
start_time = time.time()

for epoch in range(epochs):

    for i in range((n-1)//bs + 1):

        start_i = i*bs

        end_i = start_i+bs

        xb = x_train[start_i:end_i]

        yb = y_train[start_i:end_i]

        pred = model_linear(xb)

        loss = loss_func(pred, yb)

        loss.backward()

        opt.step() # Updating weights.

        opt.zero_grad()



print("time to train linear model with basics:", round(time.time() - start_time),"seconds." ,"epochs:", epochs)
def accuracy(out, yb): 

    return (torch.argmax(out, dim=1)==yb).float().mean()
from sklearn import metrics

y_pred = torch.argmax(model_linear(x_valid), dim=1)

cm = metrics.confusion_matrix(y_valid, y_pred)

cm
accuracy(model_linear(x_valid), y_valid)

# Pytorch train and valid sets

train = torch.utils.data.TensorDataset(x_train, y_train)

valid = torch.utils.data.TensorDataset(x_valid, y_valid)
train_loader = torch.utils.data.DataLoader(train, batch_size = 10, shuffle = False)

valid_loader = torch.utils.data.DataLoader(valid, batch_size = 10, shuffle = False)
train_loader
model_linear2 = nn.Sequential(nn.Linear(m,nh), nn.ReLU(), nn.Linear(nh,10))
opt = optim.SGD(model_linear2.parameters(), lr=0.01)
epochs = 1

import time

start_time = time.time()

for epoch in range(epochs):

    for batch_idx, (xb, yb) in enumerate(train_loader):

        pred = model_linear2(xb)

        loss = loss_func(pred, yb)

        loss.backward()

        opt.step()

        opt.zero_grad()

        

print("time to train linear model with dataloader:", round(time.time() - start_time),"seconds." ,"epochs:", epochs)

accuracy(model_linear2(x_valid), y_valid) # above 90%