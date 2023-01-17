import pickle, gzip, math, torch, matplotlib as mpl

import time

import matplotlib.pyplot as plt

import numpy as np

import torch.nn.functional as F

from torch import tensor

from fastai import datasets  #  importing datasets of  fastai  for convenience.

from torch import nn

import torch.optim as optim

MNIST_URL = "http://deeplearning.net/data/mnist/mnist.pkl"
# Looking to Use GPU if available

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); print(device)
# Using FastAI's helper function for convenience,this downloads the data and gives back the location of the dataset

path = datasets.download_data(MNIST_URL , ext = ".gz"); #print(path)

with gzip.open(path, 'rb') as f:

    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding = 'latin-1')



print(type(x_train))

# Converting the numpy array to Tensors

x_train, y_train, x_valid, y_valid = map(tensor, (x_train, y_train, x_valid, y_valid)) # Converting to Tensors

print(type(x_train))

# Looking at the shape
print(x_train.shape)
# Looking at the one of the image

from PIL import Image

import matplotlib.pyplot as plt

print(y_train[40000])

_ = plt.imshow(x_train[40000].view(28,28), cmap='gray')
print(y_train[0])

x_train[0].shape
num_rows,sz       = x_train.shape # observations , size of image

classes          = y_train.max()+1

num_hidden_units = 84

print(num_rows, sz, classes)
model_linear = nn.Sequential(nn.Linear(sz,84), nn.ReLU(), nn.Linear(84,10))

opt = optim.SGD(model_linear.parameters(), lr=0.01)

loss_func = F.cross_entropy

epochs = 1

batch_size = 8
# Look at the model

model_linear
start_time = time.time()

for epoch in range(epochs):

    print("starting ")

    for i in range((num_rows-1)//batch_size + 1):

        start_i = i * batch_size

        end_i = start_i + batch_size

        xb = x_train[start_i:end_i]

        yb = y_train[start_i:end_i]

        pred = model_linear(xb)

        loss = loss_func(pred, yb)

        loss.backward()

        opt.step() # Updating weights.

        opt.zero_grad()



print("Time to train linear model with basics:", round(time.time() - start_time),"seconds." ,"epochs:", epochs)
def accuracy(out, y_batch): 

    return (torch.argmax(out, dim=1)==y_batch).float().mean()
accuracy(model_linear(x_valid), y_valid)
model_linear_2layer = nn.Sequential(nn.Linear(sz,56), nn.ReLU(), nn.Linear(56,28), nn.ReLU(),nn.Linear(28,10))

opt = optim.SGD(model_linear_2layer.parameters(), lr=0.03)

loss_func = F.cross_entropy

epochs = 1

batch_size = 8
start_time = time.time()

for epoch in range(epochs):

    print("starting ")

    for i in range((num_rows-1)//batch_size + 1):

        start_i = i * batch_size

        end_i = start_i + batch_size

        xb = x_train[start_i:end_i]

        yb = y_train[start_i:end_i]

        pred = model_linear_2layer(xb)

        loss = loss_func(pred, yb)

        loss.backward()

        opt.step() # Updating weights.

        opt.zero_grad()

print(i)

print(end_i)

print("Time to train linear model with basics:", round(time.time() - start_time),"seconds." ,"epochs:", epochs)
accuracy(model_linear_2layer(x_valid), y_valid)