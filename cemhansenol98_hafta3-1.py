import torch

import torch.nn as nn

from PIL import Image

from pathlib import Path

import numpy as np

from matplotlib import pyplot as plt

import os

import math

import pandas as pd
PATH = Path('../input/mnistasjpg/trainingSet')
image = Image.open("../input/mnistasjpg/trainingSet/trainingSet/0/img_1.jpg")
plt.imshow(image, cmap='gray')
np.array(image).shape
df = pd.DataFrame(np.array(image))
df.style.set_properties().background_gradient("Greys")
list(PATH.iterdir())
Path.ls = lambda x: list(x.iterdir())
(PATH/"trainingSet").ls()
(PATH/"trainingSet/5").ls()
Image.open((PATH/"trainingSet/5").ls()[0])
five = [torch.tensor(np.array(Image.open(img)), dtype = torch.float32) for img in (PATH/"trainingSet/5").ls()]
seven = [torch.tensor(np.array(Image.open(img)), dtype = torch.float32) for img in (PATH/"trainingSet/7").ls()]
plt.imshow(five[0], cmap = "gray");
plt.imshow(seven[0], cmap = "gray");
five_stacked = torch.stack(five) / 255
five_stacked.shape
seven_stacked = torch.stack(seven) / 255
seven_stacked.shape
avr5 = five_stacked.mean(0)
avr7 = seven_stacked.mean(0)
plt.imshow(avr5, cmap = "gray");
plt.imshow(avr7, cmap = "gray");
sample_5 = five_stacked[20]
plt.imshow(sample_5, cmap = "gray");
dist_to_5 = ((sample_5 - avr5)**2).mean().sqrt()
dist_to_7 = ((sample_5 - avr7)**2).mean().sqrt()
dist_to_5.item()
dist_to_7.item()
def distance(a, b):

    return ((a - b)**2).mean((-1,-2)).sqrt()
distance(sample_5, avr5)
valid_dist_5 = distance(five_stacked, avr5)

valid_dist_5, valid_dist_5.shape
def is_five(x):

    return distance(x, avr5) < distance(x, avr7)
is_five(sample_5)
accuracy_5 = is_five(sample_5).float().mean()

accuracy_5
x = torch.tensor(2.).requires_grad_()
def f(x):

    return x**2
grad = f(x)
grad.backward()
x.grad
labels = {1:"Five", 0:"Seven"}
train_x = torch.cat([five_stacked, seven_stacked]).view(-1, 28*28)
train_y = torch.tensor([1] * len(five) + [0] * len(seven))
train_y
ds_train = list(zip(train_x,train_y))
ds_train[0]
plt.imshow(ds_train[0][0].view(28,28),cmap="gray")
labels[ds_train[0][1].item()]
plt.imshow(ds_train[0][0].view(28,28), cmap="gray");
train_x.shape
def init(size):

    return torch.randn(size, dtype=torch.float32).requires_grad_()
w = init((28*28,1))
w.shape
b = init(1)
train_x[0].shape
w.shape
(train_x[0] * w.T).sum() + b
def linear_layer(xb):

    return xb @ w + b
preds = linear_layer(train_x)
preds.shape
def accuracy(preds, actuals):

    return ((preds > 0.0).float() == actuals).float().mean().item()
accuracy(preds, train_y)
w[0] = w[0] * 1.0001
preds = linear_layer(train_x)
accuracy(preds, train_y)
def loss(preds, targets):

    return torch.where(targets==1, 1-preds, preds).mean()
def sigmoid(x):

    return 1/(1+torch.exp(-x))
def loss_func(preds, targets):

    preds = preds.sigmoid()

    return torch.where(targets==1, 1-preds, preds).mean()
class DataLoader():

    def __init__(self, ds, bs): 

        self.ds, self.bs = ds, bs

    def __iter__(self):

        n = len(self.ds)

        l = torch.randperm(n)



        

        for i in range(0, n, self.bs): 

            idxs_l = l[i:i+self.bs]

            yield self.ds[idxs_l]
w = init((28*28,1))
b = init(1)
train_dl = DataLoader((ds_train), bs = 2)
len(ds_train)
xb, yb = next(iter(train_dl))