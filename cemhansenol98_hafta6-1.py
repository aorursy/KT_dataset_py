import torch

import torch.nn as nn

from PIL import Image

from pathlib import Path

import numpy as np

from matplotlib import pyplot as plt



import os

import math

import torch.nn.functional as F
PATH= Path("../input/kodluyoruz-mist/data/mnist")
def log_softmax(x): 

    return (x.exp()/(x.exp().sum(-1,keepdim=True)) + 1e-20).log()



def nll(preds, actuals): 

    return -preds[range(actuals.shape[0]), actuals].mean()



def validation_acc(model):

    return torch.stack([accuracy(model(xb), yb) for xb, yb in valid_dl]).mean().item()



def accuracy(preds, yb): 

    return (torch.argmax(preds, dim=1, keepdim = True)==yb).float().mean()



def loss_func(preds, targets):

    preds = log_softmax(preds)

    return nll(preds, targets)



def train(model, epochs=5, valid_epoch=5):

    for epoch in range(epochs):

        for xb, yb in train_dl:

            

            preds = model(xb)

            loss = loss_func(preds, yb.squeeze())

            loss.backward()

            optim.step()

            optim.zero_grad()

        

        if epoch % valid_epoch == 0:

            print(validation_acc(model))

            

def _get_files(p, fs, extensions = None):

    p = Path(p) # to support / notation

    res = [p/f for f in fs if not f.startswith(".") 

           and ((not extensions) or f'.{f.split(".")[-1].lower()}' in extensions)]

    return res



class Dataset():

    def __init__(self, x, y): 

        self.x,self.y = x,y

    def __len__(self): 

        return len(self.x)

    def __getitem__(self, i): 

        return self.x[i].view(-1,1,28,28).cuda(),self.y[i].cuda()





class DataLoader():

    def __init__(self, ds, bs): 

        self.ds, self.bs = ds, bs

    def __iter__(self):

        n = len(self.ds)

        l = torch.randperm(n)



        

        for i in range(0, n, self.bs): 

            idxs_l = l[i:i+self.bs]

            yield self.ds[idxs_l]

            

def create_ds_from_file(src):

    imgs, labels = [], []

    

    for label in range(10):

        path = src/str(label)

        print(path)

        t = [o.name for o in os.scandir(path)]

        t = _get_files(path, t, extensions = [".jpg", ".png"])

        for e in t:

            l = [np.array(Image.open(e)).reshape(28*28)]

            imgs += l

        labels += ([label] * len(t))

    return torch.tensor(imgs,  dtype=torch.float32), torch.tensor(labels, dtype=torch.long).view(-1,1)



def stats(x):

    print(f"Mean: {x.mean()}, Std: {x.std()}")
trn_x, trn_y = create_ds_from_file(PATH/"train")
val_x,val_y = create_ds_from_file(PATH/"validation")
def multiple_normalization(train_or_valid_X):

    for i in range(len(train_or_valid_X)):

        train_or_valid_X[i] =  (train_or_valid_X[i] - torch.min(train_or_valid_X[i])) / (torch.max(train_or_valid_X[i]) - torch.min(train_or_valid_X[i]))

    return train_or_valid_X
train_norm = multiple_normalization(trn_x)
train_norm.shape
valid_norm = multiple_normalization(val_x)
valid_norm.shape
train_ds = Dataset(train_norm, trn_y)

valid_ds = Dataset(valid_norm, val_y)
train_dl = DataLoader(train_ds, 256)

valid_dl = DataLoader(valid_ds, 256)
x, y = next(iter(train_dl))
x.shape
y.shape
class Func(nn.Module):

    def __init__(self, func):

        super().__init__()

        self.func = func



    def forward(self, x): 

        return self.func(x)

    

def flatten(x):      

    return x.view(x.shape[0], -1)



def print_t(x):      

    print(x.shape)

    return x
def kaiming_uniform(x, a):

    n = x[0].shape.numel()

    std = gain(a) / math.sqrt(n)

    bound = math.sqrt(3.) * std

    x.data.uniform_(-bound, bound)

    

def kaiming_norm(x, a):

    n = x[0].shape.numel()

    std = gain(a) / math.sqrt(n)

    x.data = x.data.normal_() * std
def gain(a):

    return math.sqrt(2.0 / (1 + a**2))



def stats(x):

    print(f"Mean: {x.mean()}, Std: {x.std()}")
model = nn.Sequential(

        nn.Conv2d(1, 8, 5, padding=2,stride=2), nn.ReLU(), #14

        nn.Conv2d(8, 32, 3, padding=1,stride=2), nn.ReLU(), # 7

        nn.Conv2d(32, 32, 3, padding=1,stride=2), nn.ReLU(), # 4

        nn.Conv2d(32, 32, 3, padding=1,stride=2), nn.ReLU(), # 2

        #Func(print_t),

        nn.AdaptiveAvgPool2d(1),

        Func(flatten),

        nn.Linear(32,10)

).cuda()
for l in model:

    if isinstance(l, nn.Conv2d):

        kaiming_uniform(l.weight, a = 0)

        l.bias.data.zero_()
temp = model(x)
stats(temp)
optim = torch.optim.SGD(model.parameters(), lr=0.02, weight_decay=1e-3)
train(model,100,10)