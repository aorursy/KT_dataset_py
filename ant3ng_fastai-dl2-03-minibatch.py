%load_ext autoreload

%autoreload 2

%matplotlib inline

import numpy as np

import pandas as pd

import os

print(os.listdir("../input"))

import operator



#export

from pathlib import Path

from IPython.core.debugger import set_trace

from fastai import datasets

import pickle, gzip, math, torch, matplotlib as mpl

import matplotlib.pyplot as plt

from torch import tensor, nn, optim

from torch.utils.data import DataLoader, SequentialSampler, RandomSampler

import torch.nn.functional as F



def test(a,b,cmp,cname=None):

    if cname is None: cname=cmp.__name__

    assert cmp(a,b), f"{cname}:\n{a}\n{b}"



def test_eq(a,b): test(a,b,operator.eq, '==')

def near(a,b): return torch.allclose(a,b,1e-3,1e-5)

def test_near(a,b): test(a,b,near)

    



MNIST_URL='http://deeplearning.net/data/mnist/mnist.pkl'



def get_data():

    path = datasets.download_data(MNIST_URL, ext='.gz')

    with gzip.open(path, 'rb') as f:

        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')

    return map(tensor, (x_train, y_train, x_valid, y_valid))



def normalize(x,m,s): return (x-m)/s
x_train, y_train, x_valid, y_valid = get_data()



n,m = x_train.shape

c   = y_train.max()+1

nh  = 50

n,m,c,nh
class Model(nn.Module):

    def __init__(self, ni, nh, no):

        super().__init__()

        self.layers = [nn.Linear(ni,nh), nn.ReLU(), nn.Linear(nh,no)]

    def __call__(self, x):

        for l in self.layers: x = l(x)

        return x     



model = Model(m,nh,10)

pred  = model(x_train); pred.shape
def log_softmax(x): return (x.exp()/(x.exp().sum(-1,keepdim=True))).log()

sm_pred = log_softmax(pred); sm_pred.shape, sm_pred[:3]
def nll(inp, targ): return -inp[range(targ.shape[0]), targ].mean()

loss = nll(sm_pred, y_train); loss
# Followings all are same

def log_softmax(x): return x - x.exp().sum(-1, keepdim=True).log()

test_near(nll(log_softmax(pred), y_train), loss)



def log_softmax(x): return x - x.logsumexp(-1, keepdim=True)

test_near(nll(log_softmax(pred), y_train), loss)



test_near(F.nll_loss(F.log_softmax(pred, -1), y_train), loss)

test_near(F.cross_entropy(pred, y_train), loss)
loss_func = F.cross_entropy

def acc(out, yb): return (torch.argmax(out, -1)==yb).float().mean()
bs=64

xb, yb = x_train[:bs], y_train[:bs]

preds = model(xb)

loss_func(preds, yb), acc(preds, yb)
lr=0.5; epochs=1



for e in range(epochs):

    for i in range((n-1)//bs+1):

        xb = x_train[bs*i:bs*(i+1)]

        yb = y_train[bs*i:bs*(i+1)]

        loss = loss_func(model(xb), yb)

        

        loss.backward()

        with torch.no_grad():

            for l in model.layers:

                if hasattr(l, 'weight'):

                    l.weight -= l.weight.grad * lr

                    l.bias   -= l.bias.grad * lr

                    l.weight.grad.zero_()

                    l.bias.grad.zero_()



loss_func(model(xb), yb), acc(model(xb), yb)
class Model(nn.Module):

    def __init__(self, ni, nh, no):

        super().__init__()

        self.l1, self.l2 = nn.Linear(ni,nh), nn.Linear(nh,no)

    def __call__(self, x): return self.l2(F.relu(self.l1(x)))
model = Model(m,nh,10)

for name, l in model.named_children(): print(f"{name}: {l}")
model
def fit():

    for e in range(epochs):

        for i in range((n-1)//bs + 1):

            xb = x_train[bs*i:bs*(i+1)]

            yb = y_train[bs*i:bs*(i+1)]

            loss = loss_func(model(xb), yb)

            loss.backward()

            

            with torch.no_grad():

                for p in model.parameters(): p -= p.grad*lr

                model.zero_grad()

                

fit()

loss_func(model(xb), yb), acc(model(xb), yb)
class DummyModule():

    def __init__(self,ni,nh,no):

        self._modules = {}

        self.l1, self.l2 = nn.Linear(ni,nh), nn.Linear(nh,no)

    def __setattr__(self,k,v):

        if not k.startswith("_"): self._modules[k] = v

        super().__setattr__(k,v)

    def __repr__(self): return f'{self._modules}'

        

    def parameters(self):

        for l in self._modules.values():

            for p in l.parameters(): yield p



mdl = DummyModule(m,nh,10); mdl
[o.shape for o in mdl.parameters()]
layers = [nn.Linear(m,nh), nn.ReLU(), nn.Linear(nh,10)]



class Model(nn.Module):

    def __init__(self, layers):

        super().__init__()

        self.layers = layers

        for i, l in enumerate(self.layers): self.add_module(f'layer{i}', l)

    def __call__(self, x):

        for l in self.layers : x = l(x)

        return x



model = Model(layers); model
class SeqModel(nn.Module):

    def __init__(self, layers):

        super().__init__()

        self.layers = nn.ModuleList(layers)

    def __call__(self, x):

        for l in self.layers: x = l(x)

        return x



model = SeqModel(layers); model

model = nn.Sequential(nn.Linear(m,nh), nn.ReLU(), nn.Linear(nh,10)); model



fit()

loss_func(model(xb), yb), acc(model(xb), yb)
class Optimizer():

    def __init__(self, params, lr=0.5): 

        self.params, self.lr = list(params), lr

    def step(self):

        with torch.no_grad():

            for p in self.params: p -= p.grad*lr

    def zero_grad(self):

        for p in self.params: p.grad.data.zero_()
model = nn.Sequential(nn.Linear(m,nh), nn.ReLU(), nn.Linear(nh,10))

opt   = Optimizer(model.parameters())



for e in range(epochs):

    for i in range((n-1)//bs +1):

        xb = x_train[bs*i:bs*(i+1)]

        yb = y_train[bs*i:bs*(i+1)]

        loss = loss_func(model(xb), yb)

        loss.backward()

        opt.step()

        opt.zero_grad()



loss,accuracy = loss_func(model(xb),yb), acc(model(xb),yb); loss, accuracy
from torch import optim



def get_model():

    model = nn.Sequential(nn.Linear(m,nh), nn.ReLU(), nn.Linear(nh,10))

    return model, optim.SGD(model.parameters(), lr)



model, opt = get_model()

loss_func(model(xb), yb)
class Dataset():

    def __init__(self,x,y): self.x,self.y = x,y

    def __len__(self): return len(self.x)

    def __getitem__(self, i): return self.x[i], self.y[i]

    

train_ds, valid_ds = Dataset(x_train, y_train),Dataset(x_valid, y_valid)

assert len(train_ds) == len(x_train)

assert len(valid_ds) == len(x_valid)



for e in range(epochs):

    for i in range((n-1)//bs +1):

        xb,yb = train_ds[i*bs:(i+1)*bs]

        loss_func(model(xb), yb).backward()

        opt.step()

        opt.zero_grad()



loss,accuracy = loss_func(model(xb),yb), acc(model(xb),yb); loss,accuracy
class DataLoader():

    def __init__(self,ds,bs): self.ds,self.bs = ds,bs

    def __iter__(self):

        for i in range(0, len(self.ds), self.bs): yield self.ds[i:i+self.bs]



train_dl = DataLoader(train_ds, bs)

valid_dl = DataLoader(valid_ds, bs)

xb,yb = next(iter(valid_dl))

plt.imshow(xb[0].view(28,-1)); yb[0]
model, opt = get_model()



def fit():

    for e in range(epochs):

        for xb,yb in train_dl:

            loss_func(model(xb),yb).backward()

            opt.step(); opt.zero_grad()

fit()

loss,accuracy = loss_func(model(xb),yb), acc(model(xb),yb); loss,accuracy
class Sampler():

    def __init__(self,ds,bs,shuffle=False):

        self.n,self.bs,self.shuffle = len(ds),bs,shuffle

    def __iter__(self):

        self.idxs = torch.randperm(self.n) if self.shuffle else torch.arange(self.n)

        for i in range(0,self.n,self.bs): yield self.idxs[i:i+self.bs]

            

small_ds = Dataset(*train_ds[:10])



s1, s2 = Sampler(small_ds,3), Sampler(small_ds,3,True)

[o for o in s1], [o for o in s2]
for oo in s1:

    for o in oo:

        print(o)
for oo in s1:

    for o in oo:

        print(o)

    break
for oo in s1:

    for o in oo:

        print(o)

        break
def collate(b):

    xs,ys = zip(*b)

    return torch.stack(xs), torch.stack(ys)



class DataLoader():

    def __init__(self,ds,sampler,collate_fn=collate):

        self.ds,self.sampler,self.collate_fn=ds,sampler,collate_fn

    def __iter__(self):

        for s in self.sampler: yield self.collate_fn([self.ds[i] for i in s])

            

train_samp = Sampler(train_ds,bs,True)

valid_samp = Sampler(valid_ds,bs)

train_dl   = DataLoader(train_ds,train_samp)

valid_dl   = DataLoader(valid_ds,valid_samp)

model,opt = get_model()

fit()

loss_func(model(xb),yb), acc(model(xb),yb)
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler



train_dl = DataLoader(train_ds,bs,sampler=RandomSampler(train_ds),collate_fn=collate)

valid_dl = DataLoader(valid_ds,bs,sampler=SequentialSampler(valid_ds),collate_fn=collate)

model,opt = get_model()

fit(); loss_func(model(xb),yb), acc(model(xb),yb)
def fit(epochs, model, loss_func, opt, train_dl, valid_dl):

    for e in range(epochs):

        model.train()

        for xb,yb in train_dl:

            loss_func(model(xb),yb).backward()

            opt.step(); opt.zero_grad()

        

        model.eval()

        with torch.no_grad():

            tot_loss,tot_acc=0.,0.

            for xb,yb in valid_dl:

                pred = model(xb)

                tot_loss += loss_func(pred,yb)

                tot_acc  += acc(pred,yb)

        nv = len(valid_dl)

        print(e, tot_loss/nv, tot_acc/nv)

    return tot_loss/nv, tot_acc/nv



def get_dls(train_ds, valid_ds, bs, **kwargs):

    return (DataLoader(train_ds,bs  ,True ,**kwargs),

            DataLoader(valid_ds,bs*2,      **kwargs))



model,opt = get_model()

loss,accu = fit(5,model,loss_func,opt,*get_dls(train_ds,valid_ds,bs))