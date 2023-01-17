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
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
normal = Path('../input/surface-crack-detection/Negative')
cracks = Path('../input/surface-crack-detection/Positive')

positive = [(o,0) for o in cracks.iterdir()]
negative = [(o,1) for o in normal.iterdir()]
n = pd.DataFrame(negative, columns=["filepath","cracks"])
p = pd.DataFrame(positive, columns=["filepath","cracks"])
path_n=n["filepath"]
path_p=p["filepath"]
n_images = [torch.tensor(np.array(Image.open(img)), dtype = torch.float32) for img in path_n[:500]]
p_images=[torch.tensor(np.array(Image.open(img)), dtype = torch.float32) for img in path_p[:500]]

n_stacked = torch.stack(n_images)/ 255
p_stacked = torch.stack(p_images) / 255
n_stacked.shape
p_stacked.shape
avr_n = n_stacked.mean(0)
avr_p = p_stacked.mean(0)
plt.imshow(avr_p, cmap = "gray");
sample_p = p_stacked[20]
sample_n=n_stacked[15]
plt.imshow(sample_p, cmap = "gray");
dist_to_p = ((sample_p - avr_p)**2).mean().sqrt()

dist_to_p.item()
def distance(a, b):
    return ((a - b)**2).mean().sqrt().mean()

distance(sample_p, avr_p)
def is_cracked(x):
    return distance(x, avr_p) < distance(x, avr_n)
is_cracked(sample_p)
is_cracked(sample_n)
valid_n_images = [torch.tensor(np.array(Image.open(img)), dtype = torch.float32) for img in path_n[500:600]]
valid_p_images=[torch.tensor(np.array(Image.open(img)), dtype = torch.float32) for img in path_p[500:600]]

valid_p_stacked = torch.stack(valid_p_images)/ 255
valid_n_stacked = torch.stack(valid_n_images) / 255
is_cracked(valid_p_stacked).float().mean()
# To check labels and mappings later. It has no practical usage
labels = {1:"Normal", 0:"Cracked"}
train_x = torch.cat([n_stacked,p_stacked]).view(-1, 227*227)
train_y = torch.tensor([1] * len(n_images) + [0] * len(p_images))
train_y
train_x.shape, train_y.shape
train_y.unsqueeze_(-1)
train_y.shape
valid_x = torch.cat([ valid_n_stacked,valid_p_stacked]).view(-1, 227*227)
valid_y = torch.tensor([1] * len(valid_n_images) + [0] * len(valid_p_images))

valid_x.shape
valid_y.shape
valid_y.unsqueeze(0)
valid_y.unsqueeze(0).shape
valid_y.unsqueeze(1)
ds_train = list(zip(train_x, train_y))
ds_valid = list(zip(valid_x, valid_y))
plt.imshow(ds_train[821][0].view(227,227), cmap="gray");
labels[ds_train[821][1].item()]

class Dataset():
    def __init__(self, x, y): 
        self.x,self.y = x,y
    def __len__(self): 
        return len(self.x)
    def __getitem__(self, i): 
        return self.x[i],self.y[i]
ds_train = Dataset(train_x, train_y)
ds_valid = Dataset(valid_x, valid_y)
train_x.shape
valid_x.shape
def linear_layer(xb):
    return xb @ w + b
def init(size):
    return torch.randn(size, dtype=torch.float32).requires_grad_()
w = init((227*227,1))
b = init(1)
preds = linear_layer(train_x)
def accuracy(preds, actuals):
    return ((preds > 0.0).float() == actuals).float().mean().item()
accuracy(preds.mean(), train_y)
def loss_func(preds, targets):
    preds = preds.sigmoid()
    return torch.where(targets==1, 1-preds, preds).mean()
def sigmoid(x):
    return 1/(1+torch.exp(-x))
class DataLoader():
    def __init__(self, ds, bs): 
        self.ds, self.bs = ds, bs
    def __iter__(self):
        n = len(self.ds)
        l = torch.randperm(n)

        
        for i in range(0, n, self.bs): 
            idxs_l = l[i:i+self.bs]
            yield self.ds[idxs_l]
train_dl = DataLoader(ds_train, bs = 512)
valid_dl = DataLoader(ds_valid, bs = 512)
def calculate_grad(model, xb, yb):
    preds = model(xb)
    loss = loss_func(preds, yb)
    loss.backward()
    
def train(model, epochs=5, valid_epoch=5):
    for epoch in range(epochs):
        for xb, yb in train_dl:
            calculate_grad(model, xb, yb)
            optim.step()
            optim.zero_grad()
        
        if epoch % valid_epoch == 0:
            print(validation_acc(model))
import torch.optim as opt
lr = 1
params = w, b
model_1 = nn.Linear(28*28, 1)
optim = opt.SGD(model_1.parameters(), lr)
train(model_1, 20, 2)




