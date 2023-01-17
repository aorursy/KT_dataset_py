import torch

import numpy as np

import torchvision

from torchvision.datasets import MNIST

from torchvision.transforms import ToTensor

from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data.dataloader import DataLoader
dataset = MNIST(root='data/', download=True, transform=ToTensor())
def split_indices(n, val_pct):

    n_val = int(n*val_pct)

    idx = np.random.permutation(n)

    return idx[n_val:], idx[:n_val]
train_indices, val_indices = split_indices(len(dataset), 0.2)
batch_size = 100

train_sampler = SubsetRandomSampler(train_indices)

train_dl = DataLoader(dataset, batch_size, sampler=train_sampler)



val_sampler = SubsetRandomSampler(val_indices)

valid_dl = DataLoader(dataset, batch_size, sampler=val_sampler)
import torch.nn as nn

import torch.nn.functional as f
class MnistModel(nn.Module):

    ##feedforwarding nerual network

    def __init__(self, in_size, hidden_size, out_size):

        super().__init__()

        #hidden layer

        self.linear1 = nn.Linear(in_size, hidden_size)

        #output layer

        self.linear2 = nn.Linear(hidden_size, out_size)

        

    def forward(self, xb):

        xb = xb.view(xb.size(0), -1)

        out = self.linear1(xb)

        ##activation funtion rectified linear unit : just ignoring the minus

        out = f.relu(out) 

        out = self.linear2(out)

        return out

    

        

        
input_size = 784

num_class = 10

model = MnistModel(input_size, 32, num_class)
for t in model.parameters():

    print(t.shape)
for images, label in train_dl:

    output = model(images)

    loss = f.cross_entropy(output, label)

    print('Loss:', loss.item())

    print(output.shape)

    break
!pip install jovian
import jovian

jovian.commit()