# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
os.listdir("../input/fruits-360_dataset/fruits-360/")
import numpy as np

from torchvision import datasets

import torchvision.transforms as transforms

import torch

import torch.nn as nn

import torch.nn.functional as F

import matplotlib.pyplot as plt



batch_size = 30

transform = transforms.Compose([transforms.ToTensor(),

 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_data = datasets.ImageFolder(root = '../input/fruits-360_dataset/fruits-360/Training',transform= transform)

test_data = datasets.ImageFolder(root='../input/fruits-360_dataset/fruits-360/Test',transform = transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size= batch_size)

test_loader= torch.utils.data.DataLoader(test_data,batch_size= batch_size)

dataiter = iter(train_loader)

images,labels = dataiter.next()

images= images.numpy()

len(os.listdir("../input/fruits-360_dataset/fruits-360/Training/"))
model=nn.Sequential(nn.Linear(30000,512),

                   nn.ReLU(),

                   nn.Linear(512,256),

                   nn.ReLU(),

                   nn.Linear(256,114),

                   nn.ReLU(),

                   nn.LogSoftmax(dim=1))

criterion=nn.NLLLoss()

optimizer=optim.Adam(model.parameters(),lr=0.003)

epochs=20

print(model)

model.train()

for epoch in range(n_epochs):

    train_loss = 0.0

    for data, target in train_loader:

        optimizer.zero_grad()

        output= model(data)

        loss = criterion(output, target)

        loss.backword()

        optimizer.step()

        train_loss+= loss.item()*data.size(0)