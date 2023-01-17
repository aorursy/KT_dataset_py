# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/fruits-360_dataset/fruits-360/Training"))



# Any results you write to the current directory are saved as output.
import torch

import torch.nn as nn 

import torchvision.datasets as datasets

import torchvision.transforms as transforms

from torch.utils.data import DataLoader

import torchvision.models as models



batch_size = 128

learning_rate = 1e-3

transforms = transforms.Compose(

[

    transforms.ToTensor()

])

train_dataset = datasets.ImageFolder(root='../input/fruits-360_dataset/fruits-360/Training', transform=transforms)

test_dataset = datasets.ImageFolder(root='../input/fruits-360_dataset/fruits-360/Test', transform=transforms)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = models.vgg16_bn(pretrained=True)

model = model.to(device)

critirion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



from tqdm import tqdm 

n_epoch = 10

for epoch in range(n_epoch):

    for i, (images, labels) in enumerate(tqdm(train_dataloader)):

        images = images.to(device)

        labels = labels.to(device)

        outputs = model(images)

        loss = critirion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    print('Loss: {}'.format(loss.item()))

        
model.eval()

with torch.no_grad():

    correct = 0

    total = 0

    for (images, labels) in test_dataloader:

        images = images.to(device)

        labels = labels.to(device)

        outputs = model(images)

        _, predict = torch.max(outputs.data, 1)

        total+=labels.size(0)

        correct+=(predict==labels).sum().item()

    

    print('Test accuracy: {}'.format(100*correct/total))