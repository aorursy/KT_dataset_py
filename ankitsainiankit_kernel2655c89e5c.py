# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import torch

import torch.nn as nn

import torchvision

import torch.utils as utils

from torch.utils.data import Dataset, DataLoader

from torch import optim

import cv2

from torchvision import transforms

from torch.utils.data.sampler import SubsetRandomSampler

from torch.nn import CrossEntropyLoss

import torch.nn.functional as F

from tqdm.notebook import tqdm
model = torchvision.models.resnet34(True)

#freezing

for parameter in model.parameters():

    parameter.requires_grad = False
#getting the num_output of last avgpool layer to replace the last fully connected with

#7 class network

resnet_out_classes = model.fc.in_features
model
#replacing last layer

class SoftmaxPrediction(nn.Module):

    def __init__(self,in_features,pred_classes):

        super(SoftmaxPrediction,self).__init__()

        self.l1 = nn.Linear(in_features,pred_classes,bias=True)

    def forward(self,x):

        x = self.l1(x)

        return F.softmax(x,dim=1)



model.fc = SoftmaxPrediction(resnet_out_classes,8)
val_part = 0.2

train_transforms = transforms.Compose([

    transforms.Resize([224,224]),

    transforms.RandomHorizontalFlip(),

    transforms.RandomPerspective(),

    transforms.RandomRotation(30),

    transforms.RandomGrayscale(),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406],

                        std=[0.229, 0.224, 0.225])

])

valid_transforms = transforms.Compose([

    transforms.Resize([224,224]),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406],

                        std=[0.229, 0.224, 0.225])

])

train_data = torchvision.datasets.ImageFolder('../input/natural-images/natural_images',transform = train_transforms)

valid_data = torchvision.datasets.ImageFolder('../input/natural-images/natural_images',transform = valid_transforms)
indices = list(range(len(train_data)))

split = int(len(train_data)*0.2)

np.random.seed(42)

np.random.shuffle(indices)

val_split, train_split = indices[:split],indices[split:]
train_loader = DataLoader(train_data,sampler = SubsetRandomSampler(train_split),batch_size = 64,pin_memory = True)

val_loader = DataLoader(valid_data,sampler = SubsetRandomSampler(val_split), batch_size = 128,pin_memory = True)
epochs=4

optimizer = optim.AdamW(model.parameters())

loss_fn = CrossEntropyLoss()

loss_fn = loss_fn.cuda()

model = model.cuda()

scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr = 1e-3, epochs = epochs,steps_per_epoch = len(train_loader))

for epoch in tqdm(range(epochs),leave = False):

    print(f'\n ...... {epoch+1}/{epochs} ...... \n')

    train_loss = 0

    model.train()

    for batch in tqdm(train_loader,leave = False):

        optimizer.zero_grad()

        inputs, labels = batch[0].cuda(), batch[1].cuda()

        pred = model(inputs)

        loss = loss_fn(pred,labels)

        

        loss.backward()

        optimizer.step()

        train_loss+= loss

        scheduler.step()

    print(f'train_loss = {train_loss/(len(train_loader)*train_loader.batch_size)}')

    #validation

    model.eval()

    val_loss = 0

    correct_classify = 0

    with torch.no_grad():

        for batch in tqdm(val_loader,leave=False):

            inputs,labels = batch[0].cuda(), batch[1].cuda()

            pred = model(inputs)

            loss = loss_fn(pred,labels)

    #         print(pred,labels)

    #         break

            correct_classify += torch.sum(pred.max(1)[1] == labels).item()

            val_loss += loss

    print(f'val_loss = {val_loss/(len(val_loader)*val_loader.batch_size)}')

    print(f'accuracy = {correct_classify/(len(val_loader)*val_loader.batch_size)}')
for parameters in model.parameters():

    parameters.requires_grad = True
train_loader = DataLoader(train_data,sampler = SubsetRandomSampler(train_split),batch_size = 16,pin_memory = True)

val_loader = DataLoader(valid_data,sampler = SubsetRandomSampler(val_split), batch_size = 32,pin_memory = True)
epochs=4

optimizer = optim.AdamW([{'params':list(model.conv1.parameters())+list(model.bn1.parameters())+list(model.layer1.parameters())+list(model.layer2.parameters())},

                        {'params': list(model.layer3.parameters())+list(model.layer4.parameters())},

                        {'params': list(model.fc.parameters())}])

loss_fn = CrossEntropyLoss()

loss_fn = loss_fn.cuda()

model = model.cuda()

scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr = [1e-9,3e-6,1e-4], epochs = epochs,steps_per_epoch = len(train_loader))

for epoch in tqdm(range(epochs),leave=False):

    print(f'\n ...... {epoch+1}/{epochs} ...... \n')

    train_loss = 0

    model.train()

    for batch in tqdm(train_loader,leave = False):

        optimizer.zero_grad()

        inputs, labels = batch[0].cuda(), batch[1].cuda()

        pred = model(inputs)

        loss = loss_fn(pred,labels)

        

        loss.backward()

        optimizer.step()

        train_loss+= loss

        scheduler.step()

    print(f'train_loss = {train_loss/(len(train_loader)*train_loader.batch_size)}')

    #validation

    model.eval()

    val_loss = 0

    correct_classify = 0

    with torch.no_grad():

        for batch in tqdm(val_loader,leave = False):

            inputs,labels = batch[0].cuda(), batch[1].cuda()

            pred = model(inputs)

            loss = loss_fn(pred,labels)

    #         print(pred,labels)

    #         break

            correct_classify += torch.sum(pred.max(1)[1] == labels).item()

            val_loss += loss

    print(f'val_loss = {val_loss/(len(val_loader)*val_loader.batch_size)}')

    print(f'accuracy = {correct_classify/(len(val_loader)*val_loader.batch_size)}')