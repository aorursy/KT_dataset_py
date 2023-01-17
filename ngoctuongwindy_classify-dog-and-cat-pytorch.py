from __future__ import print_function, division



import os

import glob

import time

print(os.listdir('../input'))

import numpy as np

from PIL import Image

import matplotlib.pyplot as plt



import torch

import torch.nn as nn

import torch.nn.functional as F

from torchvision import datasets, transforms

from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader, Dataset
train1 = os.listdir('../input/training_set/training_set/cats/')

train2 = os.listdir('../input/training_set/training_set/dogs/')

len(train1+train2)
class CatDogDataset(Dataset):

    def __init__(self, path, transform = None):

        self.classes = os.listdir(path) # list classes in Dataset: cat and dog

        self.path = [f"{path}/{className}" for className in self.classes] # get class path: dog and cat 

        self.file_list = [glob.glob(f"{x}/*") for x in self.path] # get images list of each class

        self.transform = transform

        

        # Create list image informations: id_classes, name class, file name

        file = []

        for i, className in enumerate(self.classes):

            for fileName in self.file_list[i]:

                file.append([i, className, fileName]) 

                # [0, cat, cat1.jpg]

                # [1, dog, dog1.jpg]

        self.file_list = file

        file = None

    

    # get lenght of data

    def __len__(self):

        return len(self.file_list)

    

    # return image and label

    def __getitem__(self, idx):

        fileName = self.file_list[idx][2]

        #print(fileName)

        category = self.file_list[idx][0]

        img = Image.open(fileName)

        

        if self.transform:

            img = self.transform(img)

        return img, category # reshape tensor

        
## Transform data

image_size = (100, 100)

# transforms is common image transformations, specially is image transformation tpo Tensor 

transform = transforms.Compose([

    transforms.Resize(image_size),

    transforms.ToTensor(),

])
# path

path = '../input/training_set/training_set'

path_valid = '../input/test_set/test_set/'

dataset = CatDogDataset(path, transform=transform)

valid_data = CatDogDataset(path_valid, transform=transform)

## Nowly, dataset are transformed to Tensor with pytorch

shuffle = True

batch_size = 100

num_workers = 0

data_Loader = DataLoader(dataset=dataset,

                        shuffle=shuffle,

                        batch_size=batch_size,

                        num_workers=num_workers)

valid_dataset = DataLoader(valid_data, batch_size=100)

len(data_Loader)

len(valid_dataset)
class CustomModel(nn.Module):

    

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5,5), stride=2, padding=1)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5,5), stride=2, padding=1)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding=1)

        

        self.batch_norm = nn.BatchNorm2d(64)

        self.dropout = nn.Dropout(0.5)

        

        self.fc1 = nn.Linear(in_features=64*2*2, out_features=200)

        self.fc2 = nn.Linear(in_features=200, out_features=20)

        self.fc3 = nn.Linear(in_features=20, out_features=2)

        

        

        

    def forward(self, x):

        x = F.relu(self.conv1(x))

        #print(x.shape)

        x = F.max_pool2d(x,2)

        #print(x.shape)

        

        x = F.relu(self.conv2(x))

        #print(x.shape)

        x = F.max_pool2d(x,2)

        

        x = F.relu(self.batch_norm(self.conv3(x)))

        x = F.max_pool2d(x, 2)

        

        #print(x.shape)

        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))

        x = self.dropout(x)

        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        

        return x    

    

model = CustomModel()

model = model.cuda()

print(model)

        
losses = []

acc = []

epoches = 10

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

start = time.time()



for epoch in range(epoches):

    

    epoch_loss = 0

    epoch_acc = 0

    

    for X,Y in data_Loader:

        X = X.cuda()

        Y = Y.cuda()

        

        preds = model(X)

        loss = loss_fn(preds, Y)

        

        # set gradient to zero before training

        # To update parameters

        # forward + backward + optimizer

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        

        # compute accuracy for each data batch

        accuracy = ((preds.argmax(dim=1) == Y).float().mean())

        # compute sum acc on all data

        epoch_acc += accuracy

        # compute sum loss on all data

        epoch_loss += loss

        print('.', end='', flush=True)

        

    print('\n')

    # average acc on all data

    epoch_acc = epoch_acc/len(data_Loader)

    acc.append(epoch_acc)

    # average loss on all data

    epoch_loss = epoch_loss/len(data_Loader)

    losses.append(epoch_loss)

    

    print("Epoch: {}, train loss: {:.4f}, train acc: {:.4f}, time: {}".format(

         epoch + 1, epoch_loss, epoch_acc, time.time() - start))

    

    ## Valid

    with torch.no_grad():

        valid_acc = 0

        valid_loss = 0

        for X_valid, Y_valid in valid_dataset:

            X_valid = X.cuda()

            Y_valid = Y.cuda()

            

            preds_valid = model(X_valid)

            loss_valid = loss_fn(preds_valid,Y_valid)

            

            accuracy_valid = ((preds_valid.argmax(dim=1) == Y_valid).float().mean())

            valid_acc += accuracy_valid

            valid_loss += loss_valid

            

        valid_acc = valid_acc/len(valid_dataset)

        valid_loss = valid_loss/len(valid_dataset)

        

        print("Epoch: {}, train loss: {:.4f}, train acc: {:.4f}, time: {}".format(

            epoch + 1, valid_loss, valid_acc, time.time()- start))

    

torch.save(model.state_dict(), '../output')