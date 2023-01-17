# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

from PIL import Image

import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

from sklearn import preprocessing

import torch

import torch.nn as nn

import torch.nn.functional as F

from torchvision import datasets, transforms

import time

# Any results you write to the current directory are saved as output.
!ls -l ../input/train/ | grep '^d'
img_files = os.listdir('../input/train/')

len(img_files)
img_files = list(filter(lambda x: x != 'train', img_files))

len(img_files)
def train_path(p): return f"../input/train/{p}"

img_files = list(map(train_path, img_files))
class CatDogDataset(Dataset):

    def __init__(self, image_paths, transform):

        super().__init__()

        self.paths = image_paths

        self.len = len(self.paths)

        self.transform = transform

        

    def __len__(self): return self.len

    

    def __getitem__(self, index): 

        path = self.paths[index]

        image = Image.open(path).convert('RGB')

        image = self.transform(image)

        label = 0 if 'cat' in path else 1

        return (image, label)
transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize((0.5,), (0.5,))

])

import random
random.shuffle(img_files)

train_files = img_files[:20000]

valid = img_files[20000:]
train_ds = CatDogDataset(train_files, transform)

train_dl = DataLoader(train_ds, batch_size=100)

len(train_ds), len(train_dl)
valid_ds = CatDogDataset(valid, transform)

valid_dl = DataLoader(valid_ds, batch_size=100)

len(valid_ds), len(valid_dl)
class CatAndDogNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size=(5, 5), stride=2, padding=1)

        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size=(5, 5), stride=2, padding=1)

        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=(3, 3), padding=1)

        self.fc1 = nn.Linear(in_features= 64 * 6 * 6, out_features=500)

        self.fc2 = nn.Linear(in_features=500, out_features=50)

        self.fc3 = nn.Linear(in_features=50, out_features=2)

        

        

    def forward(self, X):

        X = F.relu(self.conv1(X))

        X = F.max_pool2d(X, 2)

        

        X = F.relu(self.conv2(X))

        X = F.max_pool2d(X, 2)

        

        X = F.relu(self.conv3(X))

        X = F.max_pool2d(X, 2)

        

#         print(X.shape)

        X = X.view(X.shape[0], -1)

        X = F.relu(self.fc1(X))

        X = F.relu(self.fc2(X))

        X = self.fc3(X)

        

#         X = torch.sigmoid(X)

        return X
model = CatAndDogNet().cuda()

losses = []

accuracies = []

epoches = 5

start = time.time()

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

for epoch in range(epoches):

    epoch_loss = 0

    epoch_accuracy = 0

    for X, y in train_dl:

        X = X.cuda()

        y = y.cuda()

        preds = model(X)

        loss = loss_fn(preds, y)

        

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        

        accuracy = ((preds.argmax(dim=1) == y).float().mean())

        epoch_accuracy += accuracy

        epoch_loss += loss

        print('.', end='', flush=True)

        

    epoch_accuracy = epoch_accuracy/len(train_dl)

    accuracies.append(epoch_accuracy)

    epoch_loss = epoch_loss / len(train_dl)

    losses.append(epoch_loss)

    print("Epoch: {}, train loss: {:.4f}, train accracy: {:.4f}, time: {}".format(epoch, epoch_loss, epoch_accuracy, time.time() - start))





    with torch.no_grad():

        val_epoch_loss = 0

        val_epoch_accuracy = 0

        for val_X, val_y in valid_dl:

            val_X = val_X.cuda()

            val_y = val_y.cuda()

            val_preds = model(val_X)

            val_loss = loss_fn(val_preds, val_y)



            val_epoch_loss += val_loss            

            val_accuracy = ((val_preds.argmax(dim=1) == val_y).float().mean())

            val_epoch_accuracy += val_accuracy

        val_epoch_accuracy = val_epoch_accuracy/len(valid_dl)

        val_epoch_loss = val_epoch_loss / len(valid_dl)

        print("Epoch: {}, valid loss: {:.4f}, valid accracy: {:.4f}, time: {}\n".format(epoch, val_epoch_loss, val_epoch_accuracy, time.time() - start))
test_files = os.listdir('../input/test/')

len(test_files)
!ls -l ../input/test/ | grep '^d'
test_files = list(filter(lambda x: x != 'test', test_files))

len(test_files)
def test_path(p): return f"../input/test/{p}"

test_files = list(map(test_path, test_files))
test_files[:10]
class TestCatDogDataset(Dataset):

    def __init__(self, image_paths, transform):

        super().__init__()

        self.paths = image_paths

        self.len = len(self.paths)

        self.transform = transform

        

    def __len__(self): return self.len

    

    def __getitem__(self, index): 

        path = self.paths[index]

        image = Image.open(path).convert('RGB')

        image = self.transform(image)

        fileid = path.split('/')[-1].split('.')[0]

        return (image, fileid)
test_ds = TestCatDogDataset(test_files, transform)

test_dl = DataLoader(test_ds, batch_size=100)

len(test_ds), len(test_dl)
dog_probs = []

with torch.no_grad():

    for X, fileid in test_dl:

        X = X.cuda()

        preds = model(X)

        preds_list = F.softmax(preds, dim=1)[:, 1].tolist()

        dog_probs += list(zip(list(fileid), preds_list))

#         print(dog_probs)
len(dog_probs)
dog_probs.sort(key = lambda d: int(d[0]))
dog_probs[:10]
ids = list(map(lambda x: x[0], dog_probs))

probs = list(map(lambda x: x[1], dog_probs))
ids[:10], probs[:10]
output_df = pd.DataFrame({'id':ids,'label':probs})

output_df.to_csv('output.csv', index=False)
output_df.head()
# version4: Add validation set