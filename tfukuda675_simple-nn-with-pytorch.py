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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from IPython.display import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models

from torchvision.utils import make_grid
import matplotlib.pyplot as plt
%matplotlib inline

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
train_df  = pd.read_csv("../input/digit-recognizer/train.csv") 
test_df = pd.read_csv("../input/digit-recognizer/test.csv") 
class MNISTDataset(Dataset):    
    def __init__(self, dataframe, 
            transform = transforms.Compose([transforms.ToPILImage(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=(0.5,), std=(0.5,))])
            ):
        df = dataframe
        self.n_pixels = 784
        
        if len(df.columns) == self.n_pixels:
            
            #self.X = df.values.reshape((-1,28,28))
            #print(self.X.shape)
            #self.X = self.X.astype(np.uint8)[:,:,:,None]
            #print(self.X.shape)
            #self.X = df.values.reshape((784,-1)).astype(np.uint8)[:,:,:,None]
            self.X = df.values.reshape((-1,28,28)).astype(np.uint8)
            self.y = None
        else:
            #self.X = df.iloc[:,1:].values.reshape((784,-1)).astype(np.uint8)
            #self.X = df.iloc[:,1:].astype(np.uint8)
            #self.X = df.iloc[:,1:].values.reshape((-1,28,28))
            #print(self.X.shape)
            #self.X = self.X.astype(np.uint8)[:,:,:,None]
            #print(self.X.shape)
            self.X = df.iloc[:,1:].values.reshape((-1,28,28)).astype(np.uint8)
            self.y = torch.from_numpy(df.iloc[:,0].values)
            
        self.transform = transform
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.transform(self.X[idx]), self.y[idx]
        else:
            return self.transform(self.X[idx])
model = nn.Sequential()
model.add_module('fc1', nn.Linear(28*28*1,100))
model.add_module('relu1', nn.ReLU())
model.add_module('fc2', nn.Linear(100, 100))
model.add_module('relu2', nn.ReLU())
model.add_module('fc3', nn.Linear(100,10))

print(model)
def train(train_loader, model, criterion, optimizer, epoch):
    model.train()

    for batch_idx , (data, target) in enumerate(train_loader):
        data = data.reshape((-1,784))
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (batch_idx + 1)% 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader), loss.data.item()))

def validate(val_loader, model, criterion):
    model.eval()
    loss = 0
    correct = 0
    
    for _, (data, target) in enumerate(val_loader):
        data = data.reshape((-1,784))
        output = model(data)
        
        loss += criterion(output, target).data.item()

        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        
        
    loss /= len(val_loader.dataset)
        
    print('\nOn Val set Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        loss, correct, len(val_loader.dataset),
        100.0 * float(correct) / len(val_loader.dataset)))
def split_dataframe(dataframe=None, fraction=0.9, rand_seed=1):
    df_1 = dataframe.sample(frac=fraction, random_state=rand_seed)
    df_2 = dataframe.drop(df_1.index)
    return df_1, df_2
train_transforms = transforms.Compose(
    [
     transforms.ToTensor(),
     transforms.Normalize(mean=(0.5,), std=(0.5,))])

val_test_transforms = transforms.Compose(
    [
     transforms.ToTensor(),
     transforms.Normalize(mean=(0.5,), std=(0.5,))])

optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

total_epoches = 20
batch_size = 64
for epoch in range(total_epoches):
    print("\nTrain Epoch {}".format(epoch))

    train_df_new, val_df = split_dataframe(dataframe=train_df, fraction=0.9, rand_seed=epoch)
    
    train_dataset = MNISTDataset(train_df_new, transform=train_transforms)
    val_dataset = MNISTDataset(val_df, transform=val_test_transforms)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size, shuffle=False)
    
    

    train(train_loader=train_loader, model=model, criterion=criterion, optimizer=optimizer, epoch=epoch)
    validate(val_loader=val_loader, model=model, criterion=criterion)
from torch import nn

class cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 64, kernel_size=5, padding=1)
        self.r1 = nn.ReLU(inplace=True)
        self.m1 = nn.MaxPool2d(2)
        self.c2 = nn.Conv2d(64, 16, kernel_size=3)
        self.r2 = nn.ReLU(inplace=True)
        self.m2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16*5*5, 120)
        self.r3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(120,84)
        self.r4 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(84, 10)
        self.softmax = nn.Softmax(dim=1)

        # weight init                                                                      
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = self.m1(self.r1(self.c1(x)))
        x = self.m2(self.r2(self.c2(x)))
        x = self.flatten(x)
        x = self.r3(self.fc1(x))
        x = self.r4(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x
model_cnn = cnn()
print(model_cnn)
def train_cnn(train_loader, model, criterion, optimizer, epoch):
    model.train()

    for batch_idx , (data, target) in enumerate(train_loader):
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (batch_idx + 1)% 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader), loss.data.item()))

def validate_cnn(val_loader, model, criterion):
    model.eval()
    loss = 0
    correct = 0
    
    for _, (data, target) in enumerate(val_loader):
        output = model(data)
        
        loss += criterion(output, target).data.item()

        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        
        
    loss /= len(val_loader.dataset)
        
    print('\nOn Val set Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        loss, correct, len(val_loader.dataset),
        100.0 * float(correct) / len(val_loader.dataset)))
optimizer = optim.Adam(model_cnn.parameters(), lr=0.01)

for epoch in range(total_epoches):
    print("\nTrain Epoch {}".format(epoch))

    train_df_new, val_df = split_dataframe(dataframe=train_df, fraction=0.9, rand_seed=epoch)
    
    train_dataset = MNISTDataset(train_df_new, transform=train_transforms)
    val_dataset = MNISTDataset(val_df, transform=val_test_transforms)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size, shuffle=False)
    
    

    train_cnn(train_loader=train_loader, model=model_cnn, criterion=criterion, optimizer=optimizer, epoch=epoch)
    validate_cnn(val_loader=val_loader, model=model_cnn, criterion=criterion)
