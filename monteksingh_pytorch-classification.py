# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/dataset/dataset_updated"))

# Any results you write to the current directory are saved as output.
import torch
from torchvision import datasets, transforms
import nonechucks as nc
from torch import optim, nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.Resize((128,128)) ,
                                transforms.ToTensor(),
                                transforms.Normalize( (0.5,0.5,0.5),(0.5,0.5,0.5) ) ])
traindata = datasets.ImageFolder(root= "../input/dataset/dataset_updated/training_set", 
                                 transform = transform)
traindata = nc.SafeDataset(traindata)
trainloader = nc.SafeDataLoader(traindata, batch_size=64, shuffle=True)


testdata = datasets.ImageFolder(root= "../input/dataset/dataset_updated/validation_set", 
                                 transform = transform)
testdata = nc.SafeDataset(testdata)
testloader = nc.SafeDataLoader(testdata, batch_size=64, shuffle=True)

images, label = next(iter(trainloader))
print(images.shape)
print(label)


class Network(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.con1 = nn.Conv2d(3, 16, kernel_size = 3, stride = 1, padding = 1)
        self.con2 = nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding = 1)
        self.con3 = nn.Conv2d(32, 48, kernel_size = 3, stride = 1, padding = 1)
        self.con4 = nn.Conv2d(48, 64, kernel_size = 3, stride = 1, padding = 1)
        self.con5 = nn.Conv2d(64, 72, kernel_size = 3, stride = 1, padding = 1)
        self.con6 = nn.Conv2d(72, 96, kernel_size = 3, stride = 1, padding = 1)
        
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
        #self.pool2 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        self.fc1 = nn.Linear(384, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 5)
        
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        #print(x.shape)
        x = self.pool1(F.relu(self.con1(x)))
        #print(x.shape)
        x = self.pool1(F.relu(self.con2(x)))
        #print(x.shape)
        x = self.pool1(F.relu(self.con3(x)))
        #print(x.shape)
        x = self.pool1(F.relu(self.con4(x)))
        #print(x.shape)
        x = self.pool1(F.relu(self.con5(x)))
        #print(x.shape)
        x = self.pool1(F.relu(self.con6(x)))
        x = x.view(x.shape[0], -1)
        #print(x.shape)
        x = self.dropout(F.relu(self.fc1(x)))
        #print(x.shape)
        x = self.dropout(F.relu(self.fc2(x)))
        #print(x.shape)
        x = self.dropout(F.relu(self.fc3(x)))
        #print(x.shape)
        x = F.log_softmax(self.fc4(x), 1)
        #print(x.shape)
        return x
    
model = Network()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr = 0.0008)
criterion = nn.NLLLoss()
epochs = 40

        
        
        
        
        
save = 99999999999999
trainL= []
testL = []
for e in range(epochs):
    
    train_loss = 0
    model.train()
    
    trainloaderSize = 0
    testloaderSize = 0
    for images, labels in trainloader:
        
        trainloaderSize += 1
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * images.shape[0]
    
    
    with torch.no_grad():
        
        model.eval()
        test_loss = 0
        accuracy = 0
        
        for images, labels in testloader:

            
            testloaderSize += 1
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            test_loss += loss.item() * images.shape[0]
            
            prob = torch.exp(output)
            class_val, class_num = prob.topk(1, dim = 1)
            equal = labels == class_num.view(labels.shape)
            
            accuracy += torch.mean(equal.type(torch.FloatTensor)).item()
        
        accuracy = accuracy / testloaderSize
        test_loss = test_loss/ (testloaderSize * 64)
        train_loss = train_loss/ (trainloaderSize * 64)
        
        print("Epoch-",e ,"    Test Loss-",test_loss, 
              "    Train Loss-", train_loss,"   Accuracy-",accuracy, "\n" )
        trainL.append(train_loss)
        testL.append(test_loss)
        if test_loss < save:
            
            print("Model saving")
            print("Test Loss -", test_loss, " is less than Previous Loss -", save,"\n" )
            torch.save(model.state_dict(), 'model.pth')
            save = test_loss
        
            
        
        
        
        
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
plt.plot(trainL, label='Training loss')
plt.plot(testL, label='Testing loss')
plt.legend(frameon=False)












