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
import numpy as np

train_images = np.load('../input/kmnist-train-imgs.npz')['arr_0']
test_images = np.load('../input/kmnist-test-imgs.npz')['arr_0']


train_labels = np.load('../input/kmnist-train-labels.npz')['arr_0']
test_labels = np.load('../input/kmnist-test-labels.npz')['arr_0']
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import numpy as np

class MyDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        #print(self.data.shape)
        self.data = self.data.unsqueeze(1)
        #print("Now",self.data.shape)
        self.target = torch.from_numpy(target).long()
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.data)
#import MyDataset
from torchvision import transforms
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.Normalize( (0.5,0.5,0.5), (0.5,0.5,0.5) )])

trainDataset = MyDataset(train_images, train_labels, transform)
trainloader = torch.utils.data.DataLoader(trainDataset, batch_size=64, shuffle=True)

testDataset = MyDataset(test_images, test_labels, transform)
testLoader = torch.utils.data.DataLoader(testDataset, batch_size = 64, shuffle = True)


images, labels = next(iter(testLoader))
print(labels)

import matplotlib.pyplot as plt 

images = images * 0.5 + 0.5
images = images.numpy()

print(images[0][0].shape)
imgplot = plt.imshow(images[0][0])

print(len(testLoader))
print(len(testLoader.dataset))

from torch import nn, optim
import torch.nn.functional as F

class Network(nn.Module):
    
    def __init__(self):
        super().__init__()
        #28
        self.conv1 = nn.Conv2d(1, 16, kernel_size = 3, stride = 1, padding = 1)
        #28
        #14
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding = 1)
        #14
        #7
        self.conv3 = nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 0)
        #5
        #3 pool2
        self.conv4 = nn.Conv2d(64, 96, kernel_size = 3, stride = 1, padding = 1)
        #3
        self.conv5 = nn.Conv2d(96, 96, kernel_size = 2, stride = 1, padding = 0)
        #2
        
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
        self.pool2 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        self.fc1 = nn.Linear(384, 270)
        self.fc2 = nn.Linear(270, 200)
        self.fc3 = nn.Linear(200, 100)
        self.fc4 = nn.Linear(100, 49)
        self.dropout = nn.Dropout(p = 0.2)
        
    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(x.shape[0], -1)
        #x = self.dropout(x)
        x = self.dropout( F.relu(self.fc1(x)) )
        x = self.dropout( F.relu(self.fc2(x)) )
        x = self.dropout( F.relu(self.fc3(x)) )
        x = F.log_softmax(self.fc4(x), dim = 1)
        
        return x

model = Network()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr = 0.001)
criterion = nn.NLLLoss()

epochs = 3
trainL= []
testL = []

msave = 10000000
for e in range(epochs):
    
    train_loss = 0
    
    model.train()
    
    for images, labels in trainloader:
        
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
        
        for images, labels in testLoader:
            
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss = criterion(output,labels)
            test_loss += loss.item() * images.shape[0] 
            
            prob = torch.exp(output)
            class_val, class_num = prob.topk(1, dim = 1)
            equal = labels == class_num.view(labels.shape[0])
            
            
            #print("Class Number", class_num)
            #print("Class Numver new View", class_num.view(labels.shape[0]))
            #print("Actual Class Number", labels)
            #print("Equal", equal)
            #print("Mean Correct", torch.mean(equal.type(torch.FloatTensor)).item())
    
            accuracy += torch.mean(equal.type(torch.FloatTensor)).item()
            
        tstloss = test_loss/ len(testLoader.dataset)
        acc = accuracy/ len(testLoader)
        trnloss = train_loss/len(trainloader.dataset)
        trainL.append(trnloss)
        testL.append(tstloss)
        
        print("Train Loss-",trnloss, "   Test Loss-",tstloss, "   Accuracy-", acc)
        if msave>tstloss:
            
            torch.save(model.state_dict(), 'model.pth')
            print("Model Saving   ","Test Loss-", tstloss, "  is less than-",msave)
            msave = tstloss
        print("\n")

