# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch
from torch import nn, optim

import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset

import torchvision
from torchvision import transforms

if torch.cuda.is_available():
    cuda_is_available = True
else:
    cuda_is_available = False
class DatasetMNIST(Dataset):
    
    def __init__(self, file_path, transform=None):
        self.data = pd.read_csv(file_path)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image = self.data.iloc[index, 1:].values.astype(np.uint8).reshape((28, 28, 1))
        label = self.data.iloc[index, 0]
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label
transform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                            torchvision.transforms.RandomHorizontalFlip(),
                                            torchvision.transforms.RandomRotation((-45,45)),
                                            torchvision.transforms.ToTensor()])
train_dataset = DatasetMNIST('../input/train.csv', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=40, shuffle=True)
class CNNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 5, 3, padding=1)
        self.conv2 = nn.Conv2d(5, 10, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(10, 20, 3, padding=1)
        self.conv4 = nn.Conv2d(20, 30, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(30*7*7, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = x.view(x.shape[0], -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.relu(self.fc3(x))

        return x
        
model = CNNetwork()         
if cuda_is_available:
    model = model.cuda()

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.01)
epochs = 100
for e in range(epochs):
    running_loss = 0
    running_accuracy = []
    for images, labels in train_loader:
        if cuda_is_available:
            images, labels = images.cuda(), labels.cuda()

        optimizer.zero_grad()
        
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_accuracy.append(torch.sum((torch.max(output,1)[1] == labels)).double().item()/labels.shape[0])
    else:
        print(f"Epochs: {e+1}/{epochs}; Training loss: {running_loss/len(train_loader)}; Training accuracy: {sum(running_accuracy)/len(running_accuracy)}")
class TestDatasetMNIST(Dataset):
    
    def __init__(self, file_path, transform=None):
        self.data = pd.read_csv(file_path)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image = self.data.iloc[index, :].values.astype(np.uint8).reshape((28, 28, 1))
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image
test_dataset = TestDatasetMNIST('../input/test.csv', transform=torchvision.transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=40, shuffle=False)
predictions = []

model.eval()
for images in test_loader:
    if cuda_is_available:
        images = images.cuda()
            
    with torch.no_grad():
        output = model(images)
        _, prediction = torch.topk(output, 1)
        
    predictions += [prediction[ii].item() for ii in range(len(prediction))]
    
model.train()

dt_dict = {'ImageId':[ii for ii in range(1, len(predictions)+1)], 'Label': predictions}
dt = pd.DataFrame(data=dt_dict)
dt.to_csv('my_submission.csv', index=False)
import matplotlib.pyplot as plt
def göster(x):
    print(prediction[x].item())
    plt.imshow(images[x].view(28,28))