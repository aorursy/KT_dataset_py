import numpy as np # Matrix Operations (Matlab of Python)
import pandas as pd # Work with Datasources
import matplotlib.pyplot as plt # Drawing Library

from PIL import Image

import torch # Like a numpy but we could work with GPU by pytorch library
import torch.nn as nn # Nural Network Implimented with pytorch
import torchvision # A library for work with pretrained model and datasets

from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

import glob
import os

%matplotlib inline

image_size = (100, 100)
image_row_size = image_size[0] * image_size[1]
class CatDogDataset(Dataset):
    def __init__(self, path, transform=None):
        self.classes   = os.listdir(path)
        self.path      = [f"{path}/{className}" for className in self.classes]
        self.file_list = [glob.glob(f"{x}/*") for x in self.path]
        self.transform = transform
        
        files = []
        for i, className in enumerate(self.classes):
            for fileName in self.file_list[i]:
                files.append([i, className, fileName])
        self.file_list = files
        files = None
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fileName = self.file_list[idx][2]
        classCategory = self.file_list[idx][0]
        im = Image.open(fileName)
        if self.transform:
            im = self.transform(im)
        return im.view(-1), classCategory
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
transform = transforms.Compose([
                                transforms.Resize(image_size), 
                                transforms.Grayscale(),
                                transforms.ToTensor(), 
                                transforms.Normalize(mean, std)])
path    = '../input/training_set/training_set'
dataset = CatDogDataset(path, transform=transform)
def imshow(source):
    plt.figure(figsize=(10,10))
    imt = (source.view(-1, image_size[0], image_size[0]))
    imt = imt.numpy().transpose([1,2,0])
    imt = (std * imt + mean).clip(0,1)
    plt.subplot(1,2,2)
    plt.imshow(imt)
imshow(dataset[0][0])
imshow(dataset[2][0])
imshow(dataset[6000][0])
shuffle     = True
batch_size  = 64
num_workers = 0
dataloader  = DataLoader(dataset=dataset, 
                         shuffle=shuffle, 
                         batch_size=batch_size, 
                         num_workers=num_workers)
class MyModel(torch.nn.Module):
    def __init__(self, in_feature):
        super(MyModel, self).__init__()
        self.fc1     = torch.nn.Linear(in_features=in_feature, out_features=500)
        self.fc2     = torch.nn.Linear(in_features=500, out_features=100)
        self.fc3     = torch.nn.Linear(in_features=100, out_features=1)

    def forward(self, x):
        x = F.relu( self.fc1(x) )
        x = F.relu( self.fc2(x) )
        x = F.softmax( self.fc3(x), dim=1)
        return x
model = MyModel(image_row_size)
print(model)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.95)

epochs   = 10
for epoch in range(epochs):
    for i, (X,Y) in enumerate(dataloader):
#         x, y = dataset[i]
        yhat = model(X)
        loss = criterion(yhat.view(-1), Y)
        break

yhat.view(-1).size()
# loss = criterion(yhat, y)


device = torch.device('cpu')
# if torch.cuda.is_available():
#     device = torch.device('cuda')

D_in, H, D_out = image_row_size, 100, 2
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, 500),
    torch.nn.ReLU(),
    torch.nn.Linear(500, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 2),
    torch.nn.Sigmoid()
).to(device)

# layers = []
# layers.append(nn.Linear(D_in, 500))
# layers.append(nn.ReLU())
# layers.append(nn.Linear(500, 100))
# layers.append(nn.ReLU())
# layers.append(nn.Linear(100, 2))
# layers.append(nn.Sigmoid())

# model = nn.Sequential(*layers)


print(model)
model(x)
class CatDogDataset(Dataset):
    def __init__(self, path, transform=None):
        self.classes   = os.listdir(path)
        self.path      = [f"{path}/{className}" for className in self.classes]
        self.file_list = [glob.glob(f"{x}/*") for x in self.path]
        self.transform = transform
        
        files = []
        for i, className in enumerate(self.classes):
            for fileName in self.file_list[i]:
                files.append([i, className, fileName])
        self.file_list = files
        files = None
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fileName = self.file_list[idx][2]
        classCategory = self.file_list[idx][0]
        im = Image.open(fileName)
        if self.transform:
            im = self.transform(im)
        return im, classCategory
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
transform = transforms.Compose([
                                transforms.Resize(image_size), 
                                transforms.Grayscale(),
                                transforms.ToTensor(), 
                                transforms.Normalize(mean, std)])



path    = '../input/training_set/training_set'
dataset = CatDogDataset(path, transform=transform)

shuffle     = True
batch_size  = 64
num_workers = 0
dataloader  = DataLoader(dataset=dataset, 
                         shuffle=shuffle, 
                         batch_size=batch_size, 
                         num_workers=num_workers)
class MyCNNModel(torch.nn.Module):
    def __init__(self):
        super(MyCNNModel, self).__init__()
        self.relu    = torch.nn.ReLU()
        self.fc1     = torch.nn.Linear(in_features=1, out_features=500)
        self.fc2     = torch.nn.Linear(in_features=500, out_features=100)
        self.fc3     = torch.nn.Linear(in_features=100, out_features=2)
    def forward(self, x):
#         torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3)),
#         torch.nn.ReLU(),
#         torch.nn.MaxPool2d((2,2)),
#         torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3)),
#         torch.nn.ReLU(),
#         torch.nn.MaxPool2d((2,2)),
#         torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3)),
#         torch.nn.ReLU(),
#         torch.nn.MaxPool2d((2,2)),
#         torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3)),
#         torch.nn.ReLU(),
#         torch.nn.MaxPool2d((2,2)),
#         Flatten(),
#         torch.nn.Linear(128*3*3, 512),
#         torch.nn.ReLU(),
#         torch.nn.Linear(512, 1),
#         torch.nn.Sigmoid()
        return x

class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2,2)),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2,2)),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2,2)),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2,2)),
            Flatten(),
            torch.nn.Linear(1152, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1),
            torch.nn.Sigmoid()
)
model
x, y = dataset[0]
xx = x.unsqueeze(0)
model(xx)
num_classes = 2
class UnaryNet(nn.Module):
    def __init__(self):
        super(UnaryNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)

        self.fc1_mean = nn.Linear(9680 , 140)
        self.fc2_mean = nn.Linear(140, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 9680 )
        
        mean = F.relu(self.fc1_mean(x))
        mean = self.fc2_mean(mean)

        return mean

x, y = dataset[0]

net = UnaryNet()

xx = x.unsqueeze(0)
net(xx)
