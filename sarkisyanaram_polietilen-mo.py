import os

import numpy as np

import pandas as pd

import cv2

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.model_selection import train_test_split



import torch 

import torch.nn as nn

import torch.nn.functional as F

import torchvision

import torchvision.transforms as transforms

from torch.utils.data import TensorDataset, DataLoader, Dataset
## Parameters for model



# Hyper parameters

num_epochs = 8

num_classes = 2

batch_size = 128

learning_rate = 0.002



# Device configuration

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # gpu or cpu
labels = pd.read_csv('../input/histopathologic-cancer-detection/train_labels.csv')

sub = pd.read_csv('../input/histopathologic-cancer-detection/sample_submission.csv')

train_path = '../input/histopathologic-cancer-detection/train/'

test_path = '../input/histopathologic-cancer-detection/test/'
labels.sample(5)
#Splitting data into train and val

train, val = train_test_split(labels, stratify=labels.label, test_size=0.1) 

# 10% of train data are used for validation

len(train), len(val)
class MyDataset(Dataset):

    def __init__(self, df_data, data_dir = './', transform=None):

        super().__init__()

        self.df = df_data.values

        self.data_dir = data_dir

        self.transform = transform



    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, index):

        img_name,label = self.df[index]

        img_path = os.path.join(self.data_dir, img_name+'.tif')

        image = cv2.imread(img_path)

        if self.transform is not None:

            image = self.transform(image)

        return image, label
# torchvision.transforms provides common image transformations

# transforms.Compose - composes several transforms together

trans_train = transforms.Compose([transforms.ToPILImage(), # Convert a tensor to PIL Image (Python Imaging Library)

                                  transforms.Pad(64, padding_mode='reflect'),

                                  transforms.RandomHorizontalFlip(), #randomly flips

                                  transforms.RandomVerticalFlip(), #randomly flips

                                  transforms.RandomRotation(30), #rotates the image by angle in [1,20]

                                  transforms.ToTensor(),

                                  transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])



trans_valid = transforms.Compose([transforms.ToPILImage(),

                                  transforms.Pad(64, padding_mode='reflect'),

                                  transforms.ToTensor(),

                                  transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])



dataset_train = MyDataset(df_data=train, data_dir=train_path, transform=trans_train)

dataset_valid = MyDataset(df_data=val, data_dir=train_path, transform=trans_valid)





# DataLoader - combines a dataset and a sampler, and provides an iterable over the given dataset

# there are 128 samples per batch to load

# DataLoader shuffles train data every epoch and doesnt shuffle valid data

loader_train = DataLoader(dataset = dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)

loader_valid = DataLoader(dataset = dataset_valid, batch_size=batch_size//2, shuffle=False, num_workers=0)
class SimpleCNN(nn.Module):  # Base class for my neural network

    def __init__(self):

        # ancestor constructor call

        super(SimpleCNN, self).__init__() 

        # nn.Conv2d - applies a 2D convolution

        # out_channels of previous layer should be equal to in_channels of current layer

        # 

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=2)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=2)

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=2)

        # BatchNorm2d normalizes outputs of each layer

        self.bn1 = nn.BatchNorm2d(32)

        self.bn2 = nn.BatchNorm2d(64)

        self.bn3 = nn.BatchNorm2d(128)

        self.bn4 = nn.BatchNorm2d(256)

        self.bn5 = nn.BatchNorm2d(512)

        # pooling function is 2D max

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        

        self.avg = nn.AvgPool2d(8) # 2D average pooling

        self.fc = nn.Linear(512 * 1 * 1, 2) #linear transformation to the incoming data



    def forward(self, x):

        # first convolutional layer then batchnorm, then activation then pooling layer

        # LeakyReLU as activation function

        x = self.pool(F.celu(self.bn1(self.conv1(x)))) 

        x = self.pool(F.celu(self.bn2(self.conv2(x))))

        x = self.pool(F.celu(self.bn3(self.conv3(x))))

        x = self.pool(F.celu(self.bn4(self.conv4(x))))

        x = self.pool(F.celu(self.bn5(self.conv5(x))))

        x = self.avg(x)

        x = x.view(-1, 512 * 1 * 1)

        x = self.fc(x)

        return x
model = SimpleCNN().to(device)
# Cross-entropy as loss function

criterion = nn.CrossEntropyLoss()

# AdaMax is used as gradient descent optimization algorithm

optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)
total_step = len(loader_train)

for epoch in range(num_epochs):

    for i, (images, labels) in enumerate(loader_train):

        images = images.to(device)

        labels = labels.to(device)

        

        # Forward pass

        outputs = model(images)

        loss = criterion(outputs, labels)

        #backpropagation method for grad calculating

        # Backward and optimize

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        

        if (i+1) % 100 == 0:

            # value of Loss function after each epochs

            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 

                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
# Test the model

model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)

with torch.no_grad():

    correct = 0

    total = 0

    for images, labels in loader_valid:

        images = images.to(device)

        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

          

    print('Test Accuracy of the model on the 22003 test images: {} %'.format(100 * correct / total))



# Save the model checkpoint

torch.save(model.state_dict(), 'model.ckpt')
dataset_valid = MyDataset(df_data=sub, data_dir=test_path, transform=trans_valid)

loader_test = DataLoader(dataset = dataset_valid, batch_size=32, shuffle=False, num_workers=0)
model.eval()



preds = []

for batch_i, (data, target) in enumerate(loader_test):

    data, target = data.cuda(), target.cuda()

    output = model(data)



    pr = output[:,1].detach().cpu().numpy()

    for i in pr:

        preds.append(i)

sub.shape, len(preds)

sub['label'] = preds

sub.to_csv('s.csv', index=False)