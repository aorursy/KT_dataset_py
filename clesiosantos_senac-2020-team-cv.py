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
import zipfile



with zipfile.ZipFile("../input/aerial-cactus-identification/train.zip","r") as z:

    z.extractall("/kaggle/working/train")



with zipfile.ZipFile("../input/aerial-cactus-identification/test.zip","r") as z:

    z.extractall("/kaggle/working/test")
import numpy as np

import pandas as pd

import os



import cv2

import matplotlib.pyplot as plt

%matplotlib inline
# Data path

labels = pd.read_csv('../input/aerial-cactus-identification/train.csv')

sub = pd.read_csv('../input/aerial-cactus-identification/sample_submission.csv')

train_path = '/kaggle/working/train/train'

test_path = '/kaggle/working/test/test'



print('Número de imagens de treinamento:{0}'.format(len(os.listdir(train_path))))

print('Número de imagens de teste:{0}'.format(len(os.listdir(test_path))))



labels.head()
labels['has_cactus'].value_counts()
lab = 'Has cactus','Hasn\'t cactus'

colors=['green','brown']



plt.figure(figsize=(7,7))

plt.pie(labels.groupby('has_cactus').size(), labels=lab,

        labeldistance=1.1, autopct='%1.1f%%',

        colors=colors,shadow=True, startangle=140)

plt.show()
fig,ax = plt.subplots(1,5,figsize=(15,3))



for i, idx in enumerate(labels[labels['has_cactus']==1]['id'][-5:]):

  path = os.path.join(train_path,idx)

  ax[i].imshow(cv2.imread(path)) # [...,[2,1,0]]
fig,ax = plt.subplots(1,5,figsize=(15,3))



for i, idx in enumerate(labels[labels['has_cactus']==0]['id'][-5:]):

  path = os.path.join(train_path,idx)

  ax[i].imshow(cv2.imread(path)) # [...,[2,1,0]]



# Libreries



import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader, Dataset

import torchvision

import torchvision.transforms as transforms



from sklearn.model_selection import train_test_split
## Parameters for model



# Hyper parameters

num_epochs = 25

num_classes = 2

batch_size = 128

learning_rate = 0.002



# Device configuration

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# data splitting

train, val = train_test_split(labels, stratify=labels.has_cactus, test_size=0.1)

train.shape, val.shape
train['has_cactus'].value_counts()
val['has_cactus'].value_counts()
# NOTE: class is inherited from Dataset

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

        img_path = os.path.join(self.data_dir, img_name)

        image = cv2.imread(img_path)

        if self.transform is not None:

            image = self.transform(image)

        return image, label

# Image preprocessing

# Ajustando 

trans_train = transforms.Compose([transforms.ToPILImage(),

                                  transforms.Pad(32, padding_mode='reflect'),

                                  transforms.ToTensor(),

                                  transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])



trans_valid = transforms.Compose([transforms.ToPILImage(),

                                  transforms.Pad(32, padding_mode='reflect'),

                                  transforms.ToTensor(),

                                  transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])



# Data generators

dataset_train = MyDataset(df_data=train, data_dir=train_path, transform=trans_train)

dataset_valid = MyDataset(df_data=val, data_dir=train_path, transform=trans_valid)



loader_train = DataLoader(dataset = dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)

loader_valid = DataLoader(dataset = dataset_valid, batch_size=batch_size//2, shuffle=False, num_workers=0)
# 5 camadas de convolução

# 

class SimpleCNN(nn.Module):

    def __init__(self):

        # ancestor constructor call

        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=2)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=2)

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=2)

        self.bn1 = nn.BatchNorm2d(32)

        self.bn2 = nn.BatchNorm2d(64)

        self.bn3 = nn.BatchNorm2d(128)

        self.bn4 = nn.BatchNorm2d(256)

        self.bn5 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.avg = nn.AvgPool2d(4)

        self.fc = nn.Linear(512 * 1 * 1, 2) # !!!

   

    def forward(self, x):

        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x)))) # first convolutional layer then batchnorm, then activation then pooling layer.

        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x))))

        x = self.pool(F.leaky_relu(self.bn3(self.conv3(x))))

        x = self.pool(F.leaky_relu(self.bn4(self.conv4(x))))

        x = self.pool(F.leaky_relu(self.bn5(self.conv5(x))))

        x = self.avg(x)

        #print(x.shape) # lifehack to find out the correct dimension for the Linear Layer

        x = x.view(-1, 512 * 1 * 1) # !!!

        x = self.fc(x)

        return x
model = SimpleCNN().to(device)
# Loss and optimizer

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)
# Train the model

total_step = len(loader_train)

for epoch in range(num_epochs):

    for i, (images, labels) in enumerate(loader_train):

        images = images.to(device)

        labels = labels.to(device)

        

        # Forward pass

        outputs = model(images)

        loss = criterion(outputs, labels)

        

        # Backward and optimize

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        

        if (i+1) % 100 == 0:

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

          

    print('Test Accuracy of the model on the 1750 validation images: {} %'.format(100 * correct / total))



# Save the model checkpoint

torch.save(model.state_dict(), 'model.ckpt')
# generator for test data 

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



sub['has_cactus'] = preds

sub.to_csv('sub.csv', index=False)


