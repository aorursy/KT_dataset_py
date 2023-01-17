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
import matplotlib.pyplot as plt

import matplotlib.image as Image

from sklearn.model_selection import train_test_split

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms

import torchvision.models as models

import PIL

data_dir = "../input"

train_dir = data_dir + "/train/train"

test_dir = data_dir + "/test/test"



labels = pd.read_csv(data_dir + "/train.csv")

labels.head()
balance = labels['has_cactus'].value_counts()

balance
train, valid = train_test_split(labels, stratify=labels.has_cactus, test_size=0.2)
num_epochs = 30

num_classes = 2

batch_size = 128

learning_rate = 0.002

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

device
class cactData(Dataset):

    def __init__(self, split_data, data_root = './', transform=None):

        super().__init__()

        self.df = split_data.values

        self.data_root = data_root

        self.transform = transform



    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, index):

        img_name,label = self.df[index]

        img_path = os.path.join(self.data_root, img_name)

        image = Image.imread(img_path)

        if self.transform is not None:

            image = self.transform(image)

        return image, label
mean = [0.5, 0.5, 0.5]

std = [0.5, 0.5, 0.5]

train_transf = transforms.Compose([transforms.ToPILImage(),

#                                   transforms.Normalize(mean, std),

#                                   transforms.RandomCrop(20),

                                    transforms.ColorJitter(hue=.05, saturation=.05),

                                    transforms.RandomHorizontalFlip(),

                                    transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),

#                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

                                    transforms.ToTensor()

                                  ])



valid_transf = transforms.Compose([transforms.ToPILImage(),

                                  transforms.ToTensor()])

# train_transf = transforms.Compose([transforms.RandomRotation(30),

#                                        transforms.RandomResizedCrop(224),

#                                        transforms.RandomHorizontalFlip(),

#                                        transforms.ToTensor(),

#                                        transforms.Normalize([0.485, 0.456, 0.406],

#                                                             [0.229, 0.224, 0.225])])



# valid_transf = transforms.Compose([transforms.Resize(255),

#                                       transforms.CenterCrop(224),

#                                       transforms.ToTensor(),

#                                       transforms.Normalize([0.485, 0.456, 0.406],

#                                                            [0.229, 0.224, 0.225])])
train_data = cactData(train, train_dir, train_transf)

valid_data = cactData(valid, train_dir, valid_transf)



train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0)



valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size//2, shuffle=False, num_workers=0)
class CactCNN(nn.Module):

    def __init__(self):

        super(CactCNN, self).__init__()

        self.conv1 = nn.Sequential(

            nn.Conv2d(3, 32, 4, 2, 0),

            nn.BatchNorm2d(32),

            nn.ReLU()

        )



        self.conv2 = nn.Sequential(

            nn.Conv2d(32, 64, 3, 2, 0),

            nn.BatchNorm2d(64),

            nn.ReLU()

        )



        self.conv3 = nn.Sequential(

            nn.Conv2d(64, 128, 3, 2, 0),

            nn.BatchNorm2d(128),

            nn.ReLU()

        )

        

        self.conv4 = nn.Sequential(

            nn.Conv2d(128, 256, 3, 2, 0),

            nn.BatchNorm2d(256),

            nn.ReLU()

        )

        

        self.fc = nn.Sequential(

            nn.Linear(256*1*1, 1024),

            nn.ReLU(),

            nn.Dropout(p=0.2),

            nn.Linear(1024,2)

        )

    def forward(self, x):

        x = self.conv1(x)

        x = self.conv2(x)

        x = self.conv3(x)

        x = self.conv4(x)

        x = x.view(x.shape[0],-1)

        x = self.fc(x)

        return x
# model = models.resnet18(pretrained=True)

# for param in model.parameters():

#     param.requires_grad = False

    

# from collections import OrderedDict

# classifier = nn.Sequential(nn.Linear(512, 256),

#                                  nn.ReLU(),

# #                                  nn.Dropout(0.2),

#                                  nn.Linear(256, 64),

#                                  nn.Linear(64, 2),

#                                  nn.LogSoftmax(dim=1))

# model.fc = classifier

# model
model = CactCNN().to(device)

criterion = nn.CrossEntropyLoss()

# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

optimizer = torch.optim.Adam(model.fc.parameters(), lr=learning_rate)

print(device)

model
for epoch in range(num_epochs):

    for i, (images, labels) in enumerate(train_loader):

        images = images.to(device)

        labels = labels.to(device)

#         print(images[0].shape)

        

        out = model(images)

        loss = criterion(out, labels)

        

        #optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    print('Epoch: {}/{}, Loss: {}'.format(epoch+1, num_epochs, loss.item()))
model.eval()

with torch.no_grad():

    correct = 0

    total = 0

    for images, labels in valid_loader:

        images = images.to(device)

        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    print('Test Accuracy: {} %'.format(100 * correct / total))
submit = pd.read_csv(data_dir + '/sample_submission.csv')

test_data = cactData(split_data = submit, data_root = test_dir, transform = valid_transf)

test_loader = DataLoader(dataset = test_data, batch_size=32, shuffle=False, num_workers=0)
model.eval()

predict = []

for batch_i, (data, target) in enumerate(test_loader):

    data, target = data.to(device), target.to(device)

    output = model(data)



    pr = output[:,1].detach().cpu().numpy()

    for i in pr:

        predict.append(i)



submit['has_cactus'] = predict

submit.to_csv('submission.csv', index=False)