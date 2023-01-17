# Code extensively uses the kernel https://www.kaggle.com/bonhart/pytorch-cnn-from-scratch!



# Thank you https://www.kaggle.com/bonhart
# Libraries

import os

import numpy as np

import pandas as pd

import cv2

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



import torch 

import torch.nn as nn

import torch.nn.functional as F

import torchvision

import torchvision.transforms as transforms

from torch.utils.data import TensorDataset, DataLoader, Dataset
from glob import glob

imagePatches = glob('../input/IDC_regular_ps50_idx5/**/*.png', recursive=True)
len(imagePatches)
imagePatches[0]
import fnmatch

patternZero = '*class0.png'

patternOne = '*class1.png'

classZero = fnmatch.filter(imagePatches, patternZero)

classOne = fnmatch.filter(imagePatches, patternOne)
from tqdm import tqdm

y = []

for img in tqdm(imagePatches):

    if img in classZero:

        y.append(0)

    elif img in classOne:

        y.append(1)
images_df = pd.DataFrame()
images_df["images"] = imagePatches

images_df["labels"] = y
images_df.head()
images_df.to_csv("my_frame.csv",index=False)
images_df.groupby("labels")["labels"].count()
#Splitting data into train and val

train, val = train_test_split(images_df, stratify=images_df.labels, test_size=0.2)

len(train), len(val)
class MyDataset(Dataset):

    def __init__(self, df_data,transform=None):

        super().__init__()

        self.df = df_data.values

        

        self.transform = transform



    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, index):

        img_path,label = self.df[index]

        

        image = cv2.imread(img_path)

        image = cv2.resize(image, (50,50))

        if self.transform is not None:

            image = self.transform(image)

        return image, label
## Parameters for model



# Hyper parameters

num_epochs = 10

num_classes = 2

batch_size = 128

learning_rate = 0.002



# Device configuration

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
trans_train = transforms.Compose([transforms.ToPILImage(),

                                  transforms.Pad(64, padding_mode='reflect'),

                                  transforms.RandomHorizontalFlip(), 

                                  transforms.RandomVerticalFlip(),

                                  transforms.RandomRotation(20), 

                                  transforms.ToTensor(),

                                  transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])



trans_valid = transforms.Compose([transforms.ToPILImage(),

                                  transforms.Pad(64, padding_mode='reflect'),

                                  transforms.ToTensor(),

                                  transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])



dataset_train = MyDataset(df_data=train, transform=trans_train)

dataset_valid = MyDataset(df_data=val,transform=trans_valid)



loader_train = DataLoader(dataset = dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)

loader_valid = DataLoader(dataset = dataset_valid, batch_size=batch_size//2, shuffle=False, num_workers=0)
import torch

import torch.nn as nn

import torch.nn.functional as F





class double_conv(nn.Module):

    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):

        super(double_conv, self).__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(in_ch, out_ch, 3, padding=1),

            nn.BatchNorm2d(out_ch),

            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1),

            nn.BatchNorm2d(out_ch),

            nn.ReLU(inplace=True)

        )



    def forward(self, x):

        x = self.conv(x)

        return x





class inconv(nn.Module):

    def __init__(self, in_ch, out_ch):

        super(inconv, self).__init__()

        self.conv = double_conv(in_ch, out_ch)



    def forward(self, x):

        x = self.conv(x)

        return x





class down(nn.Module):

    def __init__(self, in_ch, out_ch):

        super(down, self).__init__()

        self.mpconv = nn.Sequential(

            nn.MaxPool2d(2),

            double_conv(in_ch, out_ch)

        )



    def forward(self, x):

        x = self.mpconv(x)

        return x





class up(nn.Module):

    def __init__(self, in_ch, out_ch, bilinear=True):

        super(up, self).__init__()



        #  would be a nice idea if the upsampling could be learned too,

        #  but my machine do not have enough memory to handle all those weights

        if bilinear:

            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        else:

            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)



        self.conv = double_conv(in_ch, out_ch)



    def forward(self, x1, x2):

        x1 = self.up(x1)

        

        # input is CHW

        diffY = x2.size()[2] - x1.size()[2]

        diffX = x2.size()[3] - x1.size()[3]



        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,

                        diffY // 2, diffY - diffY//2))

        

        # for padding issues, see 

        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a

        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd



        x = torch.cat([x2, x1], dim=1)

        x = self.conv(x)

        return x





class outconv(nn.Module):

    def __init__(self, in_ch, out_ch):

        super(outconv, self).__init__()

        self.conv = nn.Conv2d(in_ch, out_ch, 1)



    def forward(self, x):

        x = self.conv(x)

        return x
class UNet(nn.Module):

    def __init__(self, n_channels, n_classes):

        super(UNet, self).__init__()

        self.inc = inconv(n_channels, 64)

        self.down1 = down(64, 128)

        self.down2 = down(128, 256)

        self.down3 = down(256, 512)

        self.down4 = down(512, 512)

        self.up1 = up(1024, 256)

        self.up2 = up(512, 128)

        self.up3 = up(256, 64)

        self.up4 = up(128, 64)

        self.outc = outconv(64, 2)

        self.fc1 = nn.Linear(2*388*388,200)

        self.fc2 = nn.Linear(200,50)

        self.fc3 = nn.Linear(50,2)



    def forward(self, x):

        x1 = self.inc(x)

        x2 = self.down1(x1)

        x3 = self.down2(x2)

        x4 = self.down3(x3)

        x5 = self.down4(x4)

        x = self.up1(x5, x4)

        x = self.up2(x, x3)

        x = self.up3(x, x2)

        x = self.up4(x, x1)

        x = self.outc(x)

        x = x.view(-1)

        x = F.leaky_relu(self.fc1(x))

        x = F.leaky_relu(self.fc2(x))

        x = F.leaky_relu(self.fc3(x))

        

        return F.sigmoid(x)
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

        self.avg = nn.AvgPool2d(2)

        self.fc = nn.Linear(512 * 1 * 1,2) # !!!

        

    def forward(self, x):

        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x)))) # first convolutional layer then batchnorm, then activation then pooling layer.

        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x))))

        x = self.pool(F.leaky_relu(self.bn3(self.conv3(x))))

        x = self.pool(F.leaky_relu(self.bn4(self.conv4(x))))

        x = self.pool(F.leaky_relu(self.bn5(self.conv5(x))))

        print(x.shape) # lifehack to find out the correct dimension for the Linear Layer

        x = x.view(-1, 512 * 1 * 1) # !!!

        x = self.fc(x)

        return x
model = UNet.to(device)
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

confusion_matrix = torch.zeros(2, 2)

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

        for t, p in zip(labels.view(-1), predicted.view(-1)):

                confusion_matrix[t.long(), p.long()] += 1

                 

    print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))



# Save the model checkpoint

torch.save(model.state_dict(), 'model.ckpt')
print(confusion_matrix)