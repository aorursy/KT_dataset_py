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
!pip install torchsummary
# Importing Libraries

import torch

import torchvision

import torchvision.transforms as transforms

from torchsummary import summary

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau



import matplotlib.pyplot as plt

%matplotlib inline



device = "cuda" if torch.cuda.is_available else "cpu"

print(device)
from torchvision import transforms

import numpy as np

import torch



# Returns a list of transformations when called



class GetTransforms():

    '''Returns a list of transformations when type as requested amongst train/test

       Transforms('train') = list of transforms to apply on training data

       Transforms('test') = list of transforms to apply on testing data'''



    def __init__(self):

        pass



    def trainparams(self):

        train_transformations = [ #resises the image so it can be perfect for our model.

            transforms.RandomHorizontalFlip(), # FLips the image w.r.t horizontal axis

            transforms.RandomRotation((-7,7)),     #Rotates the image to a specified angel

            transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)), #Performs actions like zooms, change shear angles.

            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Set the color params

            transforms.ToTensor(), # comvert the image to tensor so that it can work with torch

            transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261)) #Normalize all the images

            ]



        return train_transformations



    def testparams(self):

        test_transforms = [

            transforms.ToTensor(),

            transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261))

        ]

        return test_transforms
from torchvision import datasets

from torchvision import transforms





transformations = GetTransforms()

train_transforms = transforms.Compose(transformations.trainparams())

test_transforms = transforms.Compose(transformations.testparams())





class GetCIFAR10_TrainData():

    def __init__(self, dir_name:str):

        self.dirname = dir_name



    def download_train_data(self):

        return datasets.CIFAR10('./data', train=True, download=True, transform=train_transforms)



    def download_test_data(self):

        return datasets.CIFAR10('./data', train=False, download=True, transform=test_transforms)

data = GetCIFAR10_TrainData(os.chdir(".."))

trainset = data.download_train_data()

testset = data.download_test_data()

trainloader = torch.utils.data.DataLoader(trainset, batch_size=512,

                                          shuffle=True, num_workers=4)

testloader = torch.utils.data.DataLoader(testset, batch_size=512,

                                         shuffle=False, num_workers=4)
class BasicBlock(nn.Module):

    expansion = 1

    



    def __init__(self, in_planes, planes, stride=1):

        super(BasicBlock, self).__init__()



        DROPOUT = 0.1



        self.conv1 = nn.Conv2d(

            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(planes)

        self.dropout = nn.Dropout(DROPOUT)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,

                               stride=1, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)

        self.dropout = nn.Dropout(DROPOUT)



        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != self.expansion*planes:

            self.shortcut = nn.Sequential(

                nn.Conv2d(in_planes, self.expansion*planes,

                          kernel_size=1, stride=stride, bias=False),

                nn.BatchNorm2d(self.expansion*planes),

                nn.Dropout(DROPOUT)

            )



    def forward(self, x):

        out = F.relu(self.dropout(self.bn1(self.conv1(x))))

        out = self.dropout(self.bn2(self.conv2(out)))

        out += self.shortcut(x)

        out = F.relu(out)

        return out





class ResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10):

        super(ResNet, self).__init__()

        self.in_planes = 64



        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,

                               stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)

        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)

        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)

        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.linear = nn.Linear(512*block.expansion, num_classes)



    def _make_layer(self, block, planes, num_blocks, stride):

        strides = [stride] + [1]*(num_blocks-1)

        layers = []

        for stride in strides:

            layers.append(block(self.in_planes, planes, stride))

            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)



    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)

        out = self.layer2(out)

        out = self.layer3(out)

        out = self.layer4(out)

        out = F.avg_pool2d(out, 4)

        out = out.view(out.size(0), -1)

        out = self.linear(out)

        return F.log_softmax(out, dim=-1)





def ResNet18():

    return ResNet(BasicBlock, [2, 2, 2, 2])
# Importing Model and printing Summary

model = ResNet18().to(device)

summary(model, input_size=(3,32,32))
from tqdm import tqdm

from torch import nn

import torch.nn

from torch.functional import F

import os





def model_training(model, device, train_dataloader, optimizer, train_acc, train_losses):

            

    model.train()

    pbar = tqdm(train_dataloader)

    correct = 0

    processed = 0

    running_loss = 0.0



    for batch_idx, (data, target) in enumerate(pbar):

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        y_pred = model(data)

        loss = F.nll_loss(y_pred, target)

        



        train_losses.append(loss)

        loss.backward()

        optimizer.step()



        pred = y_pred.argmax(dim=1, keepdim=True)

        correct += pred.eq(target.view_as(pred)).sum().item()

        processed += len(data)

        # print statistics

        running_loss += loss.item()

        pbar.set_description(desc=f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

        train_acc.append(100*correct/processed)
import torch

import os

from torch.functional import F



cwd = os.getcwd()



def model_testing(model, device, test_dataloader, test_acc, test_losses, misclassified = []):

    

    model.eval()

    test_loss = 0

    correct = 0

    class_correct = list(0. for i in range(10))

    class_total = list(0. for i in range(10))

    # label = 0

    classes = ('plane', 'car', 'bird', 'cat',

           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    

    with torch.no_grad():



        for index, (data, target) in enumerate(test_dataloader):

            data, target = data.to(device), target.to(device)

            output = model(data)

            pred = output.argmax(dim=1, keepdim=True)

            

            for d,i,j in zip(data, pred, target):

                if i != j:

                    misclassified.append([d.cpu(),i[0].cpu(),j.cpu()])



            test_loss += F.nll_loss(output, target, reduction='sum').item()

            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_dataloader.dataset)

    test_losses.append(test_loss)

    

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(

        test_loss, correct, len(test_dataloader.dataset),

        100. * correct / len(test_dataloader.dataset)))

    

    test_acc.append(100. * correct / len(test_dataloader.dataset))

    return misclassified
# Training the model



criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.05, patience=2, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)

# scheduler = StepLR(optimizer, step_size=15, gamma=0.1)



train_acc = []

train_losses = []

test_acc = []

test_losses = []



EPOCHS = 40



for i in range(EPOCHS):

    print(f'EPOCHS : {i}')

    model_training(model, device, trainloader, optimizer, train_acc, train_losses)

    scheduler.step(train_losses[-1])

    misclassified = model_testing(model, device, testloader, test_acc, test_losses)
fig, axs = plt.subplots(2,2, figsize=(25,20))



axs[0,0].set_title('Train Losses')

axs[0,1].set_title('Training Accuracy')

axs[1,0].set_title('Test Losses')

axs[1,1].set_title('Test Accuracy')



axs[0,0].plot(train_losses)

axs[0,1].plot(train_acc)

axs[1,0].plot(test_losses)

axs[1,1].plot(test_acc)