# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#import numpy as np # linear algebra

#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import torch

import matplotlib.pyplot as plt

import pandas as pd

from torchvision import datasets, transforms

from torch import optim, nn

from torch.utils.data import Dataset, TensorDataset, DataLoader

import torch.nn.functional as F
# credit: intro to Deep Learning with Pytorch by Facebook

# Udacity Course.

# https://github.com/udacity/deep-learning-v2-pytorch/blob/master/intro-to-pytorch/helper.py



import numpy as np

import matplotlib.pyplot as plt



def imshow(image, ax=None, title=None, normalize=True):

    """Imshow for Tensor."""

    if ax is None:

        fig, ax = plt.subplots()

    image = image.numpy().transpose((1, 2, 0))



    if normalize:

        mean = np.array([0.485, 0.456, 0.406])

        std = np.array([0.229, 0.224, 0.225])

        image = std * image + mean

        image = np.clip(image, 0, 1)



    ax.imshow(image)

    ax.spines['top'].set_visible(False)

    ax.spines['right'].set_visible(False)

    ax.spines['left'].set_visible(False)

    ax.spines['bottom'].set_visible(False)

    ax.tick_params(axis='both', length=0)

    ax.set_xticklabels('')

    ax.set_yticklabels('')



    return ax



# credit: I copied this method from "https://www.kaggle.com/bonhart/simple-cnn-on-pytorch-for-beginers"

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
"""

!cp -r /kaggle/input/aerial-cactus-identification /kaggle/working/

!unzip /kaggle/working/aerial-cactus-identification/train.zip -d /kaggle/working

!unzip /kaggle/working/aerial-cactus-identification/test.zip -d /kaggle/working

"""
import zipfile



train_path = zipfile.ZipFile("../input/aerial-cactus-identification/train.zip", 'r') 

train_path.extractall("/kaggle/working")



train_path.close()



test_path = zipfile.ZipFile('../input/aerial-cactus-identification/test.zip', 'r')

test_path.extractall('/kaggle/working')



test_path.close()
from sklearn.model_selection import train_test_split

import cv2

import PIL

import os

#labels_data = pd.read_csv(data)



data = pd.read_csv("../input/aerial-cactus-identification/train.csv")

data.sample(10)



train, test = train_test_split(data,

                              stratify=data.has_cactus,

                              test_size=.2,

                              random_state=42)
data.head()
data['has_cactus'].value_counts()
train
train.shape, test.shape
train['has_cactus'].value_counts()
test['has_cactus'].value_counts()
TRAIN_DIR = '/kaggle/working/train'



train_transforms = transforms.Compose([

    transforms.ToPILImage(),

    transforms.Pad(32, padding_mode='reflect'),

    #transforms.RandomResizedCrop(224),

    transforms.ToTensor(),

    transforms.Normalize(mean = [0.44, 0.44, 0.44], 

                         std = [0.24, 0.24, 0.24])])



test_transforms = transforms.Compose([

    transforms.ToPILImage(),

    transforms.Pad(32, padding_mode='reflect'),

    #transforms.RandomResizedCrop(224),

    transforms.ToTensor(),

    transforms.Normalize(mean = [0.44, 0.44, 0.44], 

                         std = [0.24, 0.24, 0.24])])



train_data = MyDataset(train, data_dir=TRAIN_DIR, transform=train_transforms)

test_data = MyDataset(test, data_dir=TRAIN_DIR, transform=test_transforms)



trainloader = torch.utils.data.DataLoader(train_data, batch_size = 128, shuffle = True)

testloader = torch.utils.data.DataLoader(test_data, batch_size = 64, shuffle = True)
trainimages, trainlabels = next(iter(trainloader))



fig, axes = plt.subplots(figsize=(12, 12), ncols=5)

print('training images')

for i in range(5):

    axe1 = axes[i] 

    imshow(trainimages[i], ax=axe1, normalize=False)



print(trainimages[0].size())
testimages, testlabels = next(iter(testloader))



fig, axes = plt.subplots(figsize=(12, 12), ncols=5)

print('test images')

for i in range(5):

    axe2 = axes[i]

    imshow(testimages[i], ax=axe2, normalize=False)

print(testimages[0].size())
class cnn(nn.Module):

    def __init__(self):

        super(cnn, self).__init__()

        

                # Convolutional Neural Networks

        self.conv1 = nn.Conv2d(3, 32, 3)

        self.conv2 = nn.Conv2d(32, 64, 3)

        self.conv3 = nn.Conv2d(64, 128, 3)

        self.conv4 = nn.Conv2d(128, 256, 3)

        self.conv5 = nn.Conv2d(256, 512, 3)

        

        self.bn1 = nn.BatchNorm2d(32)

        self.bn2 = nn.BatchNorm2d(64)

        self.bn3 = nn.BatchNorm2d(128)

        self.bn4 = nn.BatchNorm2d(256)

        #self.bn5 = nn.BatchNorm2d(512)

                  

        self.fc1 = nn.Linear(4096, 2)

        

        # pooling and dropout layer

        self.pool = nn.MaxPool2d(2, 2)

        # self.dropout = nn.Dropout(0.2)

        

    def forward(self, x):

        x = self.bn1(self.pool(F.leaky_relu(self.conv1(x))))

        x = self.bn2(self.pool(F.leaky_relu(self.conv2(x))))

        x = self.bn3(self.pool(F.leaky_relu(self.conv3(x))))

        x = self.bn4(self.pool(F.leaky_relu(self.conv4(x))))

        #x = self.bn5(self.pool(F.leaky_relu(self.conv5(x))))

        

        # reshape to fit into fully connected net

        x = x.view(x.shape[0],-1)

        x = self.fc1(x)

        x = F.log_softmax(x, dim=1)

        

        return x
model = cnn()

print(model)
criterion = nn.NLLLoss() #nn.NLLLoss() # 

optimizer = torch.optim.Adam(model.parameters(), lr=0.3)

epochs = 5
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

device



model = cnn().to(device)

print(model)
train_losses, test_losses = [], []



for e in range(epochs):

    running_loss = 0

    

    for images, labels in trainloader:

        

        images = images.to(device)

        labels = labels.to(device)

        

        #ititialize gradients

        optimizer.zero_grad()

        

        # evaluate the loss

        output = model(images)

        loss = criterion(output, labels)

        

        # backpropagate the weights

        loss.backward()

        optimizer.step()

        

        running_loss += loss.item()

    

    # once complete the computation for one training batch size,

    else: 

        test_loss = 0

        accuracy = 0

        

        # turn off calculating gradients for validation

        # to save memory and time

        

        with torch.no_grad():

            model.eval() #turn off dropout to evaludate validation set

            

            for images, labels in testloader:

                images = images.to(device)

                labels = labels.to(device)



                los_ps = model(images)

                test_loss += criterion(los_ps, labels)

                

                ps = torch.exp(los_ps)

                poss, top_class = ps.topk(1, dim=1)

                

                equals = top_class == labels.view(*top_class.shape)

                accuracy += torch.mean(equals.type(torch.FloatTensor))

        

        # turn on dropout again for the next training.

        model.train() 

        

        train_losses.append(running_loss/len(trainloader))

        test_losses.append(test_loss/len(testloader))

        

        print("epoch: {}/{}  ".format(e+1, epochs),

             "training loss: {:.3f} ".format(train_losses[-1]),

             "test loss: {:.3f} ".format(test_losses[-1]),

             "test accuracy: {:.3f} ".format(accuracy/len(testloader)))
import matplotlib.pyplot as plt

%matplotlib inline

%config InlineBackend.figure_format = 'retina'
plt.plot(train_losses, label = 'Training Loss')

plt.plot(test_losses, label = 'Test Loss')

plt.legend(frameon=False)
## Parameters for model

# Hyper parameters

num_epochs = 25

num_classes = 2

batch_size = 128

learning_rate = 0.002



# Device configuration

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
"""

# Image preprocessing

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



trainloader = DataLoader(dataset = dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)

testloader = DataLoader(dataset = dataset_valid, batch_size=batch_size//2, shuffle=False, num_workers=0)

"""
# NOTE: class is inherited from nn.Module

class SimpleCNN(nn.Module):

    def __init__(self):

        # ancestor constructor call

        super(SimpleCNN, self).__init__()

        

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=2) # match out_ch and next in_ch

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
cnnmodel = SimpleCNN().to(device)
# Loss and optimizer

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adamax(cnnmodel.parameters(), lr=learning_rate)



epochs = 5
train_losses, test_losses = [], []



for e in range(epochs):

    running_loss = 0

    

    for images, labels in trainloader:

        

        images = images.to(device)

        labels = labels.to(device)

        

        #ititialize gradients

        optimizer.zero_grad()

        

        # evaluate the loss

        output = cnnmodel(images)

        loss = criterion(output, labels)

        

        # backpropagate the weights

        loss.backward()

        optimizer.step()

        

        running_loss += loss.item()

    

    # once complete the computation for one training batch size,

    else: 

        test_loss = 0

        accuracy = 0

        

        # turn off calculating gradients for validation

        # to save memory and time

        

        with torch.no_grad():

            cnnmodel.eval() #turn off dropout to evaludate validation set

            

            for images, labels in testloader:

                images = images.to(device)

                labels = labels.to(device)



                los_ps = cnnmodel(images)

                test_loss += criterion(los_ps, labels)

                

                ps = torch.exp(los_ps)

                poss, top_class = ps.topk(1, dim=1)

                

                equals = top_class == labels.view(*top_class.shape)

                accuracy += torch.mean(equals.type(torch.FloatTensor))

        

        # turn on dropout again for the next training.

        cnnmodel.train() 

        

        train_losses.append(running_loss/len(trainloader))

        test_losses.append(test_loss/len(testloader))

        

        print("epoch: {}/{}  ".format(e+1, epochs),

             "training loss: {:.3f} ".format(train_losses[-1]),

             "test loss: {:.3f} ".format(test_losses[-1]),

             "test accuracy: {:.3f} ".format(accuracy/len(testloader)))
import matplotlib.pyplot as plt

%matplotlib inline

%config InlineBackend.figure_format = 'retina'
plt.plot(train_losses, label = 'Training Loss')

plt.plot(test_losses, label = 'Test Loss')

plt.legend(frameon=False)