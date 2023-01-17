import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

import os

import multiprocessing as mp
import torch

import torch.nn as nn

import torch.nn.functional as F

from torchvision import datasets

import torchvision.transforms as transforms

from torchvision import models

from torch.utils.data.sampler import SubsetRandomSampler



def mish(x):

    return (x*torch.tanh(F.softplus(x)))
import cv2

import matplotlib.pyplot as plt



train_on_gpu = torch.cuda.is_available()



if not train_on_gpu:

    print('CUDA is not available.  Training on CPU ...')

else:

    print('CUDA is available!  Training on GPU ...')
df = pd.read_csv('../input/train.csv')

df.head()
df['has_cactus'].value_counts(normalize=True)
train_df, val_df = train_test_split(df, stratify = df.has_cactus, test_size=.2)
#Checking that validation set has same proportions as original training data

val_df['has_cactus'].value_counts(normalize=True)
#Build a class for our data to put our images and target variables into our pytorch dataloader

# https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel



class DataSet(torch.utils.data.Dataset):

    def __init__(self, labels, data_directory, transform=None):

        super().__init__()

        self.labels = labels.values

        self.data_dir = data_directory

        self.transform=transform

        

    def __len__(self):

        return len(self.labels)

    

    def __getitem__(self, index):

        name, label = self.labels[index]

        img_path = os.path.join(self.data_dir, name)

        img = cv2.imread(img_path)

        

        if self.transform is not None:

            img = self.transform(img)

        return img, label
batch_size = 32



# Transform training data with random flips and normalize it to prepare it for dataloader

train_transforms = transforms.Compose([transforms.ToPILImage(),

                                       transforms.RandomHorizontalFlip(),

                                       transforms.ToTensor(),

                                      transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])



val_transforms = transforms.Compose([transforms.ToPILImage(),

                                     transforms.ToTensor(),

                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])



train_data = DataSet(train_df,'../input/train/train', transform = train_transforms)

val_data = DataSet(val_df,'../input/train/train', transform = val_transforms)



train_data_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers=mp.cpu_count())

val_data_loader = torch.utils.data.DataLoader(val_data, batch_size = batch_size, shuffle = True, num_workers=mp.cpu_count())
#Checking what our cactus look like

fig,ax = plt.subplots(1,3,figsize=(15,5))



for i, idx in enumerate(train_df[train_df['has_cactus']==1]['id'][0:3]):

  path = os.path.join('../input/train/train',idx)

  ax[i].imshow(cv2.imread(path))
#Building a CNN from scratch

act = mish

class Net(nn.Module):

    def __init__(self):

        super().__init__()

        

        self.conv1 = nn.Conv2d(3, 16, 3, padding = 1)

        self.conv2 = nn.Conv2d(16, 32, 3, padding = 1)

        self.conv3 = nn.Conv2d(32, 64, 3, padding = 1)

        self.conv4 = nn.Conv2d(64, 128, 3, padding = 1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(2*16*16, 256)

        self.fc2 = nn.Linear(256, 128)

        self.fc3 = nn.Linear(128, 64)

        self.fc4 = nn.Linear(64, 2)

        self.dropout = nn.Dropout(p = .25)

        

    def forward(self, x):

        

        x = self.pool(act(self.conv1(x)))

        x = self.pool(act(self.conv2(x)))

        x = self.pool(act(self.conv3(x)))

        x = self.pool(act(self.conv4(x)))

        

        x = x.view(-1, 2*16*16)

        x = self.dropout(x)

        x = act(self.fc1(x))

        x = self.dropout(x)

        x = act(self.fc2(x))

        x = self.dropout(x)

        x = act(self.fc3(x))

        x = self.dropout(x)

        x = act(self.fc4(x))

        

        return x

        
model = Net()

if train_on_gpu:

    model = model.cuda()



epochs = 10

learning_rate = .0003



criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)
#Training and validation for model



best_loss = np.Inf

best_model = Net()

if train_on_gpu:

    best_model.cuda()



for epoch in range(1, epochs+1):

    train_loss = 0

    val_loss = 0

    

    model.train()

    for images, labels in train_data_loader:

        

        if train_on_gpu:

            images = images.cuda()

            labels = labels.cuda()

            

        optimizer.zero_grad()

        out = model(images)

        loss = criterion(out, labels)

        

        loss.backward()

        optimizer.step()

        

        train_loss += loss.item()

        #print('Loss: {}'.format(loss.item()))

        

    model.eval()

    for images, labels in val_data_loader:

        

        if train_on_gpu:

            images = images.cuda()

            labels = labels.cuda()

            

        out = model(images)

        loss = criterion(out, labels)

        

        val_loss += loss.item()

        

    train_loss = train_loss/len(train_data_loader.dataset)

    val_loss = val_loss/len(val_data_loader.dataset)  

    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, val_loss))

    

    #Saving the weights of the best model according to validation score

    if val_loss < best_loss:

        best_loss = val_loss

        print('Improved Model Score - Updating Best Model Parameters...')

        best_model.load_state_dict(model.state_dict())

        

        
#Check model accuracy

best_model.eval()

with torch.no_grad():

    correct = 0

    total = 0

    for images, labels in val_data_loader:

        if train_on_gpu:

            images = images.cuda()

            labels = labels.cuda()

        outputs = best_model(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

          

    print('Test Accuracy: {} %'.format(100 * correct / total))
vgg = models.vgg16()
class mish_layer(nn.Module):

    def __init__(self):

        super(mish_layer, self).__init__()

        

    def forward(self, input):

        return mish(input)
act = mish_layer()



vgg.classifier[1] = act

vgg.classifier[4] = act



for param in vgg.parameters():

    param.requires_grad=False

vgg.classifier[6] = nn.Linear(4096, 2)
vgg = vgg.cuda()



epochs = 10

learning_rate = .0003



criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adamax(vgg.parameters(), lr=learning_rate)
best_loss = np.Inf

best_model = Net()

if train_on_gpu:

    best_model.cuda()



for epoch in range(1, epochs+1):

    train_loss = 0

    val_loss = 0

    

    vgg.train()

    for images, labels in train_data_loader:

        

        if train_on_gpu:

            images = images.cuda()

            labels = labels.cuda()

            

        optimizer.zero_grad()

        out = vgg(images)

        loss = criterion(out, labels)

        

        loss.backward()

        optimizer.step()

        

        train_loss += loss.item()

        #print('Loss: {}'.format(loss.item()))

        

    vgg.eval()

    for images, labels in val_data_loader:

        

        if train_on_gpu:

            images = images.cuda()

            labels = labels.cuda()

            

        out = vgg(images)

        loss = criterion(out, labels)

        

        val_loss += loss.item()

        

    train_loss = train_loss/len(train_data_loader.dataset)

    val_loss = val_loss/len(val_data_loader.dataset)  

    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, val_loss))

    

    #Saving the weights of the best model according to validation score

    if val_loss < best_loss:

        best_loss = val_loss

        print('Improved Model Score - Updating Best Model Parameters...')

        best_model.load_state_dict(model.state_dict())
#Check model accuracy

best_model.eval()

with torch.no_grad():

    correct = 0

    total = 0

    for images, labels in val_data_loader:

        if train_on_gpu:

            images = images.cuda()

            labels = labels.cuda()

        outputs = best_model(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

          

    print('Test Accuracy: {} %'.format(100 * correct / total))