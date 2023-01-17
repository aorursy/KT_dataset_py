# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import os

import cv2

from torchvision import datasets

import matplotlib.pyplot as pyplot

from tqdm import tqdm_notebook

import numpy as np

import seaborn

import torch

import torchvision

import torch.nn as nn

import torch.nn.functional as F

import torchvision.transforms as transforms

from torch.optim.lr_scheduler import StepLR



import matplotlib.pyplot as plt



%matplotlib inline
# check if CUDA is available

train_on_gpu = torch.cuda.is_available()



if not train_on_gpu:

    print('CUDA is not available.  Training on CPU ...')

else:

    print('CUDA is available!  Training on GPU ...')
data_dir = "./data/"



train_data_dir = '/kaggle/input/horses-or-humans-dataset/horse-or-human/train/'

validation_data_dir = '/kaggle/input/horses-or-humans-dataset/horse-or-human/validation/'



loader = transforms.Compose([

#     transforms.Resize((300, 300)),

#     transforms.CenterCrop(256),

    transforms.ToTensor(),

    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

])



train_data = datasets.ImageFolder(train_data_dir, transform=loader)

validation_data = datasets.ImageFolder(validation_data_dir, transform=loader)



train_loader = torch.utils.data.DataLoader(train_data, batch_size=48, num_workers=os.cpu_count(), pin_memory=True)

validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=48, num_workers=os.cpu_count(), pin_memory=True)



unloader = transforms.ToPILImage()
import matplotlib.pyplot as plt

%matplotlib inline

    

# obtain one batch of training images

dataiter = iter(train_loader)

images, labels = dataiter.next()

images = images.numpy()

print(images.shape)

images = images[28:,2,:,:]

print(images.shape)

# plot the images in the batch, along with the corresponding labels

fig = plt.figure(figsize=(25, 4))

for idx in np.arange(20):

    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])

    ax.imshow(np.squeeze(images[idx]), 'gray')
class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3)  

        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3)  

        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)  

        self.conv4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)  

        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)  



        # max pooling layer

        self.pool = nn.MaxPool2d(2, 2)

        

        # linear layer (64 * 8 * 8 -> 512)

        self.fc1 = nn.Linear(4096, 512)

        

        # linear layer (512 -> 1)

        self.fc2 = nn.Linear(512, 1)



        # dropout_layer (p=0.25)

        self.dropout = nn.Dropout(0.25)



    def forward(self, x):

        x = F.relu(self.conv1(x)) # 3x300x300 => 4x298x298

        x = F.relu(self.conv2(x)) # 4x298x298 => 8x296x296

        x = self.pool(x) # 8x296x296 => 8x148x148

        x = self.pool(x) # 8x148x148 => 8x74x74

        x = F.relu(self.conv3(x)) # 8x74x74 => 16x72x72

        x = self.pool(x) # 16x72x72 => 16x36x36

        x = F.relu(self.conv4(x)) # 16x36x36 => 32x34x34

        x = F.relu(self.conv5(x)) # 32x34x34 => 64x32x32

        x = self.pool(x) # 64x32x32 => 64x16x16

        x = self.pool(x) # 64x16x16 => 64x8x8



        # flatten image input

        x = x.view(-1, 64 * 8 * 8)

        # add dropout layer

        x = self.dropout(x)

        # add 1st hidden layer, with relu activation function

        x = F.relu(self.fc1(x))

        # add dropout layer

        x = self.dropout(x)

        # add 2nd hidden layer

        x = self.fc2(x)

        return x
model = Net()

print(model)

if train_on_gpu:

    model.cuda()
import torch.optim as optim



# criterion = nn.BCELoss()

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)
valid_loss_min = np.Inf

n_epochs = 25



# gamma = decaying factor

scheduler = StepLR(optimizer, step_size=2, gamma=0.9)



for epoch in range(1, n_epochs+1):

    train_loss = 0.0

    

    print('Epoch:', epoch,'LR:', scheduler.get_lr())

    

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):

        target = target.unsqueeze(1)



        if train_on_gpu:

            data, target = data.cuda(), target.cuda()

        

        target = target.type(torch.FloatTensor)

        optimizer.zero_grad()

        output = model(data)

        

        output = output.type(torch.FloatTensor)



        loss  = criterion(output, target)

        loss.backward()

        optimizer.step()

        

        train_loss += loss.item()*32

        

    print("Epoch: {} \tTraining Loss: {:.6f}".format(

        epoch, train_loss/len(train_loader)))

    

    # Decay Learning Rate

    scheduler.step()
model.eval()

validation_loss = 0.0

for batch_idx, (data, target) in enumerate(validation_loader):

    target = target.unsqueeze(1)



    # move tensors to GPU if CUDA is available

    if train_on_gpu:

        data, target = data.cuda(), target.cuda()

    # forward pass: compute predicted outputs by passing inputs to the model

    output = model(data)

    # calculate the batch loss

    target = target.type(torch.FloatTensor).cuda()

    loss = criterion(output, target)

    # update average validation loss 

    validation_loss += loss.item()*32



validation_loss = validation_loss/len(validation_loader)

    

    

    

print("Validation loss {:.6f}".format(validation_loss/len(validation_loader)))