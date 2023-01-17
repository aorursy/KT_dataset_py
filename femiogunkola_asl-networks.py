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

print(os.listdir('../input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/'))
import numpy as np

import pandas as pd

import torch

import torchvision

from torchvision import transforms , datasets

from torch.utils.data import DataLoader



image_transforms = {

    # Train uses data augmentation

    'train':

    transforms.Compose([

        transforms.RandomResizedCrop(size=200, scale=(0.8, 1.0)),

        transforms.RandomRotation(degrees=15),

        transforms.ColorJitter(),

        transforms.RandomHorizontalFlip(),

        transforms.CenterCrop(size=180),  # Image net standards

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406],

                             [0.229, 0.224, 0.225])  # Imagenet standards

        

        ]),

    # Validation does not use augmentation

    'test':

    transforms.Compose([

        transforms.Resize(size=200),

        transforms.CenterCrop(size=180),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]),

}
traindir = "../input/asl-alphabet/asl_alphabet_train/asl_alphabet_train/"

testdir ="../input/asl-alphabet/asl_alphabet_train/asl_alphabet_train/"

data = {

    'train':

    datasets.ImageFolder(root=traindir, transform=image_transforms['train']),

    'test':

   datasets.ImageFolder(root=testdir, transform=image_transforms['test']),

}



# Dataloader iterators, make sure to shuffle

batch_size =100

dataloaders = {

    'train': DataLoader(data['train'], batch_size=batch_size, shuffle=True),

    'test': DataLoader(data['test'], batch_size=batch_size, shuffle=True)

}



classes = ('a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z' ,'space' ,'delete' ,'nothing')
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

    
trainiter = iter(dataloaders['train'])



train_images, labels = next(trainiter)

train_images.shape, labels.shape



fig, axes = plt.subplots(figsize=(5,4), ncols=10)

for ii in range(batch_size):

    ax = axes[ii]

    imshow(train_images[ii], ax=ax, normalize=False)

    #imshow(torchvision.utils.make_grid(train_images))

print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))



import torch.nn as nn

import torch.nn.functional as F





class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 10, 5)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(10, 16, 5)

        self.fc1 = nn.Linear(16*42*42, 120)

        self.fc2 = nn.Linear(120, 84)

        self.fc3 = nn.Linear(84, 29)



    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))

        x = self.pool(F.relu(self.conv2(x)))

        #print(x.shape)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        

        return x

        





net= Net()

net.to('cuda')
import torch.optim as optim



criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(5):  # loop over the dataset multiple times



    running_loss = 0.0

    for i, data in enumerate(dataloaders['train'], 0):

        # get the inputs; data is a list of [inputs, labels]

        inputs, labels = data[0].to('cuda'), data[1].to('cuda')





        # zero the parameter gradients

        optimizer.zero_grad()



        # forward + backward + optimize

        outputs = net(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()



        # print statistics

        running_loss += loss.item()

        if i % 100 == 99:    # print every 2000 mini-batches

            print('[%d, %5d] loss: %.3f' %

                  (epoch + 1, i + 1, running_loss / 100))

            running_loss = 0.0



print('Finished Training')
PATH = './asl_net.pth'

torch.save(net.state_dict(),PATH)
testiter = iter(dataloaders['test'])

test_images, labels2 = next(testiter)



imshow(torchvision.utils.make_grid(test_images))

print('GroundTruth:',"".join('%5s' % classes[labels2[j]] for j in range (8)))

net = Net()

net.load_state_dict(torch.load(PATH))

outputs = net(test_images)
_,predicted = torch.max(outputs,1)

print('Predicted',"".join('%5s' % classes[predicted[j]] for j in range(4)))
correct = 0

total = 0

with torch.no_grad():

    for test_data in dataloaders['test']:

        test_images, test_labels = test_data

        outputs = net(test_images)

        _, predicted = torch.max(outputs.data,1)

        total += test_labels.size(0)

        correct += (predicted == test_labels).sum().item()

        

print('Accuracy of the network on the 10000 test images: %d %%' % (

    100 * correct / total))
fig, axes = plt.subplots(figsize=(10,4), ncols=10)

for ii in range(10):

    ax = axes[ii]

    imshow(test_images[ii], ax=ax, normalize=False)

    #imshow(torchvision.utils.make_grid(train_images))

print(' '.join('%5s' % classes[test_labels[j]] for j in range(10)))