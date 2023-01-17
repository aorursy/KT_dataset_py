import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
%matplotlib inline

import torch

import torchvision

import torchvision.transforms as transforms

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

# The following is to downlaod and use pretrained models from the model-zoo

import torchvision.models as models

#--------------------------------------------------------------------------

import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm
# Initializing Hyperparameters

batch_size = 256

# Learning rate

lr = 0.001

# Number of training epochs

num_epochs = 5

# Device configuration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)
!tar -zxvf ../input/cifar10-pytorch/cifar10_pytorch/data/cifar-10-python.tar.gz

os.listdir('.')
normalize = transforms.Normalize(

   mean=[0.485, 0.456, 0.406],

   std=[0.229, 0.224, 0.225]

)

trans = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),normalize])



trainset = torchvision.datasets.CIFAR10(root='.', train=True, download=False, transform=trans)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)



testset = torchvision.datasets.CIFAR10(root='.', train=False, download=False, transform=trans)

testloader = torch.utils.data.DataLoader(testset, batch_size=2*batch_size, shuffle=False, num_workers=2)
#CLASSES IN CIFAR10

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# Download the pretrained Resnet50 model and load it to the device (GPU)

resnet50 = models.resnet50(pretrained=True)

resnet50.load_state_dict(torch.load("../input/resnet50/resnet50.pth"))

resnet50 = resnet50.to(device)

#Make sure to swith-on the internet option in the Workspace
# Freeze the layers

for param in resnet50.parameters():

    param.requires_grad = False
# Change the last layer to cifar10 number of output classes.

# Also unfreeze the penultimate layer. We will finetune just these two layers.

resnet50.fc = nn.Sequential(

                      nn.Linear(2048, 256), 

                      nn.ReLU(), 

                      nn.Linear(256, 10)

)
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(resnet50.parameters(), lr=0.001)
#TRAINING THE NETWORK



resnet50 = resnet50.to(device)

for epoch in range(num_epochs):  # loop over the dataset multiple times

    running_loss = 0.0

    pbar = tqdm(trainloader)

    i = 0

    for data in pbar:

        # get the inputs

        inputs, labels = data

        inputs = inputs.to(device)

        labels = labels.to(device)



        # zero the parameter gradients

        optimizer.zero_grad()



        # forward + backward + optimize

        outputs = resnet50(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()



        # print statistics

        running_loss += loss.item()

        

        pbar.set_description("Processing epoch {:d} minibatch {:d} train loss {:.3f}".format(epoch,\

                                                            i+1, running_loss/(i+1)))

        i += 1



print('Finished Training')
correct = 0

total = 0

i = 0

with torch.no_grad():

    pbar = tqdm(testloader)

    for data in pbar:

        images, labels = data

        images = images.to(device)

        labels = labels.to(device)

        outputs = resnet50(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

        pbar.set_description("minibatch {:d} test accuracy {:4.2f}%".format(i+1,\

                                                            100.0*correct/total))

        i += 1



print('Accuracy of the network on the 10000 test images: %4.2f %%' % (100.0 * correct / total))