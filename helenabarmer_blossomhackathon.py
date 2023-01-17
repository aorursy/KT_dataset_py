# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import json

from collections import OrderedDict



import matplotlib.pyplot as plt



import torch

from torch import nn

from torch import optim



import torchvision

from torchvision import datasets, transforms, models

from torch.autograd import Variable



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data_dir = '../input/flower_data/flower_data'

train_dir = data_dir + '/train'

valid_dir = data_dir + '/valid'

test_dir = '../input/test set/'



train_transforms = transforms.Compose([transforms.RandomRotation(30),

                                       transforms.RandomResizedCrop(224),

                                       transforms.RandomHorizontalFlip(),

                                       transforms.ToTensor(),

                                       transforms.Normalize([0.485, 0.456, 0.406],

                                                            [0.229, 0.224, 0.225])])



valid_transforms = transforms.Compose([transforms.Resize(256),

                                       transforms.CenterCrop(224),

                                       transforms.ToTensor(),

                                       transforms.Normalize([0.485, 0.456, 0.406],

                                                            [0.229, 0.224, 0.225])])



test_transforms = transforms.Compose([transforms.Resize(256),

                                       transforms.CenterCrop(224),

                                       transforms.ToTensor(),

                                       transforms.Normalize([0.485, 0.456, 0.406],

                                                            [0.229, 0.224, 0.225])])



train_data = datasets.ImageFolder(train_dir, transform=train_transforms)

valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

test_data = datasets.ImageFolder(test_dir, transform=test_transforms)



trainloader = torch.utils.data.DataLoader(train_data, batch_size=20, shuffle=True)

validloader = torch.utils.data.DataLoader(valid_data, batch_size=20)

testloader = torch.utils.data.DataLoader(test_data, batch_size=20)



print("Training data: {}".format(len(train_data)))

print("Validation data: {}".format(len(valid_data)))

print("Test data: {}".format(len(test_data)))
def imshow(image, ax=None, title=None, normalize=True):

    """Imshow for Tensor."""

    if ax is None:

        fig, ax = plt.subplots()

    image = image.numpy().transpose((1, 2, 0))

    plt.axis('off')

    plt.imshow(image)

   



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
def show_databatch(images, labels):

    out = torchvision.utils.make_grid(images)

    imshow(out, title=[train_transforms for x in labels])



# Get a batch of training data

images, labels = next(iter(trainloader))

show_databatch(images, labels)



# Show one image

imshow(images[0], normalize=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



model = models.densenet121(pretrained=True)



for param in model.parameters():

    param.requires_grad = False

    

model.classifier = nn.Sequential(nn.Linear(1024, 256),

                                 nn.ReLU(),

                                 nn.Dropout(0.2),

                                 nn.Linear(256, 102),

                                 nn.LogSoftmax(dim=1))





criterion = nn.NLLLoss()

optimizer = optim.Adam(model.classifier.parameters(), lr=0.03)

model.to(device)
epochs = 1

steps = 0

running_loss = 0

print_every = 5



for epoch in range(epochs):

    for images, labels in trainloader:

        steps += 1

        images, labels = images.to(device), labels.to(device)

        

        optimizer.zero_grad()

        

        logps = model.forward(images)

        loss = criterion(logps, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        

        if steps % print_every == 0:

            model.eval()

            test_loss = 0

            accuracy = 0

            

            with torch.no_grad():

                for images, labels in testloader:

                    images, labels = images.to(device), labels.to(device)

                    

                    logps = model.forward(images)

                    loss = criterion(logps, labels)

                    

                    test_loss += loss.item()

                    

                    ps = torch.exp(logps)

                    top_ps, top_class = ps.topk(1, dim=1)

                    equals = top_class == labels.view(*top_class.shape)

                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                

            print(f"Epoch {epoch+1}/{epochs}.."

                  f"Train loss: {running_loss/print_every:.3f}.."

                  f"Test loss: {test_loss/len(testloader):.3f}.."

                  f"Test accuracy: {accuracy/len(testloader):.3f}")

            running_loss = 0

            model.train()