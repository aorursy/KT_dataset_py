# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd 

import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import datasets, transforms

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
test_data=pd.read_csv("/kaggle/input/Kannada-MNIST/test.csv")

train_data=pd.read_csv("/kaggle/input/Kannada-MNIST/train.csv")

train_data.head()
labels=train_data.label.values

images = train_data.loc[:,train_data.columns != "label"].values

print(labels.shape)

print(images.shape)
images=torch.from_numpy(images).type(torch.FloatTensor)

labels=torch.from_numpy(labels)

print(images.dtype)
model = nn.Sequential(nn.Linear(784, 128),

                      nn.ReLU(),

                      nn.Linear(128, 64),

                      nn.ReLU(),

                      nn.Linear(64, 10),

                      nn.LogSoftmax(dim=1))

print(model)

model

criterion = nn.NLLLoss()

optimizer = optim.SGD(model.parameters(), lr=0.01)
train = torch.utils.data.TensorDataset(images,labels,)

train_loader = torch.utils.data.DataLoader(train, batch_size=64)
epochs = 30

for e in range(epochs):

    running_loss = 0

    for images, labels in train_loader:

        # Flatten MNIST images into a 784 long vector

        images = images.view(images.shape[0],-1)

        labels=labels

        optimizer.zero_grad()

        output = model(images)

        loss = criterion(output, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    else:

        print(f"Training loss : {running_loss/len(train_loader)}")
import matplotlib.pyplot as plt

import numpy as np

from torch import nn, optim

from torch.autograd import Variable





def test_network(net, trainloader):



    criterion = nn.MSELoss()

    optimizer = optim.Adam(net.parameters(), lr=0.001)



    dataiter = iter(trainloader)

    images, labels = dataiter.next()



    # Create Variables for the inputs and targets

    inputs = Variable(images)

    targets = Variable(images)



    # Clear the gradients from all Variables

    optimizer.zero_grad()



    # Forward pass, then backward pass, then update weights

    output = net.forward(inputs)

    loss = criterion(output, targets)

    loss.backward()

    optimizer.step()



    return True





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





def view_recon(img, recon):

    ''' Function for displaying an image (as a PyTorch Tensor) and its

        reconstruction also a PyTorch Tensor

    '''



    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)

    axes[0].imshow(img.numpy().squeeze())

    axes[1].imshow(recon.data.numpy().squeeze())

    for ax in axes:

        ax.axis('off')

        ax.set_adjustable('box-forced')



def view_classify(img, ps, version="MNIST"):

    ''' Function for viewing an image and it's predicted classes.

    '''

    ps = ps.data.numpy().squeeze()



    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)

    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())

    ax1.axis('off')

    ax2.barh(np.arange(10), ps)

    ax2.set_aspect(0.1)

    ax2.set_yticks(np.arange(10))

    if version == "MNIST":

        ax2.set_yticklabels(np.arange(10))

   

    ax2.set_title('Class Probability')

    ax2.set_xlim(0, 1.1)



    plt.tight_layout()

    plt.show()

images, labels = next(iter(train_loader))





img = images[0].view(1, 784)

# Turn off gradients to speed up this part

with torch.no_grad():

    logps = model(img).cpu()



# Output of the network are log-probabilities, need to take exponential for probabilities

ps = torch.exp(logps)

view_classify(img.view(1, 28, 28), ps)