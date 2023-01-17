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

%matplotlib inline

%config InlineBackend.figure_format = 'retina'



import torch

from torch import nn

import torch.nn.functional as F

from torch.autograd import Variable

from torch.optim import lr_scheduler

from torch.utils.data.sampler import SubsetRandomSampler

import torch.optim as optim

import torchvision

from torchvision import datasets, transforms, models

from collections import OrderedDict



import  helper

import time

import copy





from PIL import Image



print("Imported dependency")
#creating the netowrk

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device
#kaggle kernel dependent dataset 

data_dir = '../input/labelledrice/Labelled/'


data_transforms ={

    "train_transforms": transforms.Compose([transforms.RandomRotation(30),

                                           transforms.RandomResizedCrop(224),

                                           transforms.RandomHorizontalFlip(),

                                           transforms.ToTensor(),

                                           transforms.Normalize([0.485, 0.456, 0.406],

                                                                [0.229, 0.224, 0.225])]),

   "valid_transforms": transforms.Compose([transforms.Resize(225),

                                           transforms.CenterCrop(224),

                                           transforms.ToTensor(),

                                           transforms.Normalize([0.485, 0.456, 0.406],

                                                                [0.229, 0.224, 0.225])]), 

    "test_transforms": transforms.Compose([transforms.Resize(225),

                                           transforms.CenterCrop(224),

                                           transforms.ToTensor(),

                                           transforms.Normalize([0.485, 0.456, 0.406],

                                                                [0.229, 0.224, 0.225])])

}
# Split the dataset into train, validation and test

train_data = 0.8

valid_data = 0.1

test_data = 0.1



# Load the datasets with ImageFolder

train_data = datasets.ImageFolder(data_dir, transform=data_transforms["train_transforms"])

valid_data = datasets.ImageFolder(data_dir, transform=data_transforms["valid_transforms"])

test_data = datasets.ImageFolder(data_dir, transform=data_transforms["test_transforms"])



# Obtain training indices that will be used for validation and test

num_train = len(train_data)

indices = list(range(num_train))

# np.random.shuffle(indices)

train_count = int(0.8*num_train)

valid_count = int(0.1*num_train)

test_count = num_train - train_count - valid_count





train_idx = indices[:train_count]

valid_idx = indices[train_count:train_count+valid_count]

test_idx = indices[train_count+valid_count:]



print(len(train_idx), len(valid_idx), len(test_idx))

print("Training", train_count, np.sum(len(train_idx)/num_train))

print("Validation", valid_count, np.sum(len(valid_idx)/num_train))

print("Test", test_count, np.sum(len(test_idx)/num_train))
classes = os.listdir(data_dir)

len(classes)
#train_data = datasets.ImageFolder(data_dir, transform=data_transforms["train_transforms"])



class_names = train_data.classes

print(train_data.classes)
train_sampler = SubsetRandomSampler(train_idx)

valid_sampler = SubsetRandomSampler(valid_idx)

test_sampler = SubsetRandomSampler(test_idx)





# Define the dataloaders using the image datasets

trainloader = torch.utils.data.DataLoader(train_data, batch_size = 16, shuffle = True)

validloader = torch.utils.data.DataLoader(valid_data, batch_size = 8, sampler = valid_sampler)

testloader = torch.utils.data.DataLoader(test_data, batch_size = 8, sampler = test_sampler)

class_names = train_data.classes
import numpy as np

import matplotlib.pyplot as plt



def imshow(inp, title=None):

    inp = inp.numpy().transpose((1, 2, 0))

    # plt.figure(figsize=(10, 10))

    plt.axis('off')

    plt.imshow(inp)

    if title is not None:

        plt.title(title)

    plt.pause(0.001)



def show_databatch(inputs, classes):

    out = torchvision.utils.make_grid(inputs)

    imshow(out, title=[class_names[x] for x in classes])



# Get a batch of training data

inputs, classes = next(iter(trainloader))

show_databatch(inputs, classes)
alexnet = models.alexnet()

model_transfer = models.alexnet(pretrained=True)

#models.vgg19(pretrained=True) 
model_transfer
# If you want to train the model for more than 2 epochs, set this to True after the first run

resume_training = False



if resume_training:

    print("Loading pretrained model..")

    model_transfer.load_state_dict(torch.load('../input/alexnetmodel_ricediseases.pt'))

    print("Loaded!")




use_cuda = torch.cuda.is_available()

if use_cuda:

    print("Using CUDA")



if use_cuda:

    model_transfer.cuda() #.cuda() will move everything to the GPU side

    

criterion = nn.CrossEntropyLoss()



optimizer_ft = optim.SGD(model_transfer.parameters(), lr=0.001, momentum=0.9)

#exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
# Train the model

def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):

    '''returns trained model'''

    # Initialize tracker for minimum validation loss

    valid_loss_min = np.inf

  

    for epoch in range(1, n_epochs+1):

        # In the training loop, I track down the loss

        # Initialize variables to monitor training and validation loss

        train_loss = 0.0

        valid_loss = 0.0

    

        # Model training

        model.train()

        for batch_idx, (data,target) in enumerate(trainloader):

            # 1st step: Move to GPU

            if use_cuda:

                data,target = data.cuda(), target.cuda()

      

            # Then, clear (zero out) the gradient of all optimized variables

            optimizer.zero_grad()

            # Forward pass: compute predicted outputs by passing inputs to the model

            output = model(data)

            # Perform the Cross Entropy Loss. Calculate the batch loss.

            loss = criterion(output, target)

            # Backward pass: compute gradient of the loss with respect to model parameters

            loss.backward()

            # Perform optimization step (parameter update)

            optimizer.step()

            # Record the average training loss

            train_loss = train_loss + ((1/ (batch_idx + 1 ))*(loss.data-train_loss))

      

        # Model validation

        model.eval()

        for batch_idx, (data,target) in enumerate(validloader):

            # Move to GPU

            if use_cuda:

                data, target = data.cuda(), target.cuda()

            # Update the average validation loss

            # Forward pass: compute predicted outputs by passing inputs to the model

            output = model(data)

            # Calculate the batch loss

            loss = criterion(output, target)

            # Update the average validation loss

            valid_loss = valid_loss + ((1/ (batch_idx +1)) * (loss.data - valid_loss))

      

        # print training/validation stats

        print('Epoch: {} \tTraining Loss: {:.5f} \tValidation Loss: {:.5f}'.format(

            epoch,

            train_loss,

            valid_loss))

    

        # Save the model if validation loss has decreased

        if valid_loss <= valid_loss_min:

            print('Validation loss decreased ({:.5f} --> {:.5f}). Saving model ...'.format(

                  valid_loss_min,

                  valid_loss))

            torch.save(model.state_dict(), 'alexnetmodel_ricediseases.pt')

            valid_loss_min = valid_loss

  

    # Return trained model

    return model



# Define loaders transfer

loaders_transfer = {'train': trainloader,

                    'valid': validloader,

                    'test': testloader}



# Train the model

model_transfer = train(10, loaders_transfer, model_transfer, optimizer_ft, criterion, use_cuda, 'alexnetmodel_ricediseases.pt')
model_transfer.load_state_dict(torch.load('alexnetmodel_ricediseases.pt'))
def test(loaders, model, criterion, use_cuda):



    # monitor test loss and accuracy

    test_loss = 0.

    correct = 0.

    total = 0.



    model_transfer.eval()

    for batch_idx, (data, target) in enumerate(loaders['test']):

        # move to GPU

        if use_cuda:

            data, target = data.cuda(), target.cuda()

        # forward pass: compute predicted outputs by passing inputs to the model

        output = model(data)

        # calculate the loss

        loss = criterion(output, target)

        # update average test loss 

        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))

        # convert output probabilities to predicted class

        pred = output.data.max(1, keepdim=True)[1]

        # compare predictions to 

        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())

        total += data.size(0)

            

    print('Test Loss: {:.6f}\n'.format(test_loss))



    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (100. * correct / total, correct, total))



# call test function    

test(loaders_transfer, model_transfer, criterion, use_cuda)


