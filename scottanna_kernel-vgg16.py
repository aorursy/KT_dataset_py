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
# Imports

%matplotlib inline

%config InlineBackend.figure_format = 'retina'



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import torch

from torch import nn, optim

import torch.nn.functional as F

import torchvision

from torchvision import datasets, transforms, models

from torch.autograd import Variable

from torch.utils.data.sampler import SubsetRandomSampler
data_dir = '../input/labelledrice/Labelled/'
# Define transforms for the training and validation sets

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
# Define a custom sampler for the dataset loader avoiding recreating the dataset (just creating a new loader for each different sampling)

train_sampler = SubsetRandomSampler(train_idx)

valid_sampler = SubsetRandomSampler(valid_idx)

test_sampler = SubsetRandomSampler(test_idx)
# Define the dataloaders using the image datasets

trainloader = torch.utils.data.DataLoader(train_data, batch_size = 16, shuffle = True)

validloader = torch.utils.data.DataLoader(valid_data, batch_size = 8, sampler = valid_sampler)

testloader = torch.utils.data.DataLoader(test_data, batch_size = 8, sampler = test_sampler)
print("Classes: ")

class_names = train_data.classes

print(train_data.classes)
classes=train_data.classes
len(class_names)
def imshow(inp, title=None):

    inp = inp.numpy().transpose((1, 2, 0))

    plt.figure(figsize=(10, 10))

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
# Specify model architecture

# Load the pretrained model from pytorch

model_transfer = models.vgg16(pretrained=True)

print(model_transfer)
print(model_transfer.classifier[6].out_features) # 1000 
# Freeze training for all layers

for param in model_transfer.features.parameters():

    param.require_grad = False



# Newly created modules have require_grad=True by default

num_features = model_transfer.classifier[6].in_features

features = list(model_transfer.classifier.children())[:-1] # Remove last layer

features.extend([nn.Linear(num_features, len(class_names))]) # Add our layer with 4 outputs

model_transfer.classifier = nn.Sequential(*features) # Replace the model classifier

print(model_transfer)
use_gpu = torch.cuda.is_available()

if use_gpu:

    print("Using CUDA")
from torch.optim import lr_scheduler
if use_gpu:

    model_transfer.cuda() #.cuda() will move everything to the GPU side

    

criterion = nn.CrossEntropyLoss()

#optimizer_ft = torch.optim.Adam(model_transfer.parameters(), lr = 0.001)

optimizer_ft = optim.SGD(model_transfer.parameters(), lr=0.0001, momentum=0.9)

#exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
import time

import copy
# Train the model

def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):

    '''returns trained model'''

    # Initialize tracker for minimum validation loss

    valid_loss_min = np.inf

  

    for epoch in range(1, n_epochs+1):

        since = time.time()

        # Initialize variables to monitor training and validation loss

        train_loss = 0.0

        valid_loss = 0.0

    

        # Model training

        model.train()

        for batch_idx, (data,target) in enumerate(trainloader):

            # Move to GPU

            if use_cuda:

                data,target = data.cuda(), target.cuda()

      

            # Clear the gradient of all optimized variables

            optimizer.zero_grad()

            # Forward pass: compute predicted outputs by passing inputs to the model

            output = model(data)

            # Calculate the batch loss

            loss = criterion(output, target)

            # Backward pass: compute gradient of the loss with respect to model parameters

            loss.backward()

            # Perform a single optimization step (parameter update)

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

        

        elapsed_time = time.time() - since

        print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))   

    

        

        # Save the model if validation loss has decreased

        if valid_loss <= valid_loss_min:

            print('Validation loss decreased ({:.5f} --> {:.5f}). Saving model ...'.format(

                  valid_loss_min,

                  valid_loss))

            torch.save(model.state_dict(), 'model_transfer.pt')

            valid_loss_min = valid_loss

  

    # Return trained model

    return model
# Define loaders transfer

loaders_transfer = {'train': trainloader,

                    'valid': validloader,

                    'test': testloader}

# Train the model

model_transfer = train(15, loaders_transfer, model_transfer, optimizer_ft, criterion, use_gpu, 'model_transfer.pt')
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



    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (

        100. * correct / total, correct, total))



# call test function    

test(loaders_transfer, model_transfer, criterion, use_gpu)