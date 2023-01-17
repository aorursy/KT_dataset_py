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
#we will start off by importing the libraries as usual



from __future__ import print_function



#pytorch magic is about to happen

import torch

import torchvision

from torch import utils

import torch.optim as optim

import torch.nn as nn

import torch.nn.functional as F

from torchvision import datasets, transforms, models



#some more to manipulate and vizualize our data

import time

import numpy as np

from PIL import Image

import matplotlib.pyplot as plt

import os

import warnings
# Applying Transforms to the Data

image_transforms = {

    'train': transforms.Compose([

        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),

        transforms.RandomRotation(degrees=15),

        transforms.RandomHorizontalFlip(),

        transforms.CenterCrop(size=224),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406],

                             [0.229, 0.224, 0.225])

    ]),

    'valid': transforms.Compose([

        transforms.Resize(size=256),

        transforms.CenterCrop(size=224),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406],

                             [0.229, 0.224, 0.225])

    ])

}



#we are gonna use pretrained models so we would use the mean and std that pytorch specifies.
# Load the Data

 

# Set train and valid directory paths

train_directory = 'train'

valid_directory = 'test'

 

#hyperparams

batch_size = 16

num_labels = 102

learning_rate= 1e-4

image_size=  224 * 224

device= 'cpu'



# Load Data from folders

data = {

    'train': datasets.ImageFolder(root='../input/flower_data/flower_data/train', transform=image_transforms['train']),

    'valid': datasets.ImageFolder(root='../input/flower_data/flower_data/valid', transform=image_transforms['valid']),

}

 

# Size of Data, to be used for calculating Average Loss and Accuracy

train_data_size = len(data['train'])

valid_data_size = len(data['valid'])

 

# Create iterators for the Data loaded using DataLoader module

train_data = utils.data.DataLoader(data['train'], batch_size=batch_size, shuffle=True)

valid_data = utils.data.DataLoader(data['valid'], batch_size=batch_size, shuffle=True)

 

# Print the train, validation and test set data sizes

print("We have a total of {} training samples, and {} testing/validation samples".format(train_data_size, valid_data_size))
# Load pretrained ResNet50 Model

model = models.densenet161(pretrained= True)
#check the architecture of densenet

model
# Freeze model parameters, this means we would turn of the training for their parameters.

for param in model.parameters():

    param.requires_grad = False
input_data= model.classifier.in_features



classifier= nn.Sequential(nn.Linear(input_data, 1024),

                          nn.ReLU(),

                          nn.Linear(1024, 512),

                          nn.ReLU(),

                          nn.Dropout(0.20),

                          nn.Linear(512, num_labels),

                          nn.LogSoftmax(dim= 1))



model.classifier= classifier
#to actually use our GPU for training

train_on_gpu = torch.cuda.is_available()



if not train_on_gpu:

    print('CUDA is not available.  Training on CPU ...')

else:

    device= 'cuda'

    print('CUDA is available!  Training on GPU ...')
model.to(device)
# Define Optimizer and Loss Function

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.classifier.parameters())
epochs = 10



for epoch in range(epochs):

    epoch_start = time.time()

    print("Epoch: {}/{}".format(epoch+1, epochs))

     

    # Set to training mode

    model.train()

     

    # Loss and Accuracy within the epoch

    train_loss = 0.0

    train_acc = 0.0

     

    valid_loss = 0.0

    valid_acc = 0.0

 

    for i, (inputs, labels) in enumerate(train_data):

 

        inputs = inputs.to(device)

        labels = labels.to(device)

         

        # Clean existing gradients

        optimizer.zero_grad()

         

        # Forward pass - compute outputs on input data using the model

        outputs = model(inputs)

         

        # Compute loss

        loss = criterion(outputs, labels)

         

        # Backpropagate the gradients

        loss.backward()

         

        # Update the parameters

        optimizer.step()

         

        # Compute the total loss for the batch and add it to train_loss

        train_loss += loss.item() * inputs.size(0)

         

        # Compute the accuracy

        ret, predictions = torch.max(outputs.data, 1)

        correct_counts = predictions.eq(labels.data.view_as(predictions))

         

        # Convert correct_counts to float and then compute the mean

        acc = torch.mean(correct_counts.type(torch.FloatTensor))

         

        # Compute total accuracy in the whole batch and add to train_acc

        train_acc += acc.item() * inputs.size(0)

         

        print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))
# Validation - No gradient tracking needed

with torch.no_grad():

 

    # Set to evaluation mode

    model.eval()

 

    # Validation loop

    for j, (inputs, labels) in enumerate(valid_data):

        inputs = inputs.to(device)

        labels = labels.to(device)

 

        # Forward pass - compute outputs on input data using the model

        outputs = model(inputs)

 

        # Compute loss

        loss = criterion(outputs, labels)

 

        # Compute the total loss for the batch and add it to valid_loss

        valid_loss += loss.item() * inputs.size(0)

 

        # Calculate validation accuracy

        ret, predictions = torch.max(outputs.data, 1)

        correct_counts = predictions.eq(labels.data.view_as(predictions))

 

        # Convert correct_counts to float and then compute the mean

        acc = torch.mean(correct_counts.type(torch.FloatTensor))

 

        # Compute total accuracy in the whole batch and add to valid_acc

        valid_acc += acc.item() * inputs.size(0)

 

        print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))

     

# Find average training loss and training accuracy

avg_train_loss = train_loss/train_data_size 

avg_train_acc = train_acc/float(train_data_size)

 

# Find average training loss and training accuracy

avg_valid_loss = valid_loss/valid_data_size 

avg_valid_acc = valid_acc/float(valid_data_size)

 

history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

         

epoch_end = time.time()

 

print("Epoch : {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation : Loss : {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(epoch, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start))
from PIL import Image

def predict(model, test_image_name):

     

    transform = image_transforms['valid']

 

    test_image = Image.open(test_image_name)

    plt.imshow(test_image)

     

    test_image_tensor = transform(test_image)

 

    if torch.cuda.is_available():

        test_image_tensor = test_image_tensor.view(1, 3, 224, 224).cuda()

    else:

        test_image_tensor = test_image_tensor.view(1, 3, 224, 224)

     

    with torch.no_grad():

        model.eval()

        # Model outputs log probabilities

        out = model(test_image_tensor)

        ps = torch.exp(out)

        topk, topclass = ps.topk(1, dim=1)

        print("Output class :  ", topclass.cpu().numpy()[0][0] + 1)
predict(model, '../input/flower_data/flower_data/valid/1/image_06763.jpg')