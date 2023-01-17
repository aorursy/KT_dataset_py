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
import torch

import numpy as np

from torch import nn, optim

import torch.nn.functional as F

from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt

from collections import OrderedDict

from PIL import Image

%matplotlib inline
data_dir = '../input/flower_data/flower_data'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
# TODO: Define your transforms for the training and validation sets
#data_transforms = 

train_transform=transforms.Compose([transforms.RandomRotation(30),transforms.RandomResizedCrop(224),
                                   transforms.RandomHorizontalFlip(),transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

valid_transform=transforms.Compose([transforms.Resize(255),transforms.CenterCrop(224),transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])


# TODO: Load the datasets with ImageFolder
#image_datasets = 
train_data=datasets.ImageFolder(train_dir,transform=train_transform)

valid_data=datasets.ImageFolder(valid_dir,transform=valid_transform)


# TODO: Using the image datasets and the trainforms, define the dataloaders
#dataloaders = 

trainloader=torch.utils.data.DataLoader(train_data,batch_size=32)

validloader=torch.utils.data.DataLoader(valid_data,batch_size=32)

model = nn.Sequential(nn.Conv2d(3,16,3,padding=1),
                      nn.ReLU(),
                      nn.Conv2d(16,16,3,padding=1),
                      nn.ReLU(),
                      nn.MaxPool2d(2, 2),
                      nn.Dropout(0.25),
                      nn.Conv2d(16,32,3,padding=1),
                      nn.ReLU(),
                      nn.Conv2d(32,32,3,padding=1),
                      nn.ReLU(),
                      nn.MaxPool2d(2, 2),
                      nn.Dropout(0.25),
                      nn.Linear(32*56*56, 512),
                      nn.ReLU(),
                      nn.Linear(512, 1024),
                      nn.ReLU(),
                      nn.Linear(1024, 102),
                      nn.LogSoftmax(dim=1))

criterion=nn.NLLLoss()

optimizer=optim.Adam(model.parameters(),lr=0.01)

torch.save(model.state_dict(),'model.pt')
# number of epochs to train the model
n_epochs = 0

valid_loss_min = np.Inf # track change in validation loss

tloss=[]

vloss=[]


for epoch in range(1, n_epochs+1):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    
    ###################
    # train the model #
    ###################
    model.train()
    for data, target in trainloader:
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()*data.size(0)
        
    ######################    
    # validate the model #
    ######################
    model.eval()
    
    accuracy=0
    
    with torch.no_grad():
      
      for data, target in validloader:
          # forward pass: compute predicted outputs by passing inputs to the model
          output = model(data)
          # calculate the batch loss
          loss = criterion(output, target)
          # update average validation loss 
          valid_loss += loss.item()*data.size(0)
          
          # Accuracy Calculation
            
          ps = torch.exp(output)
          top_p, top_class = ps.topk(1, dim=1)
          equals = top_class == target.view(*top_class.shape)
          accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    
    # calculate average losses
    train_loss = train_loss/len(trainloader.dataset)
    valid_loss = valid_loss/len(validloader.dataset)
    
    tloss.append(train_loss)
    
    vloss.append(valid_loss)
        
    # print training/validation statistics 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        #model_save_name = 'model'+str(epoch)+'.pt'
        #path = F"../input/{model_save_name}" 
        #torch.save(model.state_dict(),path)
        valid_loss_min = valid_loss
        
    print("Validation Accuracy: {:.3f}".format(100*accuracy/len(validloader)))
