import os

import torch

import numpy as np

import torch.nn as nn

import torchvision.transforms as transforms

import torchvision

from torch.utils.data import Dataset, DataLoader

import torchvision.models as models

import time

from sklearn.metrics import accuracy_score

print(torch.__version__)
image_transforms = {

    'train': transforms.Compose([

        transforms.RandomResizedCrop(size=175, scale=(0.8, 1.0)),

        transforms.RandomRotation(degrees=15),

        transforms.RandomHorizontalFlip(),

        transforms.CenterCrop(size=160),

        transforms.ToTensor()

    ]),

    'valid': transforms.Compose([

      transforms.Resize([160, 160]),

    transforms.ToTensor()

    ])

}
# Load the Data

 

# Set train and valid directory paths

train_directory = '/kaggle/input/mask-dataset/dataset/train/'

valid_directory = '/kaggle/input/mask-dataset/dataset/test/'

 

# Batch size

bs = 32

 

# Number of classes

num_classes = 2

 

# Load Data from folders

data = {

    'train': torchvision.datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),

    'valid': torchvision.datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid'])

}

 

# Size of Data, to be used for calculating Average Loss and Accuracy

train_data_size = len(data['train'])

valid_data_size = len(data['valid'])

 

# Create iterators for the Data loaded using DataLoader module

train_data = DataLoader(data['train'], batch_size=bs, shuffle=True)

valid_data = DataLoader(data['valid'], batch_size=bs, shuffle=True)

 

# Print the train, validation data sizes

train_data_size, valid_data_size

mobilenetv2 = models.mobilenet_v2(pretrained=True)

mobilenetv2.classifier=nn.Sequential(

    nn.Linear(1280, 2),

    nn.Softmax(dim=1)

)

mobilenetv2.to('cuda:0')

for param in mobilenetv2.parameters():

    param.requires_grad = True

loss_func = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(mobilenetv2.parameters(),lr=0.01)
epochs =5

for epoch in range(epochs):

    epoch_start = time.time()

    print("Epoch: {}/{}".format(epoch+1, epochs))

     

    # Set to training mode

    mobilenetv2.train()

     

    # Loss and Accuracy within the epoch

    train_loss = 0.0

    train_acc = 0.0

     

    valid_loss = 0.0

    valid_acc = 0.0

    y_pred=[]

    target=[]

    b_count=0

    for i, (inputs, labels) in enumerate(train_data):

        b_count+=1

        inputs=inputs.to('cuda:0')

        labels=labels.to('cuda:0')

        # Clean existing gradients

        optimizer.zero_grad()

         

        # Forward pass - compute outputs on input data using the model

        outputs = mobilenetv2(inputs)

         

        # Compute loss

        loss = loss_func(outputs, labels)

         

        # Backpropagate the gradients

        loss.backward()

         

        # Update the parameters

        optimizer.step()

        

        for i in outputs:

            if i[1]>i[0]:

                y_pred.append(1)

            else:

                y_pred.append(0)

        labels=labels.to("cpu:0")

        target.extend(labels.numpy())

        acc=accuracy_score(target, y_pred)

        print("Batch no - {} , Loss - {} , Accuracy - {}".format(b_count,loss.item(),acc))
from sklearn.metrics import accuracy_score

yact=[]

target=[]

mobilenetv2.to('cuda:0')

with torch.no_grad():

 

    # Set to evaluation mode

    mobilenetv2.eval()

    for x,y in valid_data:

        

        x=x.to('cuda:0')

        

        yhat=mobilenetv2(x)

        for i in yhat:

            if i[1]>i[0]:

                yact.append(1)

            else:

                yact.append(0)



        target.extend(y.numpy())

print("validation accuracy = {}".format(accuracy_score(target, yact)))
#saving the model

mobilenetv2.to('cpu:0')

PATH="./model_params.pth"

torch.save(mobilenetv2.state_dict(), PATH)
the_model = mobilenetv2 = models.mobilenet_v2(pretrained=False)

the_model.classifier=nn.Sequential(

    nn.Linear(1280, 2),

    nn.Softmax()

)

the_model.load_state_dict(torch.load(PATH))