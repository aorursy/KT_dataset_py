import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import os

import glob

import seaborn as sns

import time

import cv2

import copy

import imageio

import h5py

from sklearn.metrics import recall_score

from pathlib import Path



# Importing Pytorch libraries for deep learning

import torch

import torch.nn as nn

import torch.nn.functional as F

from torchvision import models,transforms,datasets
# General path

data_dir = Path('../input/chest-xray-pneumonia/chest_xray/chest_xray')



# Path to train directory (Fancy pathlib...no more os.path!!)

train_dir = data_dir / 'train'



# Path to validation directory

val_dir = data_dir / 'val'



# Path to test directory

test_dir = data_dir / 'test'
# Function to organize all data into dataframes



def data_into_dataframes(path):



    # Get the path to the normal and pneumonia sub-directories

    normal_cases_dir    =  path / 'NORMAL'

    pneumonia_cases_dir =  path / 'PNEUMONIA'



    # Get the list of all the images

    normal_cases = normal_cases_dir.glob('*.jpeg')

    pneumonia_cases = pneumonia_cases_dir.glob('*.jpeg')



    # An empty list. We will insert the data into this list in (img_path, label) format

    data = []



    # Go through all the normal cases. The label for these cases will be 0

    for img in normal_cases:

        data.append((img,0))



    # Go through all the pneumonia cases. The label for these cases will be 1

    for img in pneumonia_cases:

        data.append((img, 1))



    # Get a pandas dataframe from the data we have in our list 

    data = pd.DataFrame(data, columns=['image', 'label'],index=None)





    # Shuffle the data 

    data = data.sample(frac=1.).reset_index(drop=True)

    

    return data





train_data      = data_into_dataframes(train_dir)

validation_data = data_into_dataframes(val_dir)

test_data       = data_into_dataframes(test_dir) 
print("The training dataset shape is: {0}".format(train_data.shape))

print("The validation dataset shape is: {0}".format(validation_data.shape))

print("The testing dataset shape is: {0}".format(test_data.shape))
# Merging the three datasets into one

total_data = pd.DataFrame()

total_data["Train"]        = train_data.label

total_data["Validation"]   = validation_data.label

total_data["Test"]         = test_data.label





# Plotting the countplot

fig, ax =plt.subplots(1,3,figsize=(15,5))

fig.suptitle('Number of labels in each dataset', fontsize=16)

sns.countplot(total_data.Train, ax=ax[0])

sns.countplot(total_data.Validation, ax=ax[1])

sns.countplot(total_data.Test, ax=ax[2])

fig.show()
# Data augmentation and normalization for training but just resize and normalization for validation



data_transforms = {

    'train': transforms.Compose([   

                                    transforms.RandomResizedCrop(224),

                                    transforms.RandomHorizontalFlip(),

                                    transforms.ToTensor(),

                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],

                                                         std=[0.229, 0.224, 0.225])

    ]),

    

    'val': transforms.Compose([

                                    transforms.Resize(256),

                                    transforms.CenterCrop(224),

                                    transforms.ToTensor(),

                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],

                                                         std=[0.229, 0.224, 0.225])

    ]),

    

    "test": transforms.Compose([     

                                    transforms.Resize(256),

                                    transforms.CenterCrop(224),

                                    transforms.ToTensor(),

                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],

                                                         std=[0.229, 0.224, 0.225])

        

    ])

}





# Create training and validation datasets



training_dataset   = datasets.ImageFolder(train_dir, transform=data_transforms["train"])

validation_dataset = datasets.ImageFolder(val_dir, transform=data_transforms["val"])

testing_dataset    = datasets.ImageFolder(test_dir, transform=data_transforms["test"])



# Create training and validation dataloaders



training_loader   = torch.utils.data.DataLoader(training_dataset, batch_size=20, shuffle=True)

validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size =4, shuffle=False)

testing_loader    = torch.utils.data.DataLoader(testing_dataset, shuffle=False)



data_loaders={"train":training_loader,"val":validation_loader,"test":testing_loader}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device
model = models.vgg19_bn(pretrained=True).to(device)
model
# Freezing the Feature extracting part and changing the number of outputs ("no-pneumonia" and "pneumonia")



for param in model.features.parameters():

    param.requires_grad = False



n_inputs = model.classifier[6].in_features

last_layer = nn.Linear(n_inputs, 2)

model.classifier[6] = last_layer

model.to(device)

print(model.classifier)
# Setting the criteria and the optimizer 



criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
epochs = 10

running_loss_history = []

running_corrects_history = []

val_running_loss_history = []

val_running_corrects_history = []

 

for e in range(epochs):

    running_loss = 0.0

    running_corrects = 0.0

    val_running_loss = 0.0

    val_running_corrects = 0.0

        

        # TRAINING

        

    for inputs, labels in data_loaders["train"]:

        inputs = inputs.to(device)

        labels = labels.to(device)

        outputs = model(inputs)

        loss = criterion(outputs, labels)



        optimizer.zero_grad()

        loss.backward()

        optimizer.step()



        _, preds = torch.max(outputs, 1)

        running_loss += loss.item()

        running_corrects += torch.sum(preds == labels.data)



        # VALIDATION

        

    else:

        with torch.no_grad():

            for val_inputs, val_labels in data_loaders["val"]:

                val_inputs = val_inputs.to(device)

                val_labels = val_labels.to(device)

                val_outputs = model(val_inputs)

                val_loss = criterion(val_outputs, val_labels)



                _, val_preds = torch.max(val_outputs, 1)

                val_running_loss += val_loss.item()

                val_running_corrects += torch.sum(val_preds == val_labels.data)



    epoch_loss   = running_loss/len(training_loader.dataset)

    epoch_acc    = running_corrects.float()/ len(training_loader.dataset)

    running_loss_history.append(epoch_loss)

    running_corrects_history.append(epoch_acc)

    

    val_epoch_loss = val_running_loss/len(validation_loader.dataset)

    val_epoch_acc = val_running_corrects.float()/ len(validation_loader.dataset)

    val_running_loss_history.append(val_epoch_loss)

    val_running_corrects_history.append(val_epoch_acc)

    print('epoch :', (e+1))

    print('training loss: {:.4f}, acc {:.4f} '.format(epoch_loss, epoch_acc.item()))

    print('validation loss: {:.4f}, validation acc {:.4f} '.format(val_epoch_loss, val_epoch_acc.item()))



test_running_corrects=0.0

for test_inputs, test_labels in data_loaders["test"]:

        test_inputs = test_inputs.to(device)

        test_labels = test_labels.to(device)

        test_outputs = model(test_inputs)

        _, test_preds = torch.max(test_outputs, 1)

        

        test_running_corrects += torch.sum(test_preds == test_labels.data)



acc = test_running_corrects.float()/ len(testing_loader.dataset)

acc
test_running_corrects=0.0

for test_inputs, test_labels in data_loaders["test"]:

        test_inputs = test_inputs.to(device)

        test_labels = test_labels.to(device)

        test_outputs = model(test_inputs)

        _, test_preds = torch.max(test_outputs, 1)

        

        test_running_corrects += torch.sum(test_preds == test_labels.data)



acc = test_running_corrects.float()/ len(testing_loader.dataset)

acc