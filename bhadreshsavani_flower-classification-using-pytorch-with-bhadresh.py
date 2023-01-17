# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import time

import cv2

import glob

import torch 

from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt

print(os.listdir("../input"))
labels= pd.read_json("../input/cat_to_name.json", typ='series')

print("No of Flower Labels : "+str(len(labels)))

all_labels=[]

for i in range(len(labels)):

    all_labels.append(labels[i+1])

print(all_labels)
labels.head(10)
train_transforms = transforms.Compose([transforms.RandomRotation(30),

                                       transforms.RandomResizedCrop(224),

                                       transforms.RandomHorizontalFlip(),

                                       transforms.ToTensor(),

                                       transforms.Normalize([0.485, 0.456, 0.406],

                                                            [0.229, 0.224, 0.225])])



test_transforms = transforms.Compose([transforms.Resize(255),

                                      transforms.CenterCrop(224),

                                      transforms.ToTensor(),

                                      transforms.Normalize([0.485, 0.456, 0.406],

                                                           [0.229, 0.224, 0.225])])



data_dir='../input/flower_data/flower_data'



# Pass transforms in here, then run the next cell to see how the transforms look

train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)

valid_data = datasets.ImageFolder(data_dir + '/valid', transform=test_transforms)

test_data = datasets.ImageFolder('../input/test set/', transform=test_transforms)



testloader = torch.utils.data.DataLoader(test_data)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
def get_unique_data():

    unique_images=[]

    unique_labels=[]

    for image,label in train_data:

        if label not in unique_labels:

            unique_images.append(image)

            unique_labels.append(label)

    return unique_images,unique_labels
import torchvision

unique_images,unique_labels=get_unique_data()

grid=torchvision.utils.make_grid(unique_images,nrow=20, padding=2)

plt.figure(figsize=(15,15))

plt.imshow(np.transpose(grid,(1,2,0)))

print('labels:',unique_labels)
image_path=data_dir+"/train/1/image_06745.jpg"

image=cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB)

print(image.shape)
plt.imshow(image)
#Show number of images per label

fig, ax = plt.subplots(figsize=(20, 30))



unique, numOfEach = np.unique(train_data.targets, return_counts=True)

ax.barh(labels, numOfEach, align='center', color='blue')

ax.set_yticklabels(all_labels)

ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('Number of each')

ax.set_title('How many of each signs do we have?')



plt.show()
model = models.densenet121(pretrained=True)

model
from torch import nn

from torch import optim

import torch.nn.functional as F



# Use GPU if it's available

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

# Freeze parameters so we don't backprop through them

for param in model.parameters():

    param.requires_grad = False

    

model.classifier = nn.Sequential(nn.Linear(1024, 256),

                                 nn.Linear(256, 102))



criterion = nn.CrossEntropyLoss() # defining loss function





optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)



model.to(device);
def train_model(model, criterion, optimizer, num_epochs=10):

    since = time.time()

    i=0

    best_model_wts = model.state_dict()

    best_acc = 0.0



    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch+1, num_epochs))

        print('-' * 10)



        # Each epoch has a training and validation phase

        for phase in ['train', 'valid']:

            if phase == 'train':

                #scheduler.step()

                model.train(True)

                dataloader=trainloader# Set model to training mode

            else:

                model.train(False)

                dataloader=validloader# Set model to evaluate mode



            running_loss = 0.0

            running_corrects = 0



            # Iterate over data.

            for data in dataloader:

                # get the inputs

                inputs, labels = data



                inputs=inputs.to(device)

                labels=labels.to(device)

        

                # zero the parameter gradients

                optimizer.zero_grad()



                # forward

                outputs = model(inputs)

                

                _, preds = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels)



                # backward + optimize only if in training phase

                if phase == 'train':

                    loss.backward()

                    optimizer.step()



                # statistics

                running_loss  += loss.item()

                running_corrects += torch.sum(preds == labels.data).item()

                

            if phase=="train":

                epoch_loss = running_loss / len(trainloader)

                epoch_acc = running_corrects / len(train_data)

    

                

            else:

                epoch_loss = running_loss / len(validloader)

                epoch_acc = running_corrects / len(valid_data)



            print('{} Loss: {:.4f} Acc: {:.4f}'.format(

                phase, epoch_loss, epoch_acc))

            

#             # deep copy the model

            if phase == 'valid' and epoch_acc > best_acc:

                best_acc = epoch_acc

#                 best_model_wts = model.state_dict()

#                 # saving a checkpoint to use for next time to save time used in training from scratch

#                 state = {'model':model.state_dict(),'optim':optimizer.state_dict()}

#                 torch.save(state,'drive/flowers classification/point_resnet_best.pth')



    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(

        time_elapsed // 60, time_elapsed % 60))

    print('Best val Acc: {:4f}'.format(best_acc))



    # load best model weights

    model.load_state_dict(best_model_wts)

    return model
trained_model= train_model(model, criterion, optimizer,num_epochs=20)
torch.save(model, "../densenet_model.pth")
model = torch.load("../densenet_model.pth")
def get_predictions():

    pred_list=[]

    for image, label in testloader:

        # zero the parameter gradients

        optimizer.zero_grad()

        image=image.to(device)

            # forward

        outputs = model(image)

        _, preds = torch.max(outputs.data, 1)

        pred_list.append(preds.item())

    return pred_list
import os

path='../input/test set/test set'

image_list=os.listdir("../input/test set/test set")
pred_list=get_predictions()

for i in range(len(pred_list)):

    print(pred_list[i], all_labels[pred_list[i]-1], image_list[i])