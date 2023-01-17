# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/preprocess"))

# Any results you write to the current directory are saved as output.
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data as utils
from torchvision import transforms, models
height = 224
width = height * 1.5
dataset_path = "../input/preprocess/preprocess"
file_paths = []
for filename in os.listdir(dataset_path):
    if 'left' in filename or 'right' in filename:
        file_paths.append(os.path.join(dataset_path, filename))
import random
from random import shuffle
random.seed(2019)
shuffle(file_paths)
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1
train_files = file_paths[:int(len(file_paths)*train_ratio)]
val_files = file_paths[int(len(file_paths)*train_ratio):int(len(file_paths)*(train_ratio + val_ratio))]
test_files = file_paths[int(len(file_paths)*(train_ratio + val_ratio)):]
class FundusDataset(utils.Dataset):   
    def __init__(self, image_paths, transform=None):
        self.image_paths_list = image_paths 
        # List of image paths      
        self.labels_list = [] 
        # List of labels correlated      
        self.transform = transform 
        # Transformation applying to each data piece            
        # Run through the folder and get the label of each image inside  
        for filename in image_paths:
            self.labels_list.append(0.0 if 'left' in filename else 1.0)
        
    def __getitem__(self, index):      
        '''      Is called when get DataLoader iterated      '''      
        # Get image path with index      
        image_path = self.image_paths_list[index]      
        # Read image with Pillow library      
        image = Image.open(image_path).convert('RGB')      
        # Get label      
        image_label = self.labels_list[index]      
        # Post-transformation apply for image      
        if self.transform != None:          
            image = self.transform(image)            
        return image, image_label      
    def __len__(self):      
        return len(self.image_paths_list)
transform = transforms.Compose([transforms.Resize((int(width), int(height))),                                
                                transforms.ToTensor(),                                
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) 
train_dataset = FundusDataset(train_files, transform)
trainloader = utils.DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataset = FundusDataset(val_files, transform)
valloader = utils.DataLoader(val_dataset, batch_size=16, shuffle=True)
test_dataset = FundusDataset(test_files, transform)
testloader = utils.DataLoader(test_dataset, batch_size=16, shuffle=True)
class FundusNet(nn.Module):
    def __init__(self, is_trained):
        super().__init__()
        self.resnet = models.resnet18(pretrained=is_trained)
        kernel_count = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(nn.Linear(2560, 1),nn.Sigmoid())
    def forward(self, x):
        x = self.resnet(x)
        return x
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:    
    print('CUDA is not available.  Training on CPU ...')
else:    
    print('CUDA is available!  Training on GPU ...')
from torch.optim.lr_scheduler import ReduceLROnPlateau
model = FundusNet(True)
if train_on_gpu:
    model = torch.nn.DataParallel(model).cuda()
state_dict = torch.load('best_model.pth')
model.load_state_dict(state_dict)
loss = nn.BCELoss()
optimizer = optim.Adam (model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'min')
import sys

epochs = 100
train_losses, val_losses = [], []
best_loss = 999999999
for e in range(epochs):
    running_loss = 0
    for step, (images, labels) in enumerate(trainloader):
        
        if train_on_gpu:               
            images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        ps = model(images)            
        loss_val = loss(ps.view(ps.size()[0]), labels.type(torch.cuda.FloatTensor).view(labels.size()[0]))
        loss_val.backward()            
        optimizer.step()
        running_loss += loss_val.item()
        sys.stdout.write(f"\rEpoch {e+1}/{epochs}... Training step {step+1}/{len(trainloader)}... Loss {running_loss/(step+1)}")
    else:
        val_loss = 0            
        accuracy = 0
        with torch.no_grad():                
            for step, (images, labels) in enumerate(valloader):                    
                if train_on_gpu:                       
                    images, labels = images.cuda(), labels.cuda()                    
                ps = model(images)
                val_loss += loss(ps.view(ps.size()[0]), labels.type(torch.cuda.FloatTensor).view(labels.size()[0]))
                pred = 1. * (ps > 0.5)
                equals = pred.type(torch.cuda.FloatTensor) ==  labels.type(torch.cuda.FloatTensor)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
                sys.stdout.write(f"\rEpoch {e+1}/{epochs}... Validating step {step+1}/{len(valloader)}... Loss {val_loss/(step+1)}")
        train_losses.append(running_loss/len(trainloader))
        val_losses.append(val_loss/len(valloader))
        scheduler.step(val_loss/len(valloader))
        print("\nEpoch: {}/{}.. ".format(e+1, epochs),                  
              "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),                  
              "Test Loss: {:.3f}.. ".format(val_loss/len(valloader)),                  
              "Test Accuracy: {:.3f}".format(accuracy/len(valloader)))
        if best_loss > val_loss/len(valloader):
            print("Improve loss of model from {} to {}".format(best_loss, val_loss/len(valloader)))
            best_loss = val_loss/len(valloader)
            torch.save(model.state_dict(), 'best_model.pth')
torch.save(model.state_dict(), 'best_model.pth')
