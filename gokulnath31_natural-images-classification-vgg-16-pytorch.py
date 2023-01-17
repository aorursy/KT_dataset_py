# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import torch
import torch.nn as nn
import torch.functional as F
from torchvision import *
from torch.utils.data import DataLoader, sampler, random_split
from torch.optim import *
import pathlib

import time

import copy

import numpy as np

from PIL import Image
from PIL import ImageFile
import albumentations
def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
device = get_default_device()
device
print(os.listdir('../input/natural-images/natural_images'))
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=256, scale=(0.5, 1)),
    transforms.RandomRotation(degrees=25),
    transforms.RandomHorizontalFlip(0.7),
    transforms.ColorJitter(),
    transforms.RandomHorizontalFlip(),
    transforms.CenterCrop(size=224),  # Image net standards
    transforms.ToTensor(),
    #transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])
validation_transform = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

all_data = datasets.ImageFolder(root='../input/natural-images/natural_images/')
train_data_len = int(len(all_data)*0.8)
valid_data_len = int((len(all_data) - train_data_len))
train_data, val_data = random_split(all_data, [train_data_len, valid_data_len])
train_data.dataset.transform = train_transform
val_data.dataset.transform = validation_transform

print(len(train_data), len(val_data))

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=True)
root = pathlib.Path('../input/natural-images/data/natural_images/')
classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])
print(classes)
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        
        self.backbone =  models.vgg16(pretrained=True)
        num_features = self.backbone.classifier[6].in_features
        self.backbone.classifier[6] = nn.Sequential(
                                            nn.Linear(num_features, 256),
                                            nn.ReLU(),
                                            nn.Dropout(0.4),
                                            nn.Linear(256, len(classes)),
                                            nn.LogSoftmax(dim=1))
        
    def forward(self,image):
        x = self.backbone(image)
        return x
    
    def freeze(self):
        # To freeze the residual layers
        for param in self.backbone.parameters():
            param.require_grad = False
        for param in self.backbone.classifier[6].parameters():
            param.require_grad = True
    
    def unfreeze(self):
        # Unfreeze all layers
        for param in self.backbone.parameters():
            param.require_grad = True
model = Model()
model = model.to(device)
criterion = nn.NLLLoss()
model.freeze()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
model.cuda()
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
def prediciton(net, data_loader):
    test_pred = torch.LongTensor()
    for i, data in enumerate(data_loader):
        if torch.cuda.is_available():
          pass

        output = net(data)
        pred = output.cpu().data.max(1, keepdim=True)[1]
        test_pred = torch.cat((test_pred, pred), dim=0)
    
    return test_pred
dataloaders = {'train':train_loader,"val":val_loader}
dataset_sizes = {'train':len(train_data),'val':len(val_data)}
print(dataset_sizes)
model_ft = train_model(model, criterion, optimizer, exp_lr_scheduler,num_epochs=15)
model.unfreeze()
model_ft = train_model(model, criterion, optimizer, exp_lr_scheduler,num_epochs=20)
torch.save(model,'vgg16-20.pth')
torch.save(model.state_dict(),'vgg16-20.pt')