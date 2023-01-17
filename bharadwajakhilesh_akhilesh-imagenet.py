# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import torch
import torchvision
from torch.utils.data import Dataset,DataLoader
import numpy as np
import math
from torchvision import datasets, models, transforms
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224,224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean = [0.5,0.5,0.5],std=[0.25,0.25,0.25])
])
dataset = datasets.ImageFolder('../input/cnn-img-data/tiny-imagenet-200/train/', transform=transform)
len(dataset),len(dataset)/500
dataloader = DataLoader(dataset,batch_size=64, shuffle=True, num_workers=10)
import matplotlib.pyplot as plt
dataiter = iter(dataloader)
for _ in range(3):
    images,label = next(dataiter)
    arr = np.squeeze(images[0].numpy()[0])
    plt.imshow(arr)
    plt.show()
    print(label[0])
from torchvision import models
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device
model = models.resnet18(pretrained = True).to(device)
num_ftrs = model.fc.in_features
print(num_ftrs)
model.fc = nn.Sequential(
    nn.Linear(512,1024),
    nn.ReLU(inplace=True),
    nn.Linear(1024,512),
    nn.ReLU(inplace=True),
    nn.Linear(512,200)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = 0.001)
from torch.optim import lr_scheduler

step_lr_scheduler = lr_scheduler.StepLR(optimizer,step_size = 4,gamma=0.1)
import time

def train_model(model,criterion,optimizer,step_lr_scheduler,num_epochs = 3):
    since = time.time()
    for epoch in range(num_epochs):
        since1 = time.time()
        print('Epoch{}/{}'.format(epoch+1,num_epochs))
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for i,(inputs,labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _,pred = torch.max(outputs,1)
            running_loss += loss.item()*inputs.size(0)
            running_corrects += torch.sum(pred==labels.data)
            
            
            if(i%100==0):
                print(f'index:{i} epoch:{epoch+1}')
        l1=[]
        ac1=[]
        epoch_loss = running_loss/len(dataset)
        epoch_acc = running_corrects.double()/len(dataset)
        l1.append(epoch_loss)
        ac1.append(epoch_acc)
        print('{} loss:{:.4f},acc:{:.4f}'.format('train',epoch_loss,epoch_acc))
        time_elapsed1 = time.time() - since1
        print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed1 // 60, time_elapsed1 % 60))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    return model,l1,ac1
model_trained,loss_list,acc_list = train_model(model,criterion,optimizer,step_lr_scheduler,num_epochs = 20)
validationDataset = datasets.ImageFolder('../input/cnn-img-data/tiny-imagenet-200/val/', transform=transform)
valDataloader = torch.utils.data.DataLoader(validationDataset, batch_size=32, shuffle=True, num_workers=10)
len(validationDataset)
def eval_val(model, criterion, optimizer, num_epoch=3):
    for epoch in range(num_epoch):
        print(f'epoch {epoch}/ {num_epoch}', '*' * 10, sep = '\n')
        model.eval()
        cur_loss, cur_correct = 0.0, 0.0
        for index, (inputs, labels) in enumerate(valDataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            if (index % 30 == 0):
                print(f'index: {index}')
            
            _, pred = torch.max(outputs, axis=1)
            cur_loss += loss.item() * inputs.size(0)
            cur_correct += torch.sum(pred == labels.data)
            
        epoch_loss = cur_loss / len(valDataloader)
        epoch_accu = cur_correct.double() / len(valDataloader)
            
        print(f'Loss: {epoch_loss:.3f}, Acc: {epoch_accu:.3f}')
        
    return model
model_tested = eval_val(model_trained, criterion, optimizer, num_epoch=1)
