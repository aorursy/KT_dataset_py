# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import gc

import os

import sys

import time

import random

import logging

import datetime as dt

import cv2

import pickle



import numpy as np

import pandas as pd

import datetime

import torch

import torch.nn as nn

import torch.optim as optim

import torch.utils.data as data

from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as fun

import torchvision as vision

from torch.autograd import Variable

from torch.optim.lr_scheduler import CosineAnnealingLR

from torchvision import datasets, models, transforms

from pathlib import Path

from PIL import Image

from contextlib import contextmanager



from joblib import Parallel, delayed

from tqdm import tqdm

from fastprogress import master_bar, progress_bar



from sklearn.metrics import fbeta_score
data_dir = '../input/traffic-signs-preprocessed/'

num_epochs = 32

batch_size = 50

learning_rate = 0.0001
train = pickle.load(open(data_dir+'train.pickle','rb'))

test = pickle.load(open(data_dir+'test.pickle','rb'))

valid = pickle.load(open(data_dir+'valid.pickle','rb'))

labels = pickle.load(open(data_dir+'labels.pickle','rb'))

#print(train)
train_labs = train['labels']

valid_labs = valid['labels']

test_labs = test['labels']



train_imgs = train['features']

valid_imgs = valid['features']

test_imgs = test['features']
print(len(train_labs),len(valid_labs),len(test_labs))
means = np.mean(train_imgs, axis=(0, 1, 2)) / 255.

stds = np.std(train_imgs, axis=(0, 1, 2)) / 255.

print(means)

print(stds)
class dataprocessor(Dataset):

    def __init__(self,image,labels,transform):

        self.image = image

        self.labels = labels

        self.transform = transform

    def __len__(self):

        return self.labels.shape[0]

    def __getitem__(self,idx):

        image = self.image[idx] 

        image= Image.fromarray(image)

        image = self.transform(image)

        image= image.tolist()

        label = np.zeros((43),dtype=int)

        label= label.tolist()

        label_idx = self.labels[idx]

        label[label_idx] = 1

        label= torch.FloatTensor(label)

        image=torch.FloatTensor(image)

        

        return [image,label]
data_transforms = {

    'train': vision.transforms.Compose([

    vision.transforms.Resize((64,64)),

    transforms.RandomResizedCrop(250),

    transforms.RandomHorizontalFlip(),

    vision.transforms.ToTensor(),

    vision.transforms.Normalize(mean=means, std=stds)

    ]),

    'val': vision.transforms.Compose([

        vision.transforms.Resize(256),

        vision.transforms.CenterCrop(224),

        vision.transforms.ToTensor(),

        vision.transforms.Normalize(mean=means, std=stds)

    ]),

}



data_transforms["test"] = data_transforms["train"]
train_dataset = dataprocessor(train_imgs,train_labs,data_transforms["train"])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True , num_workers=2, pin_memory=True)



test_dataset = dataprocessor(test_imgs,test_labs,data_transforms["test"])

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True , num_workers=2, pin_memory=True)



valid_dataset = dataprocessor(valid_imgs,valid_labs,data_transforms["test"])

valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False , num_workers=2, pin_memory=True)
def training(epoch,loss_func,optimizer,model,dataloader):

    training_loss = 0

    training_acc = 0

    for step,( x,y ) in enumerate(dataloader):

        data = Variable(x).cuda()   # batch x

        target = Variable(y).cuda()   #batch x

        model.cuda()

        output = model(data)

        target = Variable(y).cuda()

        loss = loss_func(output, target.float())   # cross entropy loss

        optimizer.zero_grad()           # clear gradients for this training step

        loss.backward()                 # backpropagation, compute gradients

        optimizer.step()                # apply gradients

        training_loss += loss.item()

        training_acc += (torch.max(output, 1)[1] == torch.max(target, 1)[1]).type(torch.FloatTensor).mean().item()

        if step==0:

            start = time.time()

            ti = 0

        elif step==100:

            ti = time.time()-start #total time = ti*(length/100)

            #print(ti)

            ti = ti*(len(dataloader)/100)

        if step % 100 == 0:

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}\tTime Remain : {} '.

                     format(epoch+1, 

                            step * len(data), 

                            len(dataloader.dataset),

                            100.*step/len(dataloader), 

                            loss.data.item(),

                            datetime.timedelta(seconds=(ti*((int(len(dataloader)-step)/len(dataloader)))))))

        data.detach()   # batch x

        target.detach()   # batch y

    epoch_loss = training_loss / len(dataloader)

    epoch_acc = training_acc / len(dataloader) #caculating the whole epoch accuracy

    print("Train MSELoss/Accuracy: \t{:.4f}\t{:.4f}".format(epoch_loss,epoch_acc))

    print("Epoch: {} finish!".format(epoch+1))

    return model

    
densenet201 = models.densenet201(pretrained='imagenet')#

resnet50 = models.resnet50(pretrained='imagenet')#

#in_features = 1920,

densenet201.classifier = nn.Linear(in_features=1920,out_features=43, bias = True)#in_features=2048

resnet50.fc = nn.Linear(in_features=2048, out_features=43, bias=True)
def training_process(net_name,model,parameters):

    print("Taining Model :{}.\n Numbers of epoch: {}".format(net_name,num_epochs))

    for epoch  in range(num_epochs) :

        loss_func=torch.nn.MSELoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.00002/(2**epoch))

        model=training(epoch,loss_func,optimizer,model,train_loader)

        valid_loss = 0

        valid_acc = 0

        with torch.no_grad():

            for step,( x,y ) in enumerate(valid_loader):

                data = Variable(x).cuda()   # batch x

                target = Variable(y).cuda()   #batch x

                model.cuda()

                output = model(data)

                target = Variable(y).cuda()

                loss = loss_func(output, target.float())   # cross entropy loss

    

                valid_loss += loss.item()

                valid_acc += (torch.max(output, 1)[1] == torch.max(target, 1)[1]).type(torch.FloatTensor).mean().item()

        epoch_valid_loss = valid_loss / len(valid_loader)

        epoch_valid_acc = valid_acc / len(valid_loader)

        print("{}\tValid MSELoss/Accuracy: \t{:.4f}\t{:.4f}".format(net_name,epoch_valid_loss,epoch_valid_acc))

    return model
trained_densenet201 = training_process('Densenet201',densenet201,densenet201.parameters())

torch.save(trained_densenet201, 'trained_densenet201.pkl')

torch.cuda.empty_cache()

!nvidia-smi
trained_resnet50 = training_process('Resnet50',resnet50,resnet50.parameters())

torch.save(trained_resnet50, 'trained_resnet50.pkl')

torch.cuda.empty_cache()

!nvidia-smi
def test(net_name,errors,model):

    test_acc = 0

    test_loss = 0

    loss_func=torch.nn.MSELoss()

    for step, (x,y) in enumerate(test_loader):

        data = Variable(x).cuda()   # batch x

        target = Variable(y).cuda()   #batch y

        

        model.cuda()

        output = model(data)

        target = Variable(y).cuda()

        loss = loss_func(output, target.float())

        test_loss += loss.item()

        test_acc += (torch.max(output, 1)[1] == torch.max(target, 1)[1]).type(torch.FloatTensor).mean().item()

        

        true_labels = torch.max(target, 1)[1]

        pred_labels = torch.max(output, 1)[1]

        for idx in range(len(true_labels)):

            if true_labels[idx] != pred_labels[idx]:

                errors.append((np.transpose(data[idx].cpu().numpy(), (1,2,0)), true_labels[idx], pred_labels[idx]))

    epoch_loss = test_loss / len(test_loader)

    epoch_acc = test_acc / len(test_loader)

    print('{} \t Test Loss: {:.4f}'.format(net_name,epoch_loss))

    print('{}\t Test Accuracy: {:.4f}'.format(net_name,epoch_acc))

    return errors
net_densenet = torch.load('trained_densenet201.pkl')

densenet201_errors = test('densenet201',[],net_densenet)
net_resnet = torch.load('trained_resnet50.pkl')

resnet50_errors = test('resnet50',[],net_resnet)