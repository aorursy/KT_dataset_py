from shutil import copyfile

copyfile(src = "../input/modules/dataset_det.py", dst = "../working/dataset_det.py")

copyfile(src = "../input/trained-model/model_3conv", dst = "../working/model")
import torch

import torch.nn as nn

import torch.optim as optim

from torch.optim import lr_scheduler

import numpy as np

import torchvision

from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt

import time

import os

import copy



import dataset_det as d_loader

import torch.utils.data as torch_d

from torch.nn import functional as F
def calcAcc (net, dataloader):

    correct = 0

    with torch.no_grad():

        for data in dataloader:

            images, labels, bb= data

            images = images.to(device)

            labels = labels.to(device)

            outputs = net(images)

            _, predicted = torch.topk(outputs.data,k = 3,dim = 1)

            _, groundtruth = torch.topk(labels.data,k = 3, dim = 1)

            for i in range(predicted.size(0)):

                correct += len(set(predicted[i].cpu().numpy()).intersection(set(groundtruth[i].cpu().numpy())))



    return (correct / (len(dataloader.dataset) * 3))
def train_model(dataloaders, model, criterion, optimizer, scheduler, num_epochs=25):

    since = time.time()



    best_model_wts = copy.deepcopy(model.state_dict())

    

    best_acc = calcAcc(model, dataloaders["val"])

    print('Initial val Acc: {}'.format(best_acc))



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

            for inputs, labels, bb in dataloaders[phase]:

                inputs = inputs.to(device)

                labels = labels.to(device)



                # zero the parameter gradients

                optimizer.zero_grad()



                # forward

                # track history if only in train

                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)

                    # _, preds = torch.max(outputs, 1)

                    _, preds = torch.topk(outputs.data,k = 3,dim = 1)

                    _, groundtruth = torch.topk(labels.data,k = 3, dim = 1)

                    loss = criterion(outputs, labels)



                    # backward + optimize only if in training phase

                    if phase == 'train':

                        loss.backward()

                        optimizer.step()



                # statistics

                running_loss += loss.item() * inputs.size(0)

                for i in range(preds.size(0)):

                    running_corrects += len(set(preds[i].cpu().numpy()).intersection(set(groundtruth[i].cpu().numpy())))

            if phase == 'train' and scheduler != None:

                scheduler.step()



            epoch_loss = running_loss / (len(dataloaders[phase].dataset))

            epoch_acc = running_corrects / (len(dataloaders[phase].dataset) * 3)



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
class LeNet(torch.nn.Module):

    def __init__(self):

        super(LeNet, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels = 6, kernel_size=5, groups=1)

        self.conv2 = torch.nn.Conv2d(6, 16, 5, 1)

        self.conv3 = torch.nn.Conv2d(16, 30, 5, 1)

        self.conv4 = torch.nn.Conv2d(30, 10, 5, 1)

        self.pool1 = torch.nn.MaxPool2d(2,2)

        self.fc1 = torch.nn.Linear(10*5*5, 500)

        self.fc1_1 = torch.nn.Linear(500, 2000)

        self.fc2 = torch.nn.Linear(2000, 9)

        self.drout = torch.nn.Dropout(p=0.5)



    def forward(self, x):

        x = self.pool1(F.relu(self.conv1(x)))

        x = self.pool1(F.relu(self.conv2(x)))

        x = self.pool1(F.relu(self.conv3(x)))

        x = F.relu(self.conv4(x))

        x = x.view(-1,10*5*5)

        x = self.drout(F.relu(self.fc1(x)))

        x = self.drout(F.relu(self.fc1_1(x)))

        x = self.fc2(x)

        x = torch.sigmoid(x)

        return x
class LeNet2(torch.nn.Module):

    def __init__(self):

        super(LeNet2, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels = 15, kernel_size=5, groups=1)

        self.conv2 = torch.nn.Conv2d(15, 25, 5, 1)

        self.pool = torch.nn.MaxPool2d(2,2)

        self.fc1 = torch.nn.Linear(25*22*22, 100)

        self.fc2 = torch.nn.Linear(100, 9)

        self.dropout = torch.nn.Dropout(0.3) 



    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))

        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1,25*22*22)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        x = torch.sigmoid(x)

        return x
BATCHSIZE=50



dataset = d_loader.Balls_CF_Detection ("../input/balls-images/train", 20999)

train_dataset, test_dataset = torch_d.random_split(dataset, [18899, 2100])



dataloaders = {}

dataloaders['train'] = torch.utils.data.DataLoader(train_dataset,batch_size=BATCHSIZE, shuffle=True)

dataloaders['val'] = torch.utils.data.DataLoader(test_dataset,batch_size=BATCHSIZE, shuffle=True)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = LeNet2().to(device)



resume_training = False



if resume_training:

    print("Loading pretrained model..")

    model.load_state_dict(torch.load('./model'))

    print("Loaded!")
criterion = torch.nn.BCELoss()



#optimizer = torch.optim.SGD(model.parameters(), lr=0.0000001, momentum=0.9)

optimizer = optim.Adam(model.parameters(), lr=0.0001)



# Decay LR by a factor of 0.1 every 7 epochs

scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
EPOCH_NUMBER = 50



model = train_model(dataloaders, model, criterion, optimizer, None,

                       num_epochs=EPOCH_NUMBER)



torch.save(model.state_dict(), "./model")
def calcError (net, dataloader):

    correct = 0

    total = 0

    with torch.no_grad():

        for data in dataloader:

            images, labels, bb= data

            images = images.to(device)

            labels = labels.to(device)

            outputs = net(images)

            _, predicted = torch.topk(outputs.data,k = 3,dim = 1)

            _, groundtruth = torch.topk(labels.data,k = 3, dim = 1)

            total += labels.size(0)

            for i in range(predicted.size(0)):

                if(len(set(predicted[i].cpu().numpy()).intersection(set(groundtruth[i].cpu().numpy()))) == 3):

                    correct += 1



    return (correct / total)
print(calcError(model, dataloaders["val"]))