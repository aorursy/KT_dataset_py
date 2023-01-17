import pandas as pd

import numpy as np

import cv2

import matplotlib.pyplot as plt

import os
train_dir = '../input/histopathologic-cancer-detection/train/'
train = pd.read_csv("../input/refs-22/Train.csv")

test = pd.read_csv("../input/refs-22/test.csv")

valid = pd.read_csv("../input/refs-22/Valid.csv")
import torch

import torch.nn as nn

import torch.optim as optim

from torch.optim import lr_scheduler

import torchvision

from torchvision import datasets, models, transforms

import time

import copy

import torch

from torch.utils import data

from torch.utils.data import DataLoader, Dataset
train.head()
train = train.sample(10000).reset_index(drop = True)

valid = valid.sample(1000).reset_index(drop = True)
class imload(Dataset):

    def __init__(self, csv, transform=None):

        self.data = csv

        self.transform = transform

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        im_link = self.data.iloc[index,0]

        image = cv2.imread(train_dir+im_link+".tif")

        label = self.data.iloc[index,1]

        if self.transform is not None:

            image = self.transform(image)

        return image, label
data_dir = '../hp_train/'
# Data augmentation and normalization for training

# Just normalization for validation

data_transforms = {

    'train': transforms.Compose([

        transforms.ToPILImage(),

        transforms.Resize(224),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]),

    'valid': transforms.Compose([

        transforms.ToPILImage(),

        transforms.Resize(224),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]),

}
train_dataset = imload(train, transform=data_transforms['train'])

valid_dataset = imload(valid, transform=data_transforms['valid'])
# we can access and get data with index by __getitem__(index)

img, lab = train_dataset.__getitem__(0)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device
def imshow(inp, title=None):

    """Imshow for Tensor."""

    f,ax = plt.subplots(figsize = (10,4))

    inp = inp.numpy().transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])

    std = np.array([0.229, 0.224, 0.225])

    inp = std * inp + mean

    inp = np.clip(inp, 0, 1)

    plt.imshow(inp)

    if title is not None:

        plt.title(title)

    plt.pause(0.001)  # pause a bit so that plots are updated





# Get a batch of training data

inputs, classes = next(iter(train_loader))

# Make a grid from batch

out = torchvision.utils.make_grid(inputs)

imshow(out)
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):

    from tqdm import tqdm

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0

    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        print('-' * 10)

        

        ###############################################################

        #Train

        model.train()

        running_loss = 0.0

        running_corrects = 0



        # Iterate over data.

        for inputs, labels in tqdm(train_loader):

            inputs = inputs.to(device)

            labels = labels.to(device)



            # zero the parameter gradients

            optimizer.zero_grad()



            # forward

            # track history if only in train

            with torch.set_grad_enabled(True):

                outputs = model(inputs)

                _, preds = torch.max(outputs, 1)

                loss = criterion(outputs, labels)



                # backward + optimize only if in training phase

                loss.backward()

                optimizer.step()



            # statistics

            running_loss += loss.item() * inputs.size(0)

            running_corrects += torch.sum(preds == labels.data)

        

        scheduler.step()



        epoch_loss = running_loss / 10000#dataset_sizes[phase]

        epoch_acc = running_corrects.double() / 10000#dataset_sizes[phase]

        print('{} Loss: {:.4f} Acc: {:.4f}'.format("Train", epoch_loss, epoch_acc))

        

        #################################################################

        #Valid

        model.eval()   # Set model to evaluate mode



        running_loss = 0.0

        running_corrects = 0



        # Iterate over data.

        for inputs, labels in tqdm(valid_loader):

            inputs = inputs.to(device)

            labels = labels.to(device)



            # zero the parameter gradients

            optimizer.zero_grad()



            # statistics

            running_loss += loss.item() * inputs.size(0)

            running_corrects += torch.sum(preds == labels.data)





        epoch_loss = running_loss / 1000

        epoch_acc = running_corrects.double() / 1000



        print('{} Loss: {:.4f} Acc: {:.4f}'.format("Valid", epoch_loss, epoch_acc))



        # deep copy the model

        if epoch_acc > best_acc:

            best_acc = epoch_acc

            best_model_wts = copy.deepcopy(model.state_dict())

        print()



    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights

    model.load_state_dict(best_model_wts)

    return model
model_ft = models.resnet18(pretrained=True)

num_ftrs = model_ft.fc.in_features

# Here the size of each output sample is set to 2.

# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).

model_ft.fc = nn.Linear(num_ftrs, 2)



model_ft = model_ft.to(device)



criterion = nn.CrossEntropyLoss()



# Observe that all parameters are being optimized

optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)



# Decay LR by a factor of 0.1 every 7 epochs

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
model_conv = torchvision.models.resnet18(pretrained=True)

for param in model_conv.parameters():

    param.requires_grad = False



# Parameters of newly constructed modules have requires_grad=True by default

num_ftrs = model_conv.fc.in_features

model_conv.fc = nn.Linear(num_ftrs, 2)



model_conv = model_conv.to(device)



criterion = nn.CrossEntropyLoss()



# Observe that only parameters of final layer are being optimized as

# opposed to before.

optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)



# Decay LR by a factor of 0.1 every 7 epochs

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
#model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,

#                       num_epochs=1)
model_conv = train_model(model_conv, criterion, optimizer_conv,

                         exp_lr_scheduler, num_epochs=1)