from __future__ import print_function

from __future__ import division

import torch

from pathlib import Path

from torch.utils.data import Dataset, DataLoader

import cv2

import torch.nn.functional as F

import torch.nn as nn

import torch.optim as optim

import pandas as pd

import numpy as np

import torchvision

from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt

import time

import os

import copy

print("PyTorch Version: ",torch.__version__)

print("Torchvision Version: ",torchvision.__version__)
path = "../input/train/"



def load_train(path):

    train_set = pd.read_csv('../input/train_labels.csv')

    train_label = np.array(train_set['invasive'].iloc[: ])

    train_files = []

    for i in range(len(train_set)):

        train_files.append(path + str(int(train_set.iloc[i][0])) +'.jpg')

    train_set['name'] = train_files

    return train_files, train_set, train_label



train_files, train_set, train_label = load_train(path)



train_set.head()
path = "../input/test/"



def load_test(path):

    test_set = pd.read_csv('../input/sample_submission.csv')

    test_files = []

    for i in range(len(test_set)):

        test_files.append(path + str(int(test_set.iloc[i][0])) +'.jpg')

    return test_files



test_files = load_test(path)



test_files[:5]
use_pretrained = True

feature_extract = True

num_classes = 2
inception = models.inception_v3(pretrained=True)
# Handle the auxilary net

model = inception

for param in model.parameters():

            param.requires_grad = False

num_ftrs = model.AuxLogits.fc.in_features

model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)

# Handle the primary net

num_ftrs = model.fc.in_features

model.fc = nn.Linear(num_ftrs,num_classes)

input_size = 299
model = model.cuda()
def augment(src, choice):

    if choice == 0:

        # Rotate 90

        src = np.rot90(src, 1)

    if choice == 1:

        # flip vertically

        src = np.flipud(src)

    if choice == 2:

        # Rotate 180

        src = np.rot90(src, 2)

    if choice == 3:

        # flip horizontally

        src = np.fliplr(src)

    if choice == 4:

        # Rotate 90 counter-clockwise

        src = np.rot90(src, 3)

    if choice == 5:

        # Rotate 180 and flip horizontally

        src = np.rot90(src, 2)

        src = np.fliplr(src)

    return src
from sklearn.model_selection import train_test_split
train_files, valid_files, train_y, valid_y = train_test_split(train_files, train_label, test_size=0.2)
def normalize(im):

    """Normalizes images with Imagenet stats."""

    imagenet_stats = np.array([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])

    return (im - imagenet_stats[0])/imagenet_stats[1]
class MyDataset(Dataset):

    def __init__(self, files, tags=None, transforms=False, sz=299):

        self.files = files

        self.tags = tags

        self.transforms = transforms

        self.sz = sz

        

    def __len__(self):

        return len(self.files)

    

    def __getitem__(self, idx):

        path = str(self.files[idx]) 

        img = cv2.imread(str(path)).astype(np.float32)

        img = cv2.resize(img, (self.sz, self.sz))

        

        img = img/255

        # center crop

        if self.transforms:

            img = augment(img, np.random.randint(6))

        # substract numbers from resnet34

        x = normalize(img)

        if self.tags is None:

            return torch.tensor(np.rollaxis(x, 2)).float()

        else:

            return torch.tensor(np.rollaxis(x, 2)).float(), self.tags[idx]
train_ds = MyDataset(train_files, train_y, transforms=True)

valid_ds = MyDataset(valid_files, valid_y)

test_ds = MyDataset(test_files)
batch_size = 64

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

valid_dl = DataLoader(valid_ds, batch_size=batch_size)

test_dl = DataLoader(test_ds, batch_size=batch_size)
# Create training and validation dataloaders

dataloaders_dict = {'train': DataLoader(train_ds, batch_size=batch_size,

                                        shuffle=True, num_workers=4),

                    'val': DataLoader(valid_ds, batch_size=batch_size,

                                        shuffle=False, num_workers=4)}

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):

    since = time.time()



    val_acc_history = []



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

                inputs = inputs.cuda()

                labels = labels.cuda()



                # zero the parameter gradients

                optimizer.zero_grad()



                # forward

                # track history if only in train

                with torch.set_grad_enabled(phase == 'train'):

                    # Get model outputs and calculate loss

                    # Special case for inception because in training it has an auxiliary output. In train

                    #   mode we calculate the loss by summing the final output and the auxiliary output

                    #   but in testing we only consider the final output.

                    if is_inception and phase == 'train':

                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958

                        outputs, aux_outputs = model(inputs)

                        loss1 = criterion(outputs, labels)

                        loss2 = criterion(aux_outputs, labels)

                        loss = loss1 + 0.4*loss2

                    else:

                        outputs = model(inputs)

                        loss = criterion(outputs, labels)



                    _, preds = torch.max(outputs, 1)



                    # backward + optimize only if in training phase

                    if phase == 'train':

                        loss.backward()

                        optimizer.step()



                # statistics

                running_loss += loss.item() * inputs.size(0)

                running_corrects += torch.sum(preds == labels.data)



            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)



            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))



            # deep copy the model

            if phase == 'val' and epoch_acc > best_acc:

                best_acc = epoch_acc

                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'val':

                val_acc_history.append(epoch_acc)



        print()



    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val Acc: {:4f}'.format(best_acc))



    # load best model weights

    model.load_state_dict(best_model_wts)

    return model, val_acc_history
    # Send the model to GPU

    model = model.cuda()



    # Gather the parameters to be optimized/updated in this run. If we are

    #  finetuning we will be updating all parameters. However, if we are

    #  doing feature extract method, we will only update the parameters

    #  that we have just initialized, i.e. the parameters with requires_grad

    #  is True.

    params_to_update = model.parameters()

    print("Params to learn:")

    if feature_extract:

        params_to_update = []

        for name,param in model.named_parameters():

            if param.requires_grad == True:

                params_to_update.append(param)

                print("\t",name)

    else:

        for name,param in model.named_parameters():

            if param.requires_grad == True:

                print("\t",name)
num_epochs = 15

optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

# Setup the loss fxn

criterion = nn.CrossEntropyLoss()



# Train and evaluate

model_ft, hist = train_model(model, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=True)
submission = pd.read_csv('../input/sample_submission.csv')
submission.head()
test_preds = np.zeros_like(submission.invasive.values)
test_preds = np.zeros_like(submission.invasive.values)

for i, x_batch in enumerate(test_dl):

    y_pred = model_ft(x_batch.cuda()).detach()

#     print(y_pred.shape)

#     print(y_pred.argmax(dim=1).shape)

    test_preds[i * batch_size:(i+1) * batch_size] = y_pred.argmax(dim=1).cpu()
submission.invasive = test_preds

submission.to_csv('submission.csv', index=False)