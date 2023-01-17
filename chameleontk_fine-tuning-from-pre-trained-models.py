# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

    
from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

data_dir = '../input/shopee-product-detection-open/'



dftrain = pd.read_csv(data_dir+"train.csv")
dftest = pd.read_csv(data_dir+"test.csv")


dftrain['dir'] = dftrain['category'].astype(str).str.zfill(2)
dftrain.head()

batch_size = 1024
use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

input_size = 224
num_epochs = 25
VERSION = 6
from torchvision import datasets
import os 

# Data augmentation and normalization for training
# Just normalization for validation

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")

all_image_dataset = datasets.ImageFolder(os.path.join(data_dir, "train/train/train"), data_transforms["train"])
test_image_dataset = datasets.ImageFolder(os.path.join(data_dir, "test/test"), data_transforms["val"])
test_loader = torch.utils.data.DataLoader(test_image_dataset, batch_size=batch_size, num_workers=8)
import random
from torch.utils.data import SubsetRandomSampler
# Split data

# Creating data indices for training and validation splits:  
dataset_size = len(all_image_dataset)
validation_split = 0.2
seed = 0

np.random.seed(seed)
random.seed(seed)

indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
np.random.shuffle(indices)

train_indices, val_indices = indices[split:], indices[:split]

# Creating data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)


train_loader = torch.utils.data.DataLoader(all_image_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=8)
val_loader = torch.utils.data.DataLoader(all_image_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=8)

print(len(train_indices), len(val_indices))
print("train loader:",len(train_loader)*batch_size)
print("val loader:",len(val_loader)*batch_size)



dataloaders_dict = {
  "train": train_loader,
  "val": val_loader,
  "test": test_loader,
}

class_names = all_image_dataset.classes
num_classes = len(class_names)
num_classes

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg19_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet161(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


# Initialize the model for this run
model_name = "resnet"
feature_extract = True
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

# Print the model we just instantiated
print(model_ft)

model_ft.cuda()

params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
# optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
optimizer_ft = optim.Adam(params_to_update, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# model

import pickle
ep = 15
print(f"Load modelv{VERSION}_t{ep}.pkl")
with open(data_dir+f'../pretrainedmodele15/modelv{VERSION}_t9.pkl', 'rb') as pickled:
    model = pickle.load(pickled)

print(f"Load optmizer{VERSION}_t{ep}.pkl")
with open(data_dir+f'../pretrainedmodele15/optmizer{VERSION}_t{ep}.pkl', 'rb') as pickled:
    optimizer_ft = pickle.load(pickled)

model_ft = model
model_ft.cuda();
import pickle
import time
import copy
import gc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, start_epoch=0):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(start_epoch, num_epochs):
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

            print("==> Phase", phase)

            idx = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                idx += 1

                if idx%20 == 0:
                  print(f"{idx}/{len(dataloaders[phase])}")

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    
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
            
            dbfile = open(f'./modelv{VERSION}_t{str(epoch)}.pkl', 'wb') 
            pickle.dump(model, dbfile)
            dbfile.close() #Dont forget this 
            
            dbfile = open(f'./optmizer{VERSION}_t{str(epoch)}.pkl', 'wb') 
            pickle.dump(optimizer, dbfile)
            dbfile.close() #Dont forget this 

            # o = predict(model, str(epoch))
        gc.collect()
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history



# Train and evaluate
best_model, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, start_epoch=6)
%%time

from tqdm import tqdm

def run_model(model, dataloaders):
    since = time.time()

    all_preds = np.array([])  
    model.eval()   # Set model to evaluate mode

    # Iterate over data.
    total_batch = len(dataloaders["test"])
    with tqdm(total=total_batch) as pbar:
      for inputs, labels in dataloaders["test"]:
          inputs = inputs.to(device)
          labels = labels.to(device)

    
          outputs = model(inputs)
          _, preds = torch.max(outputs, 1)

          all_preds = np.concatenate((all_preds, preds.cpu().data.numpy()), axis=0)
          pbar.update(1)
          

    return all_preds


all_preds = run_model(model_ft, dataloaders_dict)
print()
def split(x):
  return x.replace("../input/shopee-product-detection-open/test/test/test/", "")
  
filenames = [f for (f, _) in dataloaders_dict["test"].dataset.samples]

o = pd.DataFrame({
    "filename": filenames[0:len(all_preds)],
    "category": all_preds
})

o["category"] = o['category'].astype(float).astype(int).astype(str).str.zfill(2)
o["filename"] = o["filename"].apply(split)

A = set(dftest["filename"])
o = o[o["filename"].isin(A)]

o.to_csv(f"./submission{VERSION}k.csv", index=False)

print(f"submission{VERSION}.csv")
print(o.shape)
o.head()

# o.shape
12186
