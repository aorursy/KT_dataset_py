!pip install albumentations > /dev/null 2>&1

!pip install pretrainedmodels > /dev/null 2>&1

!pip install catalyst > /dev/null 2>&1
import numpy as np

import pandas as pd

import os

import cv2

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

import torch

from torch.utils.data import TensorDataset, DataLoader,Dataset

import torch.nn as nn

import torch.nn.functional as F

import torchvision

import torchvision.transforms as transforms

import torch.optim as optim

import time 

from PIL import Image

train_on_gpu = True

from torch.utils.data.sampler import SubsetRandomSampler

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

from sklearn.metrics import accuracy_score

import cv2



from sklearn.preprocessing import OneHotEncoder



import albumentations

import pretrainedmodels

from albumentations.pytorch import ToTensor



import collections
from catalyst.dl.utils import UtilsFactory

from catalyst.dl.experiments import SupervisedRunner

from catalyst.dl.callbacks import EarlyStoppingCallback, OneCycleLR, InferCallback
data_transforms = albumentations.Compose([

    albumentations.HorizontalFlip(),

    albumentations.VerticalFlip(),

    albumentations.RandomBrightness(),

    albumentations.Normalize(

        mean=[0.485, 0.456, 0.406],

        std=[0.229, 0.224, 0.225],

    ),

    ToTensor()

    ])

data_transforms_test = albumentations.Compose([

    albumentations.Normalize(

        mean=[0.485, 0.456, 0.406],

        std=[0.229, 0.224, 0.225],

    ),

    ToTensor()

    ])
train_df = pd.read_csv('../input/train.csv')

train, valid = train_test_split(train_df.has_cactus, stratify=train_df.has_cactus, test_size=0.1)

# creating dict with image names and labels

img_class_dict = {k:v for k, v in zip(train_df.id, train_df.has_cactus)}
class CactusDataset(Dataset):

    def __init__(self, datafolder, datatype='train', transform = transforms.Compose([transforms.CenterCrop(32),transforms.ToTensor()]), labels_dict={}):

        self.datafolder = datafolder

        self.datatype = datatype

        self.image_files_list = [s for s in os.listdir(datafolder)]

        self.transform = transform

        self.labels_dict = labels_dict

        if self.datatype == 'train':

            self.labels = [np.float32(labels_dict[i]) for i in self.image_files_list]

        else:

            self.labels = [np.float32(0.0) for _ in range(len(self.image_files_list))]



    def __len__(self):

        return len(self.image_files_list)



    def __getitem__(self, idx):

        img_name = os.path.join(self.datafolder, self.image_files_list[idx])

        img = cv2.imread(img_name)[:,:,::-1]

        image = self.transform(image=img)

        image = image['image']

        label = self.labels[idx]

        

        return image, label
dataset = CactusDataset(datafolder='../input/train/train', datatype='train', transform=data_transforms, labels_dict=img_class_dict)

test_set = CactusDataset(datafolder='../input/test/test', datatype='test', transform=data_transforms_test)
loaders = collections.OrderedDict()



train_sampler = SubsetRandomSampler(list(train.index))

valid_sampler = SubsetRandomSampler(list(valid.index))

batch_size = 512

num_workers = 0

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)

valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)



loaders["train"] = train_loader

loaders["valid"] = valid_loader

loaders["test"] = test_loader
class Flatten(nn.Module):

    def forward(self, input):

        return input.view(input.size(0), -1)

    

class Net(nn.Module):

    def __init__(

            self,

            num_classes: int,

            p: float = 0.2,

            pooling_size: int = 2,

            last_conv_size: int = 1664,

            arch: str = "densenet169",

            pretrained: str = "imagenet") -> None:

        """A simple model to finetune.

        

        Args:

            num_classes: the number of target classes, the size of the last layer's output

            p: dropout probability

            pooling_size: the size of the result feature map after adaptive pooling layer

            last_conv_size: size of the flatten last backbone conv layer

            arch: the name of the architecture form pretrainedmodels

            pretrained: the mode for pretrained model from pretrainedmodels

        """

        super().__init__()

        net = pretrainedmodels.__dict__[arch](pretrained=pretrained)

        modules = list(net.children())[:-1]  # delete last layer

        # add custom head

        modules += [nn.Sequential(

            Flatten(),

            nn.BatchNorm1d(1664),

            nn.Dropout(p),

            nn.Linear(1664, num_classes)

        )]

        self.net = nn.Sequential(*modules)



    def forward(self, x):

        logits = self.net(x)

        return torch.squeeze(logits)
# experiment setup

num_epochs = 10

logdir = "./logs/simple"



# model, criterion, optimizer

model = Net(num_classes=1)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(model.parameters(), momentum=0.99, lr=1e-2)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=2)
# model runner

runner = SupervisedRunner()



# model training

runner.train(

    model=model,

    criterion=criterion,

    optimizer=optimizer,

    scheduler=scheduler, 

    loaders=loaders,

    logdir=logdir,

    callbacks=[

        OneCycleLR(

            cycle_len=num_epochs, 

            div_factor=3,

            increase_fraction=0.3,

            momentum_range=(0.95, 0.85))

    ],

    num_epochs=num_epochs,

    verbose=False

)
# plotting training progress

UtilsFactory.plot_metrics(logdir=logdir)
model = Net(num_classes=1)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(model.parameters(), momentum=0.99, lr=1e-2)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=2)



num_epochs = 20

logdir1 = "./logs/simple1"

# model runner

runner1 = SupervisedRunner()



# model training

runner1.train(

    model=model,

    criterion=criterion,

    optimizer=optimizer,

    scheduler=scheduler, 

    loaders=loaders,

    callbacks=[

        EarlyStoppingCallback(patience=4, min_delta=0.01)

    ],

    logdir=logdir1,

    num_epochs=num_epochs,

    verbose=False

)
# plotting training progress

UtilsFactory.plot_metrics(logdir=logdir1)
test_loader = collections.OrderedDict([("infer", loaders["test"])])

runner.infer(

    model=model,

    loaders=test_loader,

    callbacks=[InferCallback()],

)
test_img = os.listdir('../input/test/test')

test_df = pd.DataFrame(test_img, columns=['id'])

test_preds = pd.DataFrame({'imgs': test_df.id.values, 'preds': runner.callbacks[0].predictions["logits"]})

test_preds.columns = ['id', 'has_cactus']

test_preds.to_csv('sub.csv', index=False)

test_preds.head()
runner1.infer(

    model=model,

    loaders=test_loader,

    callbacks=[InferCallback()],

)

test_preds['has_cactus'] = runner1.callbacks[0].predictions["logits"]

test_preds.to_csv('sub1.csv', index=False)