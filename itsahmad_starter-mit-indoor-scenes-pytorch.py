from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time

%matplotlib inline
import torch

import torchvision

import torch.nn as nn
from PIL import Image

def pil_loader(path):

    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)

    with open(path, 'rb') as f:

        img = Image.open(f)

        return img.convert('RGB')



def find_classes(dir):

    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]

    classes.sort()

    class_to_idx = {classes[i]: i for i in range(len(classes))}

    return classes, class_to_idx



def default_loader(path):

    from torchvision import get_image_backend

    if get_image_backend() == 'accimage':

        return accimage_loader(path)

    else:

        return pil_loader(path)



from torchvision import datasets

class Scenes(datasets.ImageFolder):

    def __init__(self, root, transform=None, target_transform=None,

                 train=True):

        imagelist_file = 'Images.txt'

        if train:

            imagelist_file = 'Train'+imagelist_file

        else :

            imagelist_file = 'Test' + imagelist_file

        filesnames = open(os.path.join(root, imagelist_file)).read().splitlines()

        self.root = os.path.join(root, 'indoorCVPR_09/Images')

        classes, class_to_idx = find_classes(self.root)



        images = []



        for filename in list(set(filesnames)):

            target = filename.split('/')[0]

            path = os.path.join(root, 'indoorCVPR_09/Images/' + filename)

            item = (path, class_to_idx[target])

            images.append(item)



        self.classes = classes

        self.class_to_idx = class_to_idx

        self.samples = images



        self.imgs = self.samples

        self.loader = default_loader

        self.transform = transform

        self.target_transform = target_transform



    def __len__(self):

        return len(self.samples)
datasets = {x : Scenes(root='../input/', train=True if x == 'train' else False)

            for x in ['train', 'val']}

print("number of Datasets {}.".format(len(datasets)))

print('Number of Classes {}.'.format(len(datasets['train'].classes)))

for dstype in datasets:

    print('{} set contains {} images'.format(dstype, len(datasets[dstype])))
# Data transforms

from torchvision import transforms



data_transforms = {

    'train': transforms.Compose([

        transforms.RandomResizedCrop(224),

        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]),

    'val': transforms.Compose([

        transforms.Resize(256),

        transforms.CenterCrop(224),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]),

}
datasets = {x : Scenes(root='../input/', train=True if x == 'train' else False, 

                       transform=data_transforms[x])

            for x in ['train', 'val']}

class_names = datasets['train'].classes

dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
from torch.utils import data

dataloaders = {x: data.DataLoader(datasets[x], batch_size=4,

                                             shuffle=True, num_workers=1)

              for x in ['train', 'val']}
def imshow(inp, title=None):

    """Imshow for Tensor."""

    inp = inp.numpy().transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])

    std = np.array([0.229, 0.224, 0.225])

    inp = std * inp + mean

    inp = np.clip(inp, 0, 1)

    plt.imshow(inp)

    if title is not None:

        plt.title(title)

    plt.pause(0.01)  # pause a bit so that plots are updated
# Get a batch of training data

inputs, classes = next(iter(dataloaders['train']))



# Make a grid from batch

import torchvision

out = torchvision.utils.make_grid(inputs)



imshow(out, title=[class_names[x] for x in classes])