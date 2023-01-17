import torch

import pandas as pd

import numpy as np

from torch.utils.data import Dataset, random_split, DataLoader

from PIL import Image

import torchvision.models as models

import matplotlib.pyplot as plt

import torchvision.transforms as transforms

from sklearn.metrics import f1_score

import torch.nn.functional as F

import torch.nn as nn

from torchvision.utils import make_grid

%matplotlib inline
DATA_DIR = '../input/jovian-pytorch-z2g/Human protein atlas'



TRAIN_DIR = DATA_DIR + '/train'                           # Contains training images

TEST_DIR = DATA_DIR + '/test'                             # Contains test images



TRAIN_CSV = DATA_DIR + '/train.csv'                       # Contains real labels for training images

TEST_CSV = '../input/jovian-pytorch-z2g/submission.csv'   # Contains dummy labels for test image
labels = {

    0: 'Mitochondria',

    1: 'Nuclear bodies',

    2: 'Nucleoli',

    3: 'Golgi apparatus',

    4: 'Nucleoplasm',

    5: 'Nucleoli fibrillar center',

    6: 'Cytosol',

    7: 'Plasma membrane',

    8: 'Centrosome',

    9: 'Nuclear speckles'

    

}    
## used for label encoding 

## converts 2,3 -> [0, 0, 1, 1, 0, 0, 0, 0, 0, 0]

def encode_label(label):

    #create an initial target which is all zeros

    target = torch.zeros(10)

    for l in str(label).split(' '):

        target[int(l)] = 1.

    return target
## used for label decoding

## converts converts [0, 0, 1, 1, 0, 0, 0, 0, 0, 0] ->  2,3 

def decode_target(target, text_labels=False, threshold=0.5):

    result = []

    for i, x in enumerate(target):

        if (x >= threshold):

            if text_labels:

                result.append(labels[i] + "(" + str(i) + ")")

            else:

                result.append(str(i))

    return ' '.join(result)
## used to show a single image and its label

def show_sample(img, target, invert=True):

    plt.figsize=(8,4)

    if invert:

        plt.imshow(1 - img.permute((1, 2, 0)))

    else:

        plt.imshow(img.permute(1, 2, 0))

    print('Labels:', decode_target(target, text_labels=True))
## used to show orginal images and its transformation

def show_difference(img1, target1, img2, target2, invert=True):

    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(8,4))

    ax1.set_title('Before Transformation')

    ax2.set_title('After Transformation')

    if invert:

        ax1.imshow(1 - img1.permute((1, 2, 0)))

        ax2.imshow(1 - img2.permute((1, 2, 0)))

    else:

        ax1.imshow(img1.permute((1, 2, 0)))

        ax2.imshow(img2.permute((1, 2, 0)))
# Creating torch dataset





class HumanProteinDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):

        self.df = pd.read_csv(csv_file)

        self.transform = transform

        self.root_dir = root_dir

        

    def __len__(self):

        return len(self.df)    

    

    def __getitem__(self, idx):

        row = self.df.loc[idx]

        img_id, img_label = row['Image'], row['Label']

        img_fname = self.root_dir + "/" + str(img_id) + ".png"

        img = Image.open(img_fname)

        if self.transform:

            img = self.transform(img)

        return img, encode_label(img_label)
transform = transforms.Compose([transforms.ToTensor()])

dataset = HumanProteinDataset(TRAIN_CSV, TRAIN_DIR, transform=transform)
show_sample(*dataset[0])
transform = transforms.Compose([transforms.CenterCrop(256),transforms.ToTensor()])

dataset1 = HumanProteinDataset(TRAIN_CSV, TRAIN_DIR, transform=transform)

show_difference(*dataset[0], *dataset1[0])
transform = transforms.Compose([transforms.ColorJitter(brightness=(0,10), contrast=(10,20), saturation=(30,40), hue=(-.5,.5)),transforms.ToTensor()])

dataset1 = HumanProteinDataset(TRAIN_CSV, TRAIN_DIR, transform=transform)

show_difference(*dataset[0], *dataset1[0])
transform = transforms.Compose([transforms.Grayscale(3),transforms.ToTensor()])

dataset1 = HumanProteinDataset(TRAIN_CSV, TRAIN_DIR, transform=transform)

show_difference(*dataset[0], *dataset1[0])
transform = transforms.Compose([transforms.Pad(padding=(10,20,30,40), fill=(255,0,0)),transforms.ToTensor()])

dataset1 = HumanProteinDataset(TRAIN_CSV, TRAIN_DIR, transform=transform)

show_difference(*dataset[0], *dataset1[0])
transform = transforms.Compose([transforms.RandomAffine(degrees=(-45,+45), scale=(1,2)),transforms.ToTensor()])

dataset1 = HumanProteinDataset(TRAIN_CSV, TRAIN_DIR, transform=transform)

show_difference(*dataset[0], *dataset1[0])
random_apply = [transforms.RandomAffine(degrees=(-45,+45), scale=(1,2)), transforms.Grayscale(3), transforms.ColorJitter(brightness=(0,10), contrast=(10,20), saturation=(10,15), hue=(-.5,.5))]

transform = transforms.Compose([transforms.RandomApply(random_apply, p=0.8),transforms.ToTensor()])

dataset1 = HumanProteinDataset(TRAIN_CSV, TRAIN_DIR, transform=transform)

show_difference(*dataset[0], *dataset1[0])
random_apply = [transforms.RandomAffine(degrees=(-45,+45), scale=(1,2)), transforms.Grayscale(3), transforms.ColorJitter(brightness=(0,10), contrast=(10,20), saturation=(10,15), hue=(-.5,.5))]

transform = transforms.Compose([transforms.RandomChoice(random_apply),transforms.ToTensor()])

dataset1 = HumanProteinDataset(TRAIN_CSV, TRAIN_DIR, transform=transform)

show_difference(*dataset[0], *dataset1[0])
transform = transforms.Compose([transforms.RandomCrop(128),transforms.ToTensor()])

dataset1 = HumanProteinDataset(TRAIN_CSV, TRAIN_DIR, transform=transform)

show_difference(*dataset[0], *dataset1[0])
transform = transforms.Compose([transforms.RandomResizedCrop(size=512, scale=(0.08, 1.0), ratio=(0.75, 1.33), interpolation=2),transforms.ToTensor()])

dataset1 = HumanProteinDataset(TRAIN_CSV, TRAIN_DIR, transform=transform)

show_difference(*dataset[0], *dataset1[0])
transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.9),transforms.ToTensor()])

dataset1 = HumanProteinDataset(TRAIN_CSV, TRAIN_DIR, transform=transform)

show_difference(*dataset[0], *dataset1[0])
transform = transforms.Compose([transforms.RandomVerticalFlip(p=0.9),transforms.ToTensor()])

dataset1 = HumanProteinDataset(TRAIN_CSV, TRAIN_DIR, transform=transform)

show_difference(*dataset[0], *dataset1[0])