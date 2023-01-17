import torch

from torch import nn

from torch.utils.data import DataLoader

import torchvision

from torchvision import datasets

import torchvision.transforms as transforms

from torch.utils.data.sampler import SubsetRandomSampler



import matplotlib.pyplot as plt

import numpy as np

import os

from PIL import Image

# Example Dataset

TRAIN_DIR = "../input/hot-dog-not-hot-dog/train/"

for file in os.listdir(TRAIN_DIR):

    print(file)
#Basic Transforms

SIZE = (200,200)

basic = transforms.Compose([transforms.Resize(SIZE),

                            transforms.ToTensor()])

# Normalized transforms

mean = (0.5, 0.5, 0.5)

std = (0.5, 0.5, 0.5)

norm_tran = transforms.Compose([transforms.Resize(SIZE),

                                transforms.ToTensor(), 

                                transforms.Normalize(mean=mean, std=std)])



#Simple Data Augmentation

# Data augmentations

aug_tran = transforms.Compose([transforms.Resize(SIZE),

                               transforms.RandomHorizontalFlip(),

                               transforms.RandomRotation(45),

                               transforms.ToTensor(),transforms.Normalize(mean=mean, std=std)])
# Create Dataset

basic_train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=basic)

norm_train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=norm_tran)

aug_train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=aug_tran)
to_pil = torchvision.transforms.ToPILImage()
data, lable = next(iter(basic_train_dataset))

print(lable)

img = to_pil(data)

plt.imshow(img)
# Data loaders

# Parameters for setting up data loaders

BATCH_SIZE = 20

NUM_WORKERS = 0

VALIDATION_SIZE = 0.15



basic_train_loader = DataLoader(basic_train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

norm_train_loader = DataLoader(norm_train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

aug_train_loader = DataLoader(aug_train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

def imshow(img):

    img = img / 2 + 0.5  # unnormalize

    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image
# Simple visuzlization for dataloaders to check what they are producing. 

basic_dataiter = iter(basic_train_loader)

images, labels = basic_dataiter.next()

images = images.numpy()

fig = plt.figure(figsize=(25, 4))

# display 20 images

for idx in np.arange(20):

    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])

    imshow(images[idx])

    ax.set_title([labels[idx]])

# Simple visuzlization for dataloaders to check what they are producing. 

norm_dataiter = iter(norm_train_loader)

images, labels = norm_dataiter.next()

images = images.numpy()

fig = plt.figure(figsize=(25, 4))

# display 20 images

for idx in np.arange(20):

    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])

    imshow(images[idx])

    ax.set_title([labels[idx]])

# Simple visuzlization for dataloaders to check what they are producing. 

aug_dataiter = iter(aug_train_loader)

images, labels = basic_dataiter.next()

images = images.numpy()

fig = plt.figure(figsize=(25, 4))

# display 20 images

for idx in np.arange(20):

    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])

    imshow(images[idx])

    ax.set_title([labels[idx]])