# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import torch

from torchvision import transforms, datasets

from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data import DataLoader

from matplotlib import pyplot as plt

import torchvision.transforms.functional as TF
from PIL import Image

path = '../input/x_ray_images_per_class/images_per_class/Nodule'



name = os.listdir(path)[10]

img1=Image.open(path+"/"+name)

#img = img.resize((new_width, new_height), Image.ANTIALIAS)





fig = plt.figure(figsize=(55,45))

ax = fig.add_subplot(1, 5,  1, xticks=[], yticks=[])

ax.imshow(img1, cmap='gray')

ax.set_title('Nodule')



path2 = '../input/x_ray_images_per_class/images_per_class/Edema'

name2 = os.listdir(path2)[10]

img2 = Image.open(path2+"/"+name2)

ax = fig.add_subplot(1, 5,  2, xticks=[], yticks=[])

ax.imshow(img2, cmap='gray')

ax.set_title('Edema')
batch_size = 64

path = '../input/x_ray_images_per_class/images_per_class/'

transform = transforms.Compose([transforms.ToTensor()])

data = datasets.ImageFolder(path, transform=transform)

print(len(data))

loader = DataLoader(

    data, batch_size=batch_size)



len(loader)
mean = 0.

std = 0.

nb_samples = 0.



for images, _ in loader:

    batch_samples = images.size(0)

    data = images.view(batch_samples, images.size(1), -1)

    mean += data.mean(2).sum(0)

    std += data.std(2).sum(0)

    nb_samples += batch_samples

    break



mean /= nb_samples

std /= nb_samples



print("Mean: ", mean.numpy())

print("Std: ",std.numpy())
batch_size = 64

num_workers = 0



train_transform = transforms.Compose([

    transforms.Resize(256),

    transforms.RandomHorizontalFlip(),

    transforms.Grayscale(3),

    transforms.RandomRotation(20),

    # transforms.ColorJitter(),

    transforms.Lambda(lambda img:TF.rotate(img,20)),

    transforms.ToTensor(),

    # transforms.Normalize(mean, std)

])



valid_transform = transforms.Compose([

    transforms.Resize(256),

    transforms.ToTensor(),

    #transforms.Normalize(mean, std)

])



test_transform = transforms.Compose([

    transforms.Resize(256),

    transforms.ToTensor(),

    #transforms.Normalize(mean, std)

])
train_size = 0.8 # 80 % train data

split_p = 1 - train_size 



path = '../input/x_ray_images_per_class/images_per_class'





train_data = datasets.ImageFolder(path, transform=train_transform)

valid_data = datasets.ImageFolder(path, transform=valid_transform)

test_data = datasets.ImageFolder(path, transform=test_transform)



# mix data

# index of num of train

indices = list(range(len(train_data)))

# random the index

np.random.shuffle(indices)

split = int(np.floor(split_p * len(train_data)))

# divied into two part

train_idx, test_idx = indices[split:], indices[:split]

# now divied into test into test into valid and test

indices2 = list(range(len(test_idx)))

np.random.shuffle(indices2)

split = int(np.floor(0.5 * len(test_idx)))

# divied into two part

valid_idx, test_idx = indices2[split:], indices2[:split]



len(train_idx), len(valid_idx), len(test_idx)
classes = train_data.classes

classes_to_idx = train_data.class_to_idx
# define the sampler

train_sampler = SubsetRandomSampler(train_idx)

valid_sampler = SubsetRandomSampler(valid_idx)

test_sampler = SubsetRandomSampler(test_idx)



# prepare loaders

train_loader = DataLoader(

    train_data, batch_size=batch_size,

    sampler=train_sampler, num_workers=num_workers)



valid_loader = DataLoader(

    valid_data, batch_size=batch_size,

    sampler=valid_sampler, num_workers=num_workers)



test_loader = DataLoader(

    test_data, batch_size=batch_size,

    sampler=test_sampler, num_workers=num_workers)





print("Train Loader: ", len(train_loader))

print("Valid Loader: ", len(valid_loader))

print("Test Loader: ", len(test_loader))
def visualize(loader, classes, num_of_image=2, fig_size=(25, 5)):

    data_iter = iter(loader)

    images, labels = data_iter.next()



    fig = plt.figure(figsize=fig_size)

    for idx in range(num_of_image):

        ax = fig.add_subplot(1, 5, idx + 1, xticks=[], yticks=[])

        # denormalize first

        img = images[idx] #/ 2 + 0.5

        npimg = img.numpy()

        img = np.transpose(npimg, (1, 2, 0))  # transpose

        ax.imshow(img, cmap='gray')

        ax.set_title(classes[labels[idx]])
visualize(train_loader, classes)
visualize(test_loader, classes)