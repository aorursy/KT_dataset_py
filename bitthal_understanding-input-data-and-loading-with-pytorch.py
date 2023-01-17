# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import random

from PIL import Image

from matplotlib import pyplot as plt

import xml.etree.ElementTree as ET



import torch 

import torchvision



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
img_base_path = "../input/all-dogs/all-dogs"

# Files

images = os.listdir(img_base_path)

print("Number of Images: ", len(images))
# choose 16 random images to display

images_to_display = random.choices(images, k=64)



fig = plt.figure(figsize=(25, 16))

for ii, img in enumerate(images_to_display):

    ax = fig.add_subplot(8, 8, ii + 1, xticks=[], yticks=[])

    

    img = Image.open(os.path.join(img_base_path, img))

    plt.imshow(img)
images_to_display[18]
!cat ../input/annotation/Annotation/n02091244-Ibizan_hound/n02091244_2934
# choose image

image_name = 'n02105641_169.jpg'



# read image

img = Image.open(os.path.join(img_base_path, image_name))



# display image

fig = plt.figure(figsize=(8, 12))

plt.imshow(img)
# finding annotation of this particular image

# image_name == dogBreed_number



annotation_folders = os.listdir('../input/annotation/Annotation')

breed_folder = [x for x in annotation_folders if image_name.split('_')[0] in x]

assert len(breed_folder) == 1, "Multiple Folders Found"



breed_folder = breed_folder[0]

print("Image Folder: ", breed_folder)

annotation_path = os.path.join('../input/annotation/Annotation', breed_folder, image_name[:-4])

print("Annotation Path: ", annotation_path)
# View annotations

!cat ../input/annotation/Annotation/n02105641-Old_English_sheepdog/n02105641_169
### Getting Bounding Box of dog (ROI) in Image

tree = ET.parse(annotation_path)

root = tree.getroot()

objects = root.findall('object')

for obj in objects:

    bndbox = obj.find('bndbox')

    xmin = int(bndbox.find('xmin').text)

    ymin = int(bndbox.find('ymin').text)

    xmax = int(bndbox.find('xmax').text)

    ymax = int(bndbox.find('ymax').text)

bbox = (xmin, ymin, xmax, ymax)

print("Bounding Box: ", bbox)



# crop image

img = img.crop(bbox)



# display crop image

fig = plt.figure(figsize=(8, 12))

plt.imshow(img)
# This loader will use the underlying loader plus crop the image based on the annotation

from torchvision import transforms, datasets

from torch.utils.data import DataLoader, Dataset



annotation_folders = os.listdir('../input/annotation/Annotation')

def ImageLoader(path):

    img = datasets.folder.default_loader(path) # default loader

    # Get bounding box

    breed_folder = [x for x in annotation_folders if path.split('/')[-1].split('_')[0] in x][0]

    annotation_path = os.path.join('../input/annotation/Annotation', breed_folder, path.split('/')[-1][:-4])



    tree = ET.parse(annotation_path)

    root = tree.getroot()

    objects = root.findall('object')

    for obj in objects:

        bndbox = obj.find('bndbox')

        xmin = int(bndbox.find('xmin').text)

        ymin = int(bndbox.find('ymin').text)

        xmax = int(bndbox.find('xmax').text)

        ymax = int(bndbox.find('ymax').text)

    bbox = (xmin, ymin, xmax, ymax)

    

    # return cropped image

    img = img.crop(bbox)

    img = img.resize((64, 64), Image.ANTIALIAS)

    return img







# Data Pre-procesing and Augmentation (Experiment on your own)

random_transforms = [transforms.ColorJitter(), transforms.RandomRotation(degrees=20)]



transform = transforms.Compose([

                                transforms.CenterCrop(64),

                                transforms.RandomHorizontalFlip(p=0.5),

                                transforms.RandomApply(random_transforms, p=0.3),

                                transforms.ToTensor(),

                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



# The dataset (example)

dataset = torchvision.datasets.ImageFolder(

    '../input/all-dogs/',

    loader=ImageLoader, # THE CUSTOM LOADER

    transform=transform

)
_, axes = plt.subplots(figsize=(32, 32), ncols=8, nrows=8)

for i, ax in enumerate(axes.flatten()):

    ax.imshow(dataset[i][0].permute(1, 2, 0).detach().numpy())

plt.show()