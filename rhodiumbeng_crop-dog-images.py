import numpy as np

import pandas as pd

from PIL import Image

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))
import glob

image = glob.glob('../input/all-dogs/all-dogs/*')

breed = glob.glob('../input/annotation/Annotation/*')

annot = glob.glob('../input/annotation/Annotation/*/*')

print(len(image), len(breed), len(annot))
# Let's take a look at the content of an annotation file. I choose one with two dogs in the image, and

# there are two bounding boxes specified. 

!cat ../input/annotation/Annotation/n02097658-silky_terrier/n02097658_98
import xml.etree.ElementTree as ET
def get_bbox(annot):

    """

    This extracts and returns values of bounding boxes

    """

    xml = annot

    tree = ET.parse(xml)

    root = tree.getroot()

    objects = root.findall('object')

    bbox = []

    for o in objects:

        bndbox = o.find('bndbox')

        xmin = int(bndbox.find('xmin').text)

        ymin = int(bndbox.find('ymin').text)

        xmax = int(bndbox.find('xmax').text)

        ymax = int(bndbox.find('ymax').text)

        bbox.append((xmin,ymin,xmax,ymax))

    return bbox
# test

bbox = get_bbox('../input/annotation/Annotation/n02097658-silky_terrier/n02097658_98')

print(bbox[0], bbox[1], len(bbox))
def get_image(annot):

    """

    Retrieve the corresponding image given annotation file

    """

    img_path = '../input/all-dogs/all-dogs/'

    file = annot.split('/')

    img_filename = img_path+file[-1]+'.jpg'

    return img_filename
plt.figure(figsize=(10,10))

for i in range(12):

    plt.subplot(3,4,i+1)

    plt.axis("off")

    dog = get_image(annot[i])

    im = Image.open(dog)

    im = im.resize((64,64), Image.ANTIALIAS)

    plt.imshow(im)
plt.figure(figsize=(10,10))

for i in range(12):

    bbox = get_bbox(annot[i])

    dog = get_image(annot[i])

    im = Image.open(dog)

    for j in range(len(bbox)):

        im = im.crop(bbox[j])

        im = im.resize((64,64), Image.ANTIALIAS)

    plt.subplot(3,4,i+1)

    plt.axis("off")

    plt.imshow(im)
test = '../input/annotation/Annotation/n02097658-silky_terrier/n02097658_98'

box = get_bbox(test)

dog = get_image(test)

im = Image.open(dog)

plt.imshow(im)
plt.figure(figsize=(6,6))

for j in range(len(box)):

    im = Image.open(dog)

    im = im.crop(box[j])

    im = im.resize((64,64), Image.ANTIALIAS)

    plt.subplot(1,2, j+1)

    plt.axis("off")

    plt.imshow(im)