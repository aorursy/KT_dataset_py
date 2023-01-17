import numpy as np

import pandas as pd

from PIL import Image

import matplotlib.pyplot as plt

import os

import xml.etree.ElementTree as ET

from numpy import random

import zipfile

print(os.listdir("../input"))
import glob

image = glob.glob('../input/all-dogs/all-dogs/*')

breed = glob.glob('../input/annotation/Annotation/*')

annot = glob.glob('../input/annotation/Annotation/*/*')

print(len(image), len(breed), len(annot))
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
def get_image(annot):

    """

    Retrieve the corresponding image given annotation file

    """

    img_path = '../input/all-dogs/all-dogs/'

    file = annot.split('/')

    img_filename = img_path+file[-1]+'.jpg'

    return img_filename
# initialize tensor for dog images

n_x = 64

n_c = 3

dogs = np.zeros((len(image), n_x, n_x, n_c))

print(dogs.shape)
for a in range(len(image)):

    bbox = get_bbox(annot[a])

    dog = get_image(annot[a])

    if dog == '../input/all-dogs/all-dogs/n02105855_2933.jpg':   # this jpg is not in the dataset

        continue

    im = Image.open(dog)

    im = im.crop(bbox[0])

    im = im.resize((64,64), Image.ANTIALIAS)

    dogs[a,:,:,:] = np.asarray(im) / 255.
# pick some images randomly from dogs and look at these

plt.figure(figsize=(15,8))

n_images = 60

select = random.randint(low=0,high=dogs.shape[0],size=n_images)

for i, index in enumerate(select):  

    plt.subplot(6, 10, i+1)

    plt.imshow(dogs[index])

    plt.axis('off')

plt.subplots_adjust(wspace=0.3, hspace=-0.1)

plt.show()
z = zipfile.PyZipFile('images.zip', mode='w')

for d in range(10000):

    dog_image = Image.fromarray((255*dogs[d]).astype('uint8').reshape((64,64,3)))

    f = str(d)+'.png'

    dog_image.save(f,'PNG')

    z.write(f)

    os.remove(f)

z.close()