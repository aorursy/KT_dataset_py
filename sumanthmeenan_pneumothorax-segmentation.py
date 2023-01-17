import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import glob

from PIL import Image

import os

print(os.listdir("../input/"))
df = pd.read_csv('/kaggle/input/stage_2_train.csv')

df.head()
df['ImageId'][0]
print('Total number of images: ', len(os.listdir("../input/stage_2_images/")))

os.listdir("../input/stage_2_images/")[:3]
#We need to load dcm files

import pydicom

#Subplot

fig, axes = plt.subplots(2,2, figsize = (20,20))

images_list = [i for i in glob.glob('../input/stage_2_images/' + './*.dcm')]

for img,ax in zip(images_list,axes.flatten()):

    image = pydicom.read_file(img) #To read dicom images

    img2 = image.pixel_array        #numpy array 

    img1 = Image.fromarray(img2)    #numpy array to image

    ax.set_title(img.split('/')[-1])

    ax.imshow(img1, cmap = "bone")                #use cmap = "gray"
print(os.listdir("../input/siim-acr-pneumothorax-segmentation-data"))