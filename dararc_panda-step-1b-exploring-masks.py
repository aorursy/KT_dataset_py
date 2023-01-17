

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

import matplotlib

import matplotlib.pyplot as plt # plotting figures

from PIL import Image # open and display images

import cv2 #computer vision library

from tqdm.notebook import tqdm # progress bar, tqdm shorthand for progress in Arabic 

import skimage.io #image processing
# setting the main directory and loading the train CSV file

MAIN_DIR = '../input/prostate-cancer-grade-assessment'

train = pd.read_csv(os.path.join(MAIN_DIR, 'train.csv')).set_index('image_id')
#dropping the mislabelled case we identified in EDA, this will return an error if ran more than once

susp = train[(train.gleason_score == '4+3') & (train.isup_grade != 3)]

train = train.drop([susp.index[0]])
# Check we removed our suspicious case

print(train.shape)
# Set our directories for our resized images and ensure we have all 10616 images and 10516 masks

resize_dir = '../input/panda-resized-train-data-512x512/'

img_dir = resize_dir + 'train_images/train_images/'

mask_dir = img_dir.replace('images', 'label_masks')

print(len(os.listdir(img_dir)))

print(len(os.listdir(mask_dir)))


# there are 100 images with no masks we'll just create an image to display for the case where there are no masks

no_mask_array = np.zeros((512,512,3), dtype = 'uint8')

no_mask_array[:,:,2] = np.identity(512, dtype = 'uint8')*2



# creating a batch of ids to test

id_batch = train.index[0:10]



#creating a function to take an image id and display image or mask (if exists) as required

def id2array(id, type):

    if type == 'mask':

        if os.path.isfile(os.path.join(mask_dir + id + '_mask.png')) == True:

            array = skimage.io.imread(os.path.join(mask_dir + id + '_mask.png'))

        else:

            array = no_mask_array

    else:

        array = skimage.io.imread(os.path.join(img_dir + id + '.png'))

    return array
img_array_batch = [id2array(item, 'image') for item in id_batch]

fig, axs = plt.subplots(5, 2, figsize=(25,25))

for i in range(0,10):

    axs[(i//2), (i%2)].imshow(img_array_batch[i])

plt.show()
#let us now test visualise one of our masks

mask_test = id2array(train.index[8032], 'mask')



plt.figure()

plt.title("Mask with default cmap")

plt.imshow(mask_test[:,:,2], interpolation='nearest')

plt.show()



plt.figure()

plt.title("Mask with custom cmap")

# Optional: create a custom color map

cmap = matplotlib.colors.ListedColormap(['black', 'gray', 'green', 'yellow', 'orange', 'red'])

plt.imshow(mask_test[:,:,2], cmap=cmap, interpolation='nearest', vmin=0, vmax=5)

plt.show()
# we set up two colour maps as described above

cmap_rad = matplotlib.colors.ListedColormap(['black', 'gray', 'gray', 'yellow', 'orange', 'red'])

cmap_kar = matplotlib.colors.ListedColormap(['black', 'gray', 'purple'])





# this function will take 5 image ids and display an image and the related mask if there is one if not display the no mask image defined above

def plot5(ids):

    img_arrays = [id2array(item, 'image') for item in ids]

    mask_arrays = [id2array(item, 'mask') for item in ids]

    fig, axs = plt.subplots(5, 2, figsize=(15,25))

    for i in range(0,5):

        image_id = ids[i]

        data_provider = train.loc[image_id, 'data_provider']

        gleason_score = train.loc[image_id, 'gleason_score']

        axs[i, 0].imshow(img_arrays[i])

        mask_array = mask_arrays[i]

        if data_provider == 'karolinska':

            axs[i, 1].imshow(mask_array[:,:,2], cmap=cmap_kar, interpolation='nearest', vmin=0, vmax=2)

        else:

            axs[i, 1].imshow(mask_array[:,:,2], cmap=cmap_rad, interpolation='nearest', vmin=0, vmax=5)

        for j in range(0,2):

            axs[i,j].set_title(f"ID: {image_id}\nSource: {data_provider} Gleason: {gleason_score}")

    plt.show()

    
plot5(train.index[1103:1108])