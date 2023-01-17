import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

%matplotlib inline
in_dir = '/kaggle/input/blood-cells/dataset2-master/dataset2-master/images'
from tensorflow.keras.preprocessing.image import load_img, img_to_array



imgs = []

labels = []

train_dir = os.path.join(in_dir, "TRAIN")

for cell in os.listdir(train_dir):

    for img in os.listdir(os.path.join(train_dir, cell)):

        img_path = os.path.join(train_dir, cell, img)

        # convert image to array and flatten it

        curr_img = img_to_array(load_img(img_path))

        curr_img = curr_img.flatten()

        imgs.append(curr_img)

        labels.append(cell)

# select random images to visualize

import random

random.seed(1)



# function to plot images in grid like fashion

def plot_gallery(images, title, h, w, n_row=3, n_col=6):

    plt.figure(figsize=(1.7 * n_col, 2.3 * n_row))

    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)

    rand_sample = random.sample(range(0,len(images)), k=18)

    for n,i in enumerate(rand_sample):

        plt.subplot(n_row, n_col, n + 1)

        plt.imshow(images[i].reshape((h,w)))

        plt.title(title[i], size=12)

        plt.xticks(())

        plt.yticks(())
# plot images

plot_gallery(imgs, labels, 512, 450)
from skimage.feature import daisy



def create_image_preview(img, title):

    plt.imshow(img, cmap=plt.cm.gray)

    plt.title(title)

    plt.xticks(())

    plt.yticks(())



#picking an image to test daisy features

test_image = imgs[1].reshape(240, 320)    



features, img_desc = daisy(test_image, step=16, radius=4, rings=3, histograms=5, orientations=8, visualize=True)



create_image_preview(test_image, "Plain image")

plt.show()

create_image_preview(img_desc, "DAISY image")

plt.show()
#taking apply_daisy function for notebook 04 at https://github.com/eclarson/MachineLearningNotebooks

def apply_daisy(image, shape):

    feat = daisy(image.reshape(shape), step=16, radius=4, rings=3, histograms=5, orientations=8, visualize=False)

    return feat.reshape((-1))
#for image in imgs:

    