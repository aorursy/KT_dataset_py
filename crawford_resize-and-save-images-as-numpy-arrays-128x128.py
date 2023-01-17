import cv2

import os

import random

import matplotlib.pylab as plt

from glob import glob

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split
# ../input/

PATH = os.path.abspath(os.path.join('..', 'input'))



# ../input/sample/images/

SOURCE_IMAGES = os.path.join(PATH, "sample", "images")



# ../input/sample/images/*.png

images = glob(os.path.join(SOURCE_IMAGES, "*.png"))



# Load labels

labels = pd.read_csv('../input/sample_labels.csv')
# First five images paths

images[0:5]
r = random.sample(images, 3)

r



# Matplotlib black magic

plt.figure(figsize=(16,16))

plt.subplot(131)

plt.imshow(cv2.imread(r[0]))



plt.subplot(132)

plt.imshow(cv2.imread(r[1]))



plt.subplot(133)

plt.imshow(cv2.imread(r[2]));    
# Example of bad x-ray and good reason to use data augmentation

e = cv2.imread(os.path.join(SOURCE_IMAGES,'00030209_008.png'))



plt.imshow(e)



labels[labels["Image Index"] == '00030209_008.png']
def proc_images():

    """

    Returns two arrays: 

        x is an array of resized images

        y is an array of labels

    """

    

    disease="Infiltration"



    x = [] # images as arrays

    y = [] # labels Infiltration or Not_infiltration

    WIDTH = 128

    HEIGHT = 128



    for img in images:

        base = os.path.basename(img)

        finding = labels["Finding Labels"][labels["Image Index"] == base].values[0]



        # Read and resize image

        full_size_image = cv2.imread(img)

        x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))



        # Labels

        if disease in finding:

            #finding = str(disease)

            finding = 1

            y.append(finding)



        else:

            #finding = "Not_" + str(disease)

            finding = 0

            y.append(finding)



    return x,y
x,y = proc_images()
# Set it up as a dataframe if you like

df = pd.DataFrame()

df["labels"]=y

df["images"]=x
print(len(df), df.images[0].shape)
np.savez("x_images_arrays", x)

np.savez("y_infiltration_labels", y)
!ls -1