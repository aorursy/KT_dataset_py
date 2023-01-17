import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras as ks # neural network models



# For working with images

import cv2 as cv

import matplotlib.image as mpimg

import tqdm



# Potentially useful tools - you do not have to use these

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical



import os



# Input data files are available in the "../input/" directory.

# Any results you write to the current directory are saved as output.
# CONSTANTS

# You may not need all of these, and you may find it useful to set some extras



CATEGORIES = ['airplane','car','cat','dog','flower','fruit','motorbike','person']



IMG_WIDTH = 100

IMG_HEIGHT = 100

TRAIN_PATH = '../input/natural_images/natural_images/'

TEST_PATH = '../input/evaluate/evaluate/'
# To find data:

folders = os.listdir(TRAIN_PATH)



images = []



for folder in folders:

    files = os.listdir(TRAIN_PATH + folder)

    images += [(folder, file, folder + '/' + file) for file in files]



image_locs = pd.DataFrame(images, columns=('class','filename','file_loc'))



# data structure is three-column table

# first column is class, second column is filename, third column is image address relative to TRAIN_PATH

image_locs.head()
# Your code here
# Example values:

filenames = ['test001','test002','test003','test004']

predictions = ['car','cat','fruit','motorbike']
# Save results



# results go in dataframe: first column is image filename, second column is category name

# category names are: airplane, car, cat, dog, flower, fruit, motorbike, person

df = pd.DataFrame()

df['filename'] = filenames

df['label'] = predictions

df = df.sort_values(by='filename')



df.to_csv('results.csv', header=True, index=False)