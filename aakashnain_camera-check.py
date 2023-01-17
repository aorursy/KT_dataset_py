# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import os

import glob

from pathlib import Path

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.image as mimg

from skimage.io import imread, imshow, imsave

from PIL import Image

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
input_path = Path('../input')

train_path = input_path / 'train'

test_path = input_path / 'test'
# Get all the folders. Each folder name is based on the device name

cameras = os.listdir(train_path)



# Initialize an empty list

train_images = []



# Iterate over each file in each sub-folder

for camera in cameras:

    for fname in sorted(os.listdir(train_path / camera)):

        train_images.append((camera, '../input/train/' + camera + '/' + fname))

        

# Convert the list to a pandas dataframe and save it for future use

train = pd.DataFrame(data=train_images, columns=['camera', 'fname'])

print("Total number of training samples: ", train.shape[0])

train.to_csv('./train_data.csv', index=None)

# Random samples from the data

train.sample(10)
# Do the same for test images

test_images = []

for fname in sorted(os.listdir(test_path)):

    test_images.append('../input/test/' + fname)



test = pd.DataFrame(test_images, columns=['fname'])

print("Number of test samples: ", test.shape[0])

test.to_csv('./test_data.csv', index=None)

test.head(10)
# Let's look at some samples from the training data first

f,ax = plt.subplots(2,5, figsize=(15,5))

for i in range(10):

    img = imread(train['fname'][i])

    ax[i//5, i%5].imshow(img)

    ax[i//5, i%5].axis('off')

plt.show()    
# Let's look at some samples from the test data 

f,ax = plt.subplots(2,5, figsize=(15,10))

for i in range(10):

    # Use PIL to read the tiff file

    img = Image.open(test["fname"][i])

    # Convert it into a numpy array

    img = np.array(img)

    ax[i//5, i%5].imshow(img)

    ax[i//5, i%5].axis('off')

plt.show()