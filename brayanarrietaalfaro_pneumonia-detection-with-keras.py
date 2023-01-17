# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Linear algebra

import numpy as np

# Data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd 

import os

# Keras

from keras.preprocessing.image import ImageDataGenerator, load_img

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense

from keras import backend as K

# Input data files are available in the "../input/" directory.
# Directories

train_directory="../input/stage_2_train_images/"

test_directory="../input/stage_2_test_images/"

validate_samples = 2560
filenames = os.listdir(train_directory)

print(len(filenames))

train_filenames= filenames[validate_samples:]

validate_filenames= filenames[:validate_samples]

train_filenames_lenght=len(train_filenames)

validate_filenames_lenght=len(validate_filenames)
train_df = pd.read_csv('../input/stage_2_train_labels.csv')

train_df.head()
pneumonia_locations=train_df[train_df.Target != 0]

# Sort by patientId

pneumonia_locations.groupby(['patientId'], sort=False)

pneumonia_locations.head()
any(train_df['patientId'].duplicated())