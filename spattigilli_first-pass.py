# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import os

import numpy as np

import random 

import math

from PIL import Image



batch_size = 128

size = 256

train_dir = "../input/train/"

test_dir = "../input/test_stg1"



sub_dirs = os.listdir(train_dir)

sub_dirs.remove(".DS_Store")



def train_set(data_images):

    dataset = np.zeros(shape=(len(data_images),size,size,3), dtype= int)

    labels = np.zeros(shape=(len(data_images),len(sub_dirs)), dtype= int)

    num_images = 0;

    for image in data_images:

        img = Image.open(image[0]).resize((size,size))    

        dataset[num_images,:,:,:] = np.asarray(img)

        labels[num_images,:] = (np.arange(len(sub_dirs)) == sub_dirs.index(image[1])).astype(float)

        num_images += 1

    return dataset, labels



def test_set():

    dataset = np.zeros(shape=(len(test_images),size,size,3), dtype= float)

    num_images = 0;

    for image in test_images:

        img = Image.open(train_images[0][0]).resize((size,size))

        np.asarray(img)

        dataset[num_images,:,:,:] = np.asarray(img)

        num_images += 1

    return dataset



# Train Set

train_images = [(train_dir+"/"+sub_dir+"/"+image, sub_dir) for sub_dir in sub_dirs for image in os.listdir(train_dir+sub_dir)]

random.shuffle(train_images)

valid_offset = math.floor(len(train_images)*0.6)

test_offset = math.floor(len(train_images)*0.8)

print(valid_offset)

print(test_offset)

train_dataset, train_labels = train_set(train_images[:valid_offset-1])

valid_dataset, valid_labels = train_set(train_images[valid_offset:test_offset-1])

test_dataset, test_labels = train_set(train_images[test_offset:])

print(valid_dataset.shape)

print(test_dataset.shape)

print(valid_labels.shape)

print(test_labels.shape)



print(train_dataset.shape)