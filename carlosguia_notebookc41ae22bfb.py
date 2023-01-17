# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import random



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib.cm as cm



import tensorflow as tf
# read training data from CSV file 

data = pd.read_csv('../input/train.csv')



print('data({0[0]},{0[1]})'.format(data.shape))

print (data.head())
images = data.iloc[:,1:].values

images = images.astype(np.float)

images = np.multiply(images, 1 / 255.0)

print ('images({0[0]}, {0[1]})'.format(images.shape))



labels_flat = data[[0]].values.ravel()

image_size = images.shape[1]



image_width = image_height = np.sqrt(image_size).astype(np.uint8)

print ('image_size: {0} = {1}x{2}'.format(image_size, image_width, image_height))

def display(img):

    image = img.reshape(image_width, image_height)

    plt.axis('off')

    plt.imshow(image, cmap = cm.binary)



def display_with_label(index):

    display(images[index])

    print ('Label: {0}'.format(labels_flat[index]))

    
display_with_label(858)
labels_count = np.unique(labels_flat).shape[0]


def dense_to_one_hot(dense, classes):

    labels = dense.shape[0]

    offset = np.arange(labels) * classes

    one_hot = np.zeros((labels, classes))

    one_hot.flat[offset + dense.ravel()] = 1

    return one_hot



labels = dense_to_one_hot(labels_flat, labels_count).astype(np.uint8)

valid_p = 10

train_p = 100 - valid_p

assert 50 <= train_p <= 99, 'We want to keep train_p in a sensible range: %.2f%%' % (train_p)



samples = images.shape[0]



valid = int(samples * valid_p / 100.0)

train = samples - valid





np.random.seed(19820622)

p = np.random.permutation(samples)



validation_images = images[p[:valid]]

validation_labels = labels[p[:valid]]



train_images = images[p[valid:]]

train_labels = labels[p[valid:]]