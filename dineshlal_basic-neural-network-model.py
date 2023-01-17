#Import some basic modules

import numpy as np

import pandas as pd
# Import training data

train = pd.read_csv('../input/train.csv')



print (train.shape)
#Convert dataframe to numpy array, as Neural network will need that as input datatype

train1 = train.iloc[:,1:].values

train1 = train1.astype(np.float)



# divide by 255 to convert pixel values between 0 and 1

train1 = np.multiply(train1, 1.0 / 255.0)



print (train1.shape)
no_of_pixels = train1.shape[1]

print ('no_of_pixels = {0}\n'.format(no_of_pixels))



# Image height and image width

image_width = np.sqrt(no_of_pixels).astype(np.uint8)

image_height = np.sqrt(no_of_pixels).astype(np.uint8)



print ('image_width => {0}\nimage_height => {1}'.format(image_width,image_height))
%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib.cm as cm



# Plot Image

plt.subplot(221)

plt.imshow(train1[5].reshape(image_width,image_height), cmap=cm.binary)

plt.subplot(222)

plt.imshow(train1[6].reshape(image_width,image_height), cmap=cm.binary)

plt.subplot(223)

plt.imshow(train1[7].reshape(image_width,image_height), cmap=cm.binary)

plt.subplot(224)

plt.imshow(train1[8].reshape(image_width,image_height), cmap=cm.binary)
#Retrieve labels form input data

labels = train[[0]].values.ravel()



print('labels({0})'.format(len(labels)))
#Calculate distinct labels in the data

distinct_labels = np.unique(labels).shape[0]



print('distinct_labels => {0}'.format(distinct_labels))
#Convert Label to binary matrix using using one hot encoding

from keras.utils import np_utils

y = np_utils.to_categorical(labels)

print('labels({0[0]},{0[1]})'.format(y.shape))
#Split into train and test

from sklearn.cross_validation import train_test_split



x_train, x_test, y_train, y_test = train_test_split(train1, y, test_size=0.05, random_state=42)



print('x_train({0[0]},{0[1]})'.format(x_train.shape))

print('x_test({0[0]},{0[1]})'.format(x_test.shape))

print('y_train({0[0]},{0[1]})'.format(y_train.shape))

print('y_test({0[0]},{0[1]})'.format(y_test.shape))
#Import other module requirements

import numpy

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.utils import np_utils
# fix random seed for reproducibility

seed = 7

numpy.random.seed(seed)