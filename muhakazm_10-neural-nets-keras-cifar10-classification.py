# installing tensorflow

!pip install -U tensorflow==2.0.0-alpha0
from numpy.random import seed

seed(888) # we do this for preventing different results as we're gonna be using so many random numbers

import tensorflow

#from tensorflow import set_random_seed

#set_random_seed(404)
import os

import numpy as np

import tensorflow as tf

import keras



# TO download the data set, go to: cs.toronto.edu/~kriz/cfar.html

# But thanks to keras, we just need to import the whole bunch of things.

from keras.datasets import cifar10



from IPython.display import display

from keras.preprocessing.image import array_to_img



import matplotlib.pyplot as plt



%matplotlib inline
LABEL_NAMES = ['Plane','Car','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']
(x_train_all,y_train_all),(x_test, y_test) = cifar10.load_data()
# wanna go further

print(type(cifar10))

print(type(x_train_all))
# IPython for displaying image

x_train_all[0]
# see the actual immage

pic = array_to_img(x_train_all[7])

display(pic)
y_train_all.shape
# 7 itu maksudnya horse

print(y_train_all[7][0])

print(LABEL_NAMES[y_train_all[7][0]])
# Kalau ini pakai matplotlib.pyplot

plt.imshow(x_train_all[4])

plt.xlabel(LABEL_NAMES[y_train_all[4][0]],fontsize=15)

plt.show()
plt.figure(figsize=(15,5))

for i in range(10):

    # create subplot; requires 3 inputs: rows, columns, i+1

    plt.subplot(1,10,i+1)

    plt.yticks([]) # to remove apa gitu

    plt.xticks([]) # to remove apa gitu

    plt.xlabel([LABEL_NAMES[y_train_all[i][0]]],fontsize=15)

    plt.imshow(x_train_all[i])