# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Nothing remarkable about this fork except that it is the first one I've ever done
from PIL import Image

import numpy as np

import pandas as pd

import os

import cv2

import tensorflow

from tensorflow import keras

from tensorflow.keras.datasets import mnist

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten

from tensorflow.keras.optimizers import RMSprop



import matplotlib.pyplot as plt

%matplotlib inline
# https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.load.html

# https://stackoverflow.com/questions/44289302/how-do-i-view-data-object-contents-within-an-npz-file

# https://stackoverflow.com/questions/32682928/loading-arrays-from-npz-files-in-pythhon

# https://stackoverflow.com/questions/48429408/open-and-view-npz-file-in-python



train_imgs = np.load('../input/k49-train-imgs.npz')['arr_0']

test_imgs = np.load('../input/k49-test-imgs.npz')['arr_0']

train_labels = np.load('../input/k49-train-labels.npz')['arr_0']

test_labels = np.load('../input/k49-test-labels.npz')['arr_0']
len(train_imgs)
type(train_imgs)
train_imgs_len = len(train_imgs)

test_imgs_len = len(test_imgs)
train_imgs_x = train_imgs.reshape(train_imgs_len, 784) # 60000 items, flattened from 28x28 to linear 784

test_imgs_x = test_imgs.reshape(test_imgs_len, 784)



# normalize data (starts as 0-255 integer)

# step 1 converts them to float

train_imgs_x = train_imgs_x.astype('float32')

test_imgs_x = test_imgs_x.astype('float32')



#step 2 converts them to range 0-1

train_imgs_x /= 255

test_imgs_x /= 255
num_classes=len(np.unique(train_labels))

print(num_classes)
train_labels_y = tensorflow.keras.utils.to_categorical(train_labels, num_classes)

test_labels_y = tensorflow.keras.utils.to_categorical(test_labels, num_classes) # converts to one hot
def display_sample(num):

    #Print the one-hot array of this sample's label 

    print(train_labels_y[num])  

    #Print the label converted back to a number

    label = train_labels_y[num].argmax(axis=0)

    #Reshape the 768 values to a 28x28 image

    image = train_imgs_x[num].reshape([28,28])

    plt.title('Sample: %d  Label: %d' % (num, label))

    plt.imshow(image, cmap=plt.get_cmap('gray_r'))

    plt.show()

    

display_sample(1234)
print("train_images: {}".format(train_imgs.shape))

print("train_label: {}".format(train_labels.shape))

print("test_data: {}".format(test_imgs.shape))

print("test_label: {}".format(test_labels.shape))
# Flat model

model = Sequential() # means you can add layers one at a time

model.add(Dense(512, activation='relu', input_shape=(784,))) # dense first hidden layer of 512 neurons

model.add(Dense(num_classes, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy',

              optimizer=RMSprop(),

              metrics=['accuracy'])
history = model.fit(train_imgs_x, train_labels_y,

                    batch_size=100,

                    epochs=10,

                    verbose=2,

                    validation_data=(test_imgs_x, test_labels_y))
#2D model

from tensorflow.keras import backend as K



if K.image_data_format() == 'channels_first':

    train_imgs_2Dx = train_imgs.reshape(train_imgs.shape[0], 1, 28, 28)

    test_imgs_2Dx = test_imgs.reshape(test_imgs.shape[0], 1, 28, 28)

    input_shape = (1, 28, 28)

else:

    train_imgs_2Dx = train_imgs.reshape(train_imgs.shape[0], 28, 28, 1)

    test_imgs_2Dx = test_imgs.reshape(test_imgs.shape[0], 28, 28, 1)

    input_shape = (28, 28, 1)

    

train_imgs_2Dx = train_imgs_2Dx.astype('float32') # data originally 8-bit bytes (256 bits?)?

test_imgs_2Dx = test_imgs_2Dx.astype('float32')

train_imgs_2Dx /= 255 # to transform images data to a value between 0 and 1

test_imgs_2Dx /= 255
train_labels_2Dy = tensorflow.keras.utils.to_categorical(train_labels, num_classes)

test_labels_2Dy = tensorflow.keras.utils.to_categorical(test_labels, num_classes) # converts to one hot
model = Sequential() # allows model to be built in layers

model.add(Conv2D(32, kernel_size=(3, 3), # 32 regional fields that model will usw to sample the image

                 activation='relu',

                 input_shape=input_shape)) # input shape defined in line 6/10

# 64 3x3 kernels

model.add(Conv2D(64, (3, 3), activation='relu'))

# Reduce by taking the max of each 2x2 block

model.add(MaxPooling2D(pool_size=(2, 2))) # max pixel found (removes blurry ones to save space?)

# Dropout to avoid overfitting

model.add(Dropout(0.25))

# Flatten the results to one dimension for passing into our final layer

model.add(Flatten())

# A hidden layer to learn with

model.add(Dense(128, activation='relu'))

# Another dropout

model.add(Dropout(0.5))

# Final categorization from 0-9 with softmax

model.add(Dense(num_classes, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
history = model.fit(train_imgs_2Dx, train_labels_2Dy,

                    batch_size=100,

                    epochs=10,

                    verbose=2, # right for jupyter

                    validation_data=(test_imgs_2Dx, test_labels_2Dy))