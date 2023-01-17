import os
import numpy as np
np.random.seed(123)
import pandas as pd

from glob import glob
import matplotlib.pyplot as plt
%matplotlib inline

import keras.backend as K

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, \
                         Flatten, Convolution2D, MaxPooling2D, \
                         BatchNormalization, UpSampling2D
from keras.utils import np_utils

from skimage.io import imread
from sklearn.model_selection import train_test_split
# set channels first notation
K.set_image_dim_ordering('th')
# helper function to load the image and downsample it by 2
jimread = lambda x: np.expand_dims(imread(x)[::2, ::2],0)
BASE_IMAGE_PATH = os.path.join('..', 'input')
all_images = glob(os.path.join(BASE_IMAGE_PATH, '2d_images', '*.tif'))
all_masks = ['_masks'.join(c_file.split('_images')) for c_file in all_images]
print(len(all_masks), 'matching files found')
%matplotlib inline
test_image = jimread(all_images[0])
test_mask = jimread(all_masks[0])
fig, (ax1 ,ax2) = plt.subplots(1, 2)
ax1.imshow(test_image[0])
ax2.imshow(test_mask[0])
print('Total samples are', len(all_images))
print('Image resolution is', test_image.shape)
images = np.stack([jimread(i) for i in all_images], 0)
masks = np.stack([jimread(i) for i in all_masks], 0) / 255.0
X_train, X_test, y_train,  y_test = train_test_split(images, masks, test_size=0.1)
print('Training input is', X_train.shape)
print('Training output is {}, min is {}, max is {}'.format(y_train.shape, y_train.min(), y_train.max()))
print('Testing set is', X_test.shape)
# Create a deep nn
model = Sequential()
model.add(Convolution2D(filters=32, 
                        kernel_size=(3, 3), 
                        activation='relu', 
                        input_shape=images.shape[1:],
                        padding='same'
                        ))
model.add(Convolution2D(filters=64, 
                        kernel_size=(3, 3), 
                        activation='sigmoid', 
                        input_shape=images.shape[1:],
                        padding='same'
                        ))
model.add(Convolution2D(filters=128, 
                        kernel_size=(3, 3), 
                        activation='sigmoid', 
                        input_shape=images.shape[1:],
                        padding='same'
                        ))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dense(128, activation='relu'))
model.add(Convolution2D(filters=1, 
                        kernel_size=(3, 3), 
                        activation='sigmoid', 
                        input_shape=images.shape[1:],
                        padding='same'
                        ))
model.add(UpSampling2D(size=(2,2)))

model.compile(loss='binary_crossentropy',
             optimizer='rmsprop',
             metrics=['accuracy','mse'])
print(model.summary())
history = model.fit(X_train, y_train, validation_split=0.10, epochs=100, batch_size=30)
history.history.keys()
fig, ax = plt.subplots(1,2)
ax[0].plot(history.history['acc'], 'b')
ax[0].set_title('Accuraccy')
ax[1].plot(history.history['loss'], 'r')
ax[1].set_title('Loss')
fig, (ax1, ax2, ax3) = plt.subplots(1,3)
ax1.imshow(X_test[2,0])
ax2.imshow(y_test[2,0])
ax3.imshow(model.predict(X_test)[2,0])
