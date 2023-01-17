# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from IPython.display import display, Image
from matplotlib.pyplot import imshow
from keras.layers import Conv2D, UpSampling2D, InputLayer
from keras.models import Sequential
from keras.preprocessing.image import img_to_array, load_img
from skimage.color import lab2rgb, rgb2lab

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/rural_and_urban_photos/train/urban"))



# Any results you write to the current directory are saved as output.

INPUT_IMAGE_SRC = '../input/rural_and_urban_photos/train/urban/urban_11.jpeg'
display(Image(INPUT_IMAGE_SRC, width=225))
image = img_to_array(load_img(INPUT_IMAGE_SRC, target_size=(200,200))) / 255
lab_image = rgb2lab(image)
lab_image.shape
lab_image_norm = (lab_image + [0, 128, 128]) / [100, 255, 255]
# The input will be the black and white layer
X = lab_image_norm[:,:,0]
# The outpts will be the ab channels
Y = lab_image_norm[:,:,1:]

# The Conv2D layer we will use later expects the inputs and training outputs to be of the following format:
# (samples, rows, cols, channels), so we need to do some reshaping
# https://keras.io/layers/convolutional/
X = X.reshape(1, X.shape[0], X.shape[1], 1)
Y = Y.reshape(1, Y.shape[0], Y.shape[1], 2)
model = Sequential()
model.add(InputLayer(input_shape=(None, None, 1)))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(2, (3,3), activation='tanh', padding='same'))

# Finish model
model.compile(optimizer='rmsprop', loss='mse')
model.fit(x=X, y=Y, batch_size=1, epochs=1000, verbose=0)
model.evaluate(X, Y, batch_size=1)
output = model.predict(X)
cur = np.zeros((200, 200, 3))
cur[:,:,0] = X[0][:,:,0]
cur[:,:,1:] = output[0]


#imshow(rgb_image.astype('float32'))

#cur = (cur * [100, 255, 255]) - [0 ,128, 128]
#output

#rgb_image *= [100, 255, 255]
cur = (cur * [100, 255, 255]) - [0, 128, 128]
rgb_image = lab2rgb(cur)
imshow(rgb_image)