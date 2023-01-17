# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from IPython.display import display, Image

from matplotlib.pyplot import imshow

from keras.layers import Conv2D, UpSampling2D, InputLayer

from keras.models import Sequential

from keras.preprocessing.image import img_to_array, load_img

from skimage.color import lab2rgb, rgb2lab





import os

print(os.listdir("../input/urban-and-rural-photos/rural_and_urban_photos/train/"))
INPUT_IMAGE_SRC = '../input/urban-and-rural-photos/rural_and_urban_photos/train/urban/urban_11.jpeg'

display(Image(INPUT_IMAGE_SRC, width=225))
#Converting RGB to LAB

image = img_to_array(load_img(INPUT_IMAGE_SRC, target_size=(200,200))) / 255

lab_image = rgb2lab(image)

lab_image.shape
lab_image_norm = (lab_image + [0,128,128]) / [100,255,255]


#Input ---> Black and White Layer

X = lab_image_norm[:,:,0]



#output ---> ab channels

Y = lab_image_norm[:,:,1:]
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



#Finish model

model.compile(optimizer = 'rmsprop', loss = "mse")

model.fit(x=X, y=Y, batch_size = 1, epochs = 2000, verbose = 1)
model.evaluate(X,Y, batch_size = 1)
output = model.predict(X)

cur = np.zeros((200, 200, 3))

cur[:,:,0] = X[0][:,:,0]

cur[:,:,1:] = output[0]





cur = (cur * [100, 255, 255]) - [0, 128, 128]

rgb_image = lab2rgb(cur)

imshow(rgb_image)

INPUT_IMAGE_SRC_PRE = '../input/urban-and-rural-photos/rural_and_urban_photos/train/urban/urban_1.jpeg'

display(Image(INPUT_IMAGE_SRC_PRE, width=225))
imagepre = img_to_array(load_img(INPUT_IMAGE_SRC_PRE, target_size=(200,200))) / 255

lab_image_pre = rgb2lab(imagepre)

lab_image_norm_pre = (lab_image_pre + [0,128,128]) / [100,255,255]

X = lab_image_norm[:,:,0]

X = X.reshape(1, X.shape[0], X.shape[1], 1)
output = model.predict(X)

cur = np.zeros((200, 200, 3))

cur[:,:,0] = X[0][:,:,0]

cur[:,:,1:] = output[0]





cur = (cur * [100, 255, 255]) - [0, 128, 128]

rgb_image = lab2rgb(cur)

imshow(rgb_image)
