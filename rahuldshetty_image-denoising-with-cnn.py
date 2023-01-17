# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from skimage import io

import cv2

import matplotlib.pyplot as plt



def display(img):

    img = img.astype('uint8')

    plt.imshow(img,cmap='gray')

    plt.show()

    

    

paths =['/kaggle/input/image-colorization/ab/ab/ab1.npy','/kaggle/input/image-colorization/ab/ab/ab2.npy','/kaggle/input/image-colorization/ab/ab/ab3.npy']



lsamples = np.load('/kaggle/input/image-colorization/l/gray_scale.npy')



images = np.zeros((25000,224,224,3),dtype='uint8')

i = 0

for path in paths:

    ab_img = np.load(path)

    images[i:i+ab_img.shape[0],:,:,0] = lsamples[i:i+ab_img.shape[0]]    

    images[i:i+ab_img.shape[0],:,:,1:] = ab_img

    i += ab_img.shape[0]

    

for i in range(25000):

    images[i] = cv2.cvtColor(images[i],cv2.COLOR_LAB2RGB)

    

display(images[54])

display(images[12524])

display(images[21000])

del ab_img

del lsamples
images = (images[:4000]).astype('float32')

images_noise = (images[:4000]).astype('float32')

import skimage

images = images/255

images_noise = images_noise/255



def add_noise(img,mode='gaussian'):

    return skimage.util.random_noise(img, mode=mode)



for i in range(images_noise.shape[0]):

    images_noise[i] = add_noise(images_noise[i])

    

display(images_noise[54]*255)
from tensorflow.keras import layers,models

from tensorflow.keras import *
model = Sequential()

model.add(layers.Conv2D(32,(3,3),(1,1),padding='same',input_shape=(None,None,3)))

model.add(layers.MaxPooling2D((2,2),2))



model.add(layers.Conv2D(64,(3,3),(2,2),padding='same'))

model.add(layers.ReLU())

model.add(layers.MaxPooling2D((2,2),2))





model.add(layers.Conv2D(128,(3,3),(1,1),padding='same'))

model.add(layers.ReLU())

model.add(layers.MaxPooling2D((2,2),2))



model.add(layers.Conv2D(256,(3,3),(1,1),padding='same'))

model.add(layers.ReLU())

model.add(layers.MaxPooling2D((2,2),2))



model.add(layers.Conv2DTranspose(256,(3,3),(2,2),padding='same'))

model.add(layers.ReLU())

model.add(layers.Conv2DTranspose(128,(3,3),(2,2),padding='same'))

model.add(layers.ReLU())

model.add(layers.Conv2DTranspose(64,(3,3),(2,2),padding='same'))

model.add(layers.ReLU())

model.add(layers.Conv2DTranspose(32,(3,3),(2,2),padding='same'))

model.add(layers.ReLU())

model.add(layers.Conv2DTranspose(3,(3,3),(2,2),padding='same'))

model.add(layers.ReLU())



model.summary()

model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
history = model.fit(images,images_noise,validation_split=0.15,epochs=400)
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
indexes = [4,1000,50,68,99,565]

for idx in indexes:

    display(images_noise[idx]*255)

    pred = (model.predict(np.array([images_noise[idx]])))[0] * 255

    display(pred) 