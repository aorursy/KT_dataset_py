# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from keras.applications.vgg19 import VGG19

from keras.models import Sequential

from keras.utils import to_categorical

from keras.layers import Dense,Flatten

from keras.datasets import cifar10

import cv2

from keras import backend as K

import matplotlib.pyplot as plt

from keras.optimizers import SGD

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from os import listdir, makedirs

from os.path import join, exists, expanduser



cache_dir = expanduser(join('~', '.keras'))

if not exists(cache_dir):

    makedirs(cache_dir)

datasets_dir = join(cache_dir, 'datasets') # /cifar-10-batches-py

if not exists(datasets_dir):

    makedirs(datasets_dir)





!cp ../input/cifar-10-python.tar.gz ~/.keras/datasets/

!ln -s  ~/.keras/datasets/cifar-10-python.tar.gz ~/.keras/datasets/cifar-10-batches-py.tar.gz

!tar xzvf ~/.keras/datasets/cifar-10-python.tar.gz -C ~/.keras/datasets/

# this part means

#Inclusion of data set in working environment
(x_train,y_train),(x_test,y_test) = cifar10.load_data() # load the data 
numberOfClass = 2000

y_train = to_categorical(y_train,numberOfClass)# y train and y test transform to categorical type

y_test = to_categorical(y_test,numberOfClass)

input_shape = x_train.shape[1:]
#Visualize 

plt.imshow(x_train[3119].astype(np.uint8)) #3119 is random number 

plt.axis("off") # close the axis

plt.show()

#this img shape is 32,32,3
#increase dimension

def resize_img(img):

    numberOfImage = img.shape[0]

    new_array = np.zeros((numberOfImage,48,48,3))# first img shape is 32,32,3 but we want shape 48,48,3 

    for i in range(numberOfImage):

        new_array[i] = cv2.resize(img[i,:,:,:],(48,48))

    return new_array
x_train = resize_img(x_train)

x_test = resize_img(x_test)
plt.figure()

plt.imshow(x_train[3119].astype(np.uint8)) #3119 is random number 

plt.axis("off") # close the axis

plt.show()

# And this img shape is 48,48,3
#VGG19

vgg19 = VGG19(include_top = False, weights = "imagenet",input_shape = (48,48,3))
print(vgg19.summary())

# Son satır MaxPooling2D bundan sonra fuly con layers flatten dense olmalıydı .include_top = False bunları çıkartır.

#Transfer learningde genelde bunlar  çıkartılır



# Last line MaxPooling2D should now be fuly con layers flatten dense .include_top = False removes them.

#Transfer learning is usually removed
vgg19_layer_list =vgg19.layers

print(vgg19_layer_list)
model = Sequential()

for layer in vgg19_layer_list:

    model.add(layer)

print(model.summary())
#transfer learning

for layer in model.layers:

    layer.trainable = False
#fuly con layers

model.add(Flatten())

model.add(Dense(1024))

model.add(Dense(numberOfClass, activation="softmax"))

print(model.summary())
#compile part

model.compile( loss = "categorical_crossentropy", optimizer ="rmsprop", metrics =["accuracy"])
history = model.fit(x_train ,y_train, validation_split= 0.2 , epochs = 15, batch_size = 1000)
# model save

model.save_weights("example.h5")
#visualize

plt.plot(history.history["loss"], label = "train loss")

plt.plot(history.history["val_loss"], label = "val loss")

plt.legend()

plt.show()

plt.figure()

plt.plot(history.history["acc"], label = "train acc")

plt.plot(history.history["val_acc"], label = "val acc")

plt.legend()

plt.show()