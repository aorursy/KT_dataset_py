import matplotlib.pyplot as plt

import matplotlib.image as mpimg

TRAIN_DIR = "../input/chest_xray/chest_xray/train/"

TEST_DIR =  "../input/chest_xray/chest_xray/test/"



plt.imshow(mpimg.imread('../input/chest_xray/chest_xray/test/PNEUMONIA/person78_bacteria_382.jpeg'))

import matplotlib.pyplot as plt

import cv2                 

from random import shuffle

from tqdm import tqdm  

import scipy

import skimage

from skimage.transform import resize

TRAIN_DIR = "../input/chest_xray/chest_xray/train/"

TEST_DIR =  "../input/chest_xray/chest_xray/test/"

Pimages = os.listdir(TRAIN_DIR + "PNEUMONIA")

Nimages = os.listdir(TRAIN_DIR + "NORMAL")

def plotter(i):

    imagep1 = cv2.imread(TRAIN_DIR+"PNEUMONIA/"+Pimages[i])

    imagep1 = skimage.transform.resize(imagep1, (150, 150, 3) , mode = 'reflect')

    imagen1 = cv2.imread(TRAIN_DIR+"NORMAL/"+Nimages[i])

    imagen1 = skimage.transform.resize(imagen1, (150, 150, 3))

    pair = np.concatenate((imagen1, imagep1), axis=1)

    print("(Left)-NORMAL ----- (Right)-PNEUMONIA")

    plt.figure(figsize=(10,5))

    plt.imshow(pair)

    plt.show()

for i in range(0,5):

    plotter(i)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import keras

from keras.models import Sequential

import cv2                 

from random import shuffle

from tqdm import tqdm  

import scipy

import skimage

from skimage.transform import resize

from keras.layers import Conv2D,MaxPool2D,Dense,Dropout,Softmax,Input,Flatten

from keras.optimizers import Adam,RMSprop,SGD

from keras.layers.merge import add

from keras.layers import Dense, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from keras.layers import BatchNormalization

# Any results you write to the current directory are saved as output.

 #Initializing the CNN



model = Sequential()

model.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3),activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(32,(3,3),activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), activation="relu", padding="same"))

model.add(Conv2D(filters=64, kernel_size=(3,3), activation="relu", padding="same"))



model.add(Dropout(rate=0.25))

model.add(Flatten())

model.add(Dense(units=512,activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(rate=0.4))

model.add(Dense(units=1,activation='sigmoid'))

#model.add(Dense(2, activation="softmax"))



model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])



from keras.preprocessing.image import ImageDataGenerator

train_model=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

test_model=ImageDataGenerator(rescale=1./255)

train_set = train_model.flow_from_directory('../input/chest_xray/chest_xray/train',target_size=(64,64), batch_size=32, class_mode='binary')

validation_generator = test_model.flow_from_directory('../input/chest_xray/chest_xray/val', target_size=(64, 64), batch_size=32,

                                                        class_mode='binary')

test_set = test_model.flow_from_directory('../input/chest_xray/chest_xray/test',target_size=(64,64), batch_size=32, class_mode='binary')

model.summary()

model.fit_generator(train_set, steps_per_epoch=5216/32, epochs=8, validation_data = validation_generator, validation_steps=624/32)   
print(os.listdir("../input/chest_xray/chest_xray"))

print(os.listdir("../input/chest_xray/chest_xray/train/"))