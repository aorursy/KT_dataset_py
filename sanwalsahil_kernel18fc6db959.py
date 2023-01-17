# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

    #for filename in filenames:

        #print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import cv2

import pandas as pd

import matplotlib.pyplot as plt

import cv2

import tensorflow as tf
def getImageLabels(direc):

    Images = []

    Labels = []

    i = 0

    for filename in os.listdir(direc):

        # get label

        label = filename.split('.')[0]

        

        Labels.append(allLabels.index(label))

        

        img = cv2.imread(direc+filename)

        img = cv2.resize(img,(150,150))

        

        Images.append(img)

        i = i+1

        if i==5000:

            break

        

    return Labels,Images
imagePath = '../input/dogs-vs-cats/train/train/'

allLabels = ['cat','dog']

Labels,Images = getImageLabels(imagePath)
Images = np.array(Images)

Labels = np.array(Labels)
Images = Images/255
Images.max()
Images.min()
ImagesBck = Images

LabelsBck = Labels
Images = ImagesBck

Lables = LabelsBck
datagen = tf.keras.preprocessing.image.ImageDataGenerator(

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    fill_mode='nearest'

)

#datagen.fit(Images)
#Labels = tf.keras.utils.to_categorical(Labels)
from sklearn.model_selection import train_test_split

Image_train,Image_valid,Label_train,Label_valid = train_test_split(Images,Labels,test_size=.2)
Image_train.shape


datagen.fit(Image_train)
#datagen.fit(Image_label)
#datagen.fit(Image_train)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D,Dense,Dropout,Flatten,BatchNormalization




model = Sequential()



model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid')) # 2 because we have cat and dog classes

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
Label_valid.shape
#model.fit(Image_train,Label_train,epochs=50,validation_data=(Image_valid,Label_valid))
model.fit_generator(datagen.flow(Image_train, Label_train), epochs=50,validation_data=(Image_valid,Label_valid))
from keras.applications.vgg16 import VGG16

base_model = VGG16(weights='imagenet',include_top=False,input_shape=(150,150,3))
from keras.layers import GlobalAveragePooling2D

from keras.models import Model
from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout, Flatten

from keras.layers import Conv2D

from keras.layers import MaxPooling2D,MaxPool2D,AveragePooling2D

from keras.layers.normalization import BatchNormalization

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator
x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(1024,activation='relu')(x)

x = Dropout(0.2)(x)

x = Dense(1024,activation='relu')(x)

x = Dropout(0.2)(x)

predictions = Dense(1,activation='sigmoid')(x)



model = Model(inputs=base_model.input,outputs=predictions)



for layer in base_model.layers:

    layer.trainable = False
model.summary()
model.compile(Adam(lr=.0001), loss='binary_crossentropy', metrics=['accuracy'])
Labels.shape
historyd = model.fit_generator(datagen.flow(Image_train,Label_train,batch_size=32),epochs=20,validation_data=(Image_valid,Label_valid))
model.save('model.h5')
from IPython.display import FileLink

FileLink('model.h5')