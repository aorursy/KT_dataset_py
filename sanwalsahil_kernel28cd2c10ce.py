# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    print(dirname)



# Any results you write to the current directory are saved as output.
import tensorflow as tf

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle

import cv2
def getAllLabels(loc):

    dictLabel = {}

    

    for labels in os.listdir(loc):

        #print(labels)

        

        name = pd.read_fwf(loc+labels)

        new_name = name.iloc[0,0]

        labels = labels.replace('.txt','')

        #print(type(name))

        #break

        dictLabel[labels] = new_name

        

        #break

    return dictLabel
allLabels = getAllLabels('../input/butterfly-dataset/leedsbutterfly/descriptions/')
allLabels
def getImages(loc):

    Labels = []

    Images = []

    

    for img in os.listdir(loc):

        #print(img[:3].replace('0',""))

        #break

        image = cv2.imread(loc+img)

        image = cv2.resize(image,(120,120))

        

        Images.append(image)

        if img[:3] == '010':

            Labels.append('10')

        else:

            Labels.append(img[:3].replace('0',""))

        

    return shuffle(Images,Labels)
Images,Labels = getImages('../input/butterfly-dataset/leedsbutterfly/images/')
Images = np.array(Images)

Labels = np.array(Labels)
Images.shape
Labels.shape
Images = Images/255
Images.max()
Images.min()
Images.dtype
Labels = Labels.astype(int)
Labels.dtype
type(Labels)

np.unique(Labels)
import keras

Labels = keras.utils.to_categorical(Labels,num_classes=11)
x_train,x_test,y_train,y_test = train_test_split(Images,Labels,test_size=.2)
x_train.shape
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    fill_mode='nearest'

)

train_datagen.fit(x_train)
from keras.applications.vgg16 import VGG16

base_model = VGG16(weights='imagenet',include_top=False,input_shape=(120,120,3))
from keras.layers import GlobalAveragePooling2D

from keras.models import Model

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout, Flatten

from keras.layers import Conv2D

from keras.layers import MaxPooling2D,MaxPool2D,AveragePooling2D

from keras.layers.normalization import BatchNormalization

from keras.optimizers import Adam
x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(1024,activation='relu')(x)

x = Dropout(0.2)(x)

x = Dense(1024,activation='relu')(x)

x = Dropout(0.2)(x)

predictions = Dense(11,activation='softmax')(x)



model = Model(inputs=base_model.input,outputs=predictions)



for layer in base_model.layers:

    layer.trainable = False
model.summary()
model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
historyd = model.fit_generator(train_datagen.flow(x_train,y_train,batch_size=32),

                               validation_data=(x_test,y_test),epochs=30)
model.save('my_model.h5') 



from IPython.display import FileLink

FileLink('my_model.h5')