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


import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

import cv2

import tensorflow as tf

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

import keras

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout, Flatten

from keras.layers import Conv2D

from keras.layers import MaxPooling2D,MaxPool2D,AveragePooling2D

from keras.layers.normalization import BatchNormalization

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator
def get_labels(directory):

    Labels = []

    

    for labels in os.listdir(directory):

        Labels.append(labels)

        

    return Labels

    
allLabels = get_labels('../input/flowers-recognition/flowers/flowers')
allLabels
def get_images(directory):

    Images = []

    Labels = []

    

    for labels in os.listdir(directory):

        label = allLabels.index(labels)

        

        for image_file in os.listdir(directory+labels):

            image = cv2.imread(directory+labels+r'/'+image_file)

            if image is not None:

                image = cv2.resize(image,(120,120))

                

                Images.append(image)

                Labels.append(label)

            else:

                print(directory+labels+r'/'+image_file)

            

            

    return shuffle(Images,Labels)
Images,Labels = get_images('../input/flowers-recognition/flowers/flowers/')
bi = Images

bl = Labels

Images = bi

Labels = bl
Images = np.array(Images)

Labels = np.array(Labels)
Images.shape
Labels.shape
#convert images to grayscal

Images = np.sum(Images/3,axis=3,keepdims=True)
plt.imshow(Images[0].squeeze(),cmap='gray')
#normalize images

Images = Images/255
Images.min()
Images.max()
x_train,x_valid,y_train,y_valid = train_test_split(Images,Labels,test_size=0.2,random_state=42)
#x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,

#                                                test_size=0.2,random_state=42)
x_train.shape
train_datagen = ImageDataGenerator(

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    fill_mode='nearest'

)

train_datagen.fit(x_train)
#output = (input-filter +1)/stride

model = Sequential()

#1st convolutional layer

model.add(Conv2D(filters=64,kernel_size=(5,5),padding='same',activation='relu',input_shape=(150,150,3)))#(150-5+1)/1

model.add(MaxPool2D())#output = 73

model.add(BatchNormalization())#o = 73

model.add(Dropout(0.2))#o = 73



#2nd convolutional layer

model.add(Conv2D(filters=128,kernel_size=(3,3),padding='same',activation='relu'))#(73-3+1)/1

model.add(MaxPool2D(strides=(2,2)))#output = 35

model.add(BatchNormalization())#o = 35

model.add(Dropout(0.2))#o = 35



#3rd convolutional layer

model.add(Conv2D(filters=128,kernel_size=(3,3),padding='same',activation='relu'))#(35-3+1)/1

model.add(MaxPool2D(strides=(2,2)))#output = 16

model.add(BatchNormalization())#o = 16

model.add(Dropout(0.2))#o = 16

model.summary()



#4th convolutional layer

model.add(Conv2D(filters=128,kernel_size=(3,3),padding='same',activation='relu'))#(16-3+1)/1

model.add(MaxPool2D(strides=(2,2)))#output = 16

model.add(BatchNormalization())#o = 16

model.add(Dropout(0.2))#o = 16

model.summary()



#5th convolutional layer

model.add(Conv2D(filters=128,kernel_size=(3,3),padding='same',activation='relu'))#(16-3+1)/1

model.add(MaxPool2D(strides=(2,2)))#output = 16

model.add(BatchNormalization())#o = 16

model.add(Dropout(0.2))#o = 16



#6th convolutional layer

model.add(Conv2D(filters=128,kernel_size=(3,3),padding='same',activation='relu'))#(16-3+1)/1

model.add(MaxPool2D(strides=(2,2)))#output = 16

model.add(BatchNormalization())#o = 16

model.add(Dropout(0.2))#o = 16



model.add(Flatten())

# 1st Fully Connected Layer

model.add(Dense(1024,activation="relu"))

model.add(Dropout(0.5))

model.add(BatchNormalization())



model.add(Dense(1024,activation="relu"))

model.add(Dropout(0.5))

model.add(BatchNormalization())









# Add output layer

model.add(Dense(5,activation="softmax"))

model.summary()
model.compile(loss='sparse_categorical_crossentropy',

              optimizer=Adam(lr=0.0001),

              metrics=['accuracy'])
#history = model.fit(x_train,y_train,epochs=50,validation_data=(x_val,y_val))
2766/86
#historyd = model.fit_generator(train_datagen.flow(x_train,y_train,batch_size=32),epochs=100,validation_data=(x_val,y_val))
Images = bi

Labels = bl
Images = np.array(Images)

Labels = np.array(Labels)
Labels
Images = Images/255
Images.shape
Labels.shape
import keras

Labels = keras.utils.to_categorical(Labels,num_classes=5,dtype='int32')
x_train,x_test,y_train,y_test = train_test_split(Images,Labels,test_size=0.2,random_state=42)
train_datagen = ImageDataGenerator(

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    fill_mode='nearest'

)

train_datagen.fit(x_train)
x_train.shape
from keras.applications.vgg16 import VGG16

base_model = VGG16(weights='imagenet',include_top=False,input_shape=(120,120,3))

#base_model = MobileNetV2(input_shape=(150,150,3),weights='imagenet',include_top=False)
from keras.layers import GlobalAveragePooling2D

from keras.models import Model
base_model
base_model.summary()
x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(1024,activation='relu')(x)

x = Dropout(0.2)(x)

x = Dense(1024,activation='relu')(x)

x = Dropout(0.2)(x)

predictions = Dense(5,activation='softmax')(x)



model = Model(inputs=base_model.input,outputs=predictions)



for layer in base_model.layers:

    layer.trainable = False
model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
#model.fit(x_train,y_train,batch_size=32,epochs = 50,validation_data=(x_test,y_test))

historyd = model.fit_generator(train_datagen.flow(x_train,y_train,batch_size=32),epochs=20,validation_data=(x_test,y_test))
#save the weights:

model.save('my_model.h5') 



from IPython.display import FileLink

FileLink('my_model.h5')