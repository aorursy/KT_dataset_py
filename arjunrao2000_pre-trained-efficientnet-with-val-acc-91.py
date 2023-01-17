# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/cell_images/cell_images"))



# Any results you write to the current directory are saved as output.
input_dir = "../input/cell_images/cell_images"

uninfected_dir = "../input/cell_images/cell_images/Uninfected"

infected_dir = "../input/cell_images/cell_images/Parasitized"



import os

import cv2

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import json

from keras.models import Sequential, Model

from keras.layers import Dense, Flatten, Activation, Dropout, GlobalAveragePooling2D

from keras.preprocessing.image import ImageDataGenerator

from keras import optimizers, applications

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

from keras import backend as K
img_array = []

count = 0

for img in os.listdir(uninfected_dir):

    img_array.append(cv2.imread(os.path.join(uninfected_dir,img)))

    count+=1

    if(count == 5):

        break

        

count = 0



for img in os.listdir(infected_dir):

    img_array.append(cv2.imread(os.path.join(infected_dir,img)))

    count+=1

    if(count == 5):

        break
plt.figure(figsize=[10,10])

i = 0

for img_name in img_array:

    plt.subplot(2, 5,i+1)

    plt.imshow(img_name)

    if(i<5):

        plt.title("Uninfected")

    else:

        plt.title("Infected")

    i+=1

    
!pip install -U efficientnet

from keras import applications

from efficientnet import EfficientNetB3

from keras import callbacks

from keras.models import Sequential
img_array[0].shape
train_datagen = ImageDataGenerator(rescale = 1/255.,

                                  horizontal_flip = True,

                                  width_shift_range = 0.2,

                                  height_shift_range = 0.2,

                                  fill_mode = 'nearest',

                                  validation_split = 0.15,

                                  zoom_range = 0.3,

                                  rotation_range = 30)

val_datagen = ImageDataGenerator(rescale = 1/255.,

                                validation_split = 0.15)



train_generator = train_datagen.flow_from_directory(

    directory = input_dir,

    class_mode = "binary",

    batch_size = 64,

    target_size = (32,32),

    subset = "training"

    )



val_generator = val_datagen.flow_from_directory(

    directory = input_dir,

    class_mode = "binary",

    batch_size = 64,

    target_size = (32,32),

    subset = "validation"

    )
efficient_net = EfficientNetB3(

    weights='imagenet',

    input_shape=(32,32,3),

    include_top=False,

    pooling='max'

)

model = Sequential()

model.add(efficient_net)

model.add(Dense(units = 120, activation='relu'))

model.add(Dense(units = 120, activation = 'relu'))

model.add(Dense(units = 1, activation='sigmoid'))

model.summary()
from keras.optimizers import Adam

model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit_generator(

    train_generator,

    epochs = 75,

    steps_per_epoch = 15,

    validation_data = val_generator,

    validation_steps = 7

)
acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1,len(acc) + 1)



plt.plot(epochs,acc,'bo',label = 'Training Accuracy')

plt.plot(epochs,val_acc,'b',label = 'Validation Accuracy')

plt.title('Training and Validation Accuracy')

plt.legend()

plt.figure()



plt.plot(epochs,loss,'bo',label = 'Training loss')

plt.plot(epochs,val_loss,'b',label = 'Validation Loss')

plt.title('Training and Validation Loss')

plt.legend()



plt.show()