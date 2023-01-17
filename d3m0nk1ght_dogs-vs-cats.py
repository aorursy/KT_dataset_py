# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.preprocessing.image import ImageDataGenerator

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/cats_dogs/CATS_DOGS/train"))



train_datagen=ImageDataGenerator(rotation_range=30,width_shift_range=0.1,height_shift_range=0.1,rescale=1/255,

                             shear_range=0.2,zoom_range=0.2

                            ,horizontal_flip=True,fill_mode='nearest')

train_datagen.flow_from_directory('../input/cats_dogs/CATS_DOGS/train')



test_datagen=ImageDataGenerator(rotation_range=30,width_shift_range=0.1,height_shift_range=0.1,rescale=1/255,

                             shear_range=0.2,zoom_range=0.2

                            ,horizontal_flip=True,fill_mode='nearest')

test_datagen.flow_from_directory('../input/cats_dogs/CATS_DOGS/test')    



# Any results you write to the current directory are saved as output.
from keras.models import Sequential

from keras.layers import Dense, Conv2D, Flatten,MaxPooling2D,Dropout

from keras import layers





model = Sequential()

model.add(layers.Conv2D(32, (3, 3), input_shape=(64,64,3)))

model.add(layers.Activation('relu'))

model.add(layers.MaxPooling2D(pool_size=(2, 2)))



model.add(layers.Conv2D(32, (3, 3)))

model.add(layers.Activation('relu'))

model.add(layers.MaxPooling2D(pool_size=(2, 2)))



model.add(layers.Conv2D(64, (3, 3)))

model.add(layers.Activation('relu'))

model.add(layers.MaxPooling2D(pool_size=(2, 2)))



model.add(layers.Flatten())

model.add(layers.Dense(64))

model.add(layers.Activation('relu'))

#model.add(layers.Dropout(0.5))

model.add(layers.Dense(1))

model.add(layers.Activation('sigmoid'))



model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])



model.summary()

#model.fit(Xtr, Ytr,batch_size=32, validation_data=(Xte, Yte), epochs=8)



     

        
training_set=train_datagen.flow_from_directory(directory="../input/cats_dogs/CATS_DOGS/train",target_size=(64,64),class_mode='binary')

testing_set=test_datagen.flow_from_directory(directory="../input/cats_dogs/CATS_DOGS/train",target_size=(64,64),class_mode='binary')



model.fit_generator(training_set,steps_per_epoch = 8000//32,epochs =20 ,validation_data = testing_set,validation_steps = 2000//32)