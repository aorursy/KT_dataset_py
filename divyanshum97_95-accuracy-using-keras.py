# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/cell_images/cell_images"))

train_dir="../input/cell_images/cell_images"



# Any results you write to the current directory are saved as output.
import keras

from keras.layers import Conv2D,Flatten,Dropout,Dense,MaxPooling2D,SeparableConv2D

from keras.models import Sequential

from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

model=Sequential()

model.add(Conv2D(32,kernel_size=(3,3),input_shape=(128,128,3),activation='relu'))

model.add(Conv2D(32,kernel_size=3,activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))

model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))



model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))

model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Flatten())

model.add(Dropout(0.5))

model.add(Dense(512,activation='relu'))

model.add(Dense(1,activation='sigmoid'))

model.summary()

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

train_datagen=ImageDataGenerator(rescale=1./255,validation_split=0.2)



train_generator=train_datagen.flow_from_directory(train_dir,

                                               target_size=(128,128),

                                               batch_size=32,

                                               class_mode='binary',

                                               subset='training'

                                               )



validation_generator=train_datagen.flow_from_directory(train_dir,

                                               target_size=(128,128),

                                               batch_size=32,

                                               class_mode='binary',

                                               subset='validation'

                                               )

"""

callbacks=[keras.callbacks.ReduceLROnPlateau(

monitor='val_loss',

factor=0.1,

patience=10)]"""







history=model.fit_generator(train_generator,

                           steps_per_epoch=345,

                           epochs=5,

                           validation_data=validation_generator,

                           validation_steps=86)

                           #callbacks=callbacks)

hist=history.history

hist.keys()

val_acc=hist['val_acc']

val_loss=hist['val_loss']

acc=hist['acc']

loss=hist['loss']

epochs=range(1,len(acc)+1)

plt.plot(epochs,acc,'bo',label='Training_Accuracy')

plt.plot(epochs,val_acc,'b',label='Validation_Accuracy')

plt.figure()

plt.show()

plt.plot(epochs,loss,'bo',label='Training_Loss')

plt.plot(epochs,val_loss,'b',label='Validation_Loss')

plt.show()
