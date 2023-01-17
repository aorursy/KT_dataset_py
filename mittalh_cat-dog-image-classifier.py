# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import keras 

       



import tensorflow as tf

import matplotlib.pyplot as plt

import numpy as np

import matplotlib.image as mpimg

import os

from keras.preprocessing.image import ImageDataGenerator

from keras.models import load_model

from keras.optimizers import RMSprop

from keras.preprocessing import image
os.getcwd()
train_datagen=ImageDataGenerator(rescale=1/255)

train_generator=train_datagen.flow_from_directory("/kaggle/working/../input/dogs-cats-images/dog vs cat/dataset/training_set",

                                                 target_size=(50,50),batch_size=100,class_mode='binary')

test_datagen=ImageDataGenerator(rescale=1/255)

test_generator=test_datagen.flow_from_directory("/kaggle/working/../input/dogs-cats-images/dog vs cat/dataset/test_set",target_size=(50,50),

                                               batch_size=100,class_mode='binary')
model=keras.models.Sequential([ keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(50,50,3)),

                                  keras.layers.MaxPooling2D(2,2),

                                  keras.layers.Conv2D(64,(3,3),activation='relu'),

                                  keras.layers.MaxPooling2D(2,2),

                                  keras.layers.Flatten(),

                                  keras.layers.Dense(128,activation='relu'),

                                  keras.layers.Dense(1,activation='sigmoid')

                                  

    

    

    

])
model.summary()
model.compile(loss='binary_crossentropy',optimizer=RMSprop(lr=0.001),metrics=['acc'])
his=model.fit_generator(train_generator,steps_per_epoch=80,epochs=1,validation_data=test_generator,validation_steps=20,verbose=1)
model.save("kaggle.h5")
plt.plot(his.history['acc'])

plt.plot(his.history['val_acc'])

plt.title("accuracy plot")

plt.legend(['training','validation'])
os.chdir(r'/kaggle/input/dogs-cats-images/dog vs cat/dataset/test_set/cats')

test_image=image.load_img("cat.4001.jpg",target_size=(50,50))

test_image=image.img_to_array(test_image)

test_image=np.expand_dims(test_image,axis=0)

pi=mpimg.imread("cat.4001.jpg")

plt.imshow(pi)

y=model.predict(test_image)

if(y==0):

    print("Below image is of a cat")

else:

    print("Below image is of a dog")