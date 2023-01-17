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
from keras.models import Sequential

from keras.layers import Dense,Conv2D,Flatten,Dropout,Activation,MaxPooling2D

from keras.preprocessing.image import img_to_array,load_img,ImageDataGenerator

from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt

from glob import glob

from sklearn.datasets import load_files

train_path="/kaggle/input/fruits/fruits-360_dataset/fruits-360/Training/"

test_path="/kaggle/input/fruits/fruits-360_dataset/fruits-360/Test/"
img=load_img(train_path+"Grapefruit White/31_100.jpg")

plt.imshow(img)

plt.axis("off")

plt.show()

img_den_array=img_to_array(img)

print(img_den_array.shape)
classname=glob(train_path+"/*") ##type:List

len_classname=len(classname)

print(len_classname)
model=Sequential()

model.add(Conv2D(32,(3,3),input_shape=img_den_array.shape))

model.add(Activation("relu"))

model.add(MaxPooling2D())



model.add(Conv2D(32,(3,3)))

model.add(Activation("relu"))

model.add(MaxPooling2D())



model.add(Conv2D(64,(3,3)))

model.add(Activation("relu"))

model.add(MaxPooling2D())



model.add(Flatten())

model.add(Dense(1024))

model.add(Activation("relu"))

model.add(Dropout(0.5))



model.add(Dense(len_classname))

model.add(Activation("softmax"))



model.compile(loss="categorical_crossentropy",

             optimizer="rmsprop",

             metrics=["accuracy"])



## This area For Data Generator 

train_datagen=ImageDataGenerator(rescale=1./255,

                  shear_range=0.3,

                  horizontal_flip=True,

                  zoom_range=0.4)

test_datagen=ImageDataGenerator(rescale=1./255)







train_generator=train_datagen.flow_from_directory(train_path,

                                                  target_size=img_den_array.shape[:2],

                                                  batch_size=32,

                                                  color_mode="rgb",

                                                  class_mode="categorical")

validation_data=test_datagen.flow_from_directory(test_path,

                                                  target_size=img_den_array.shape[:2],

                                                  batch_size=32,

                                                  color_mode="rgb",

                                                  class_mode="categorical")
checkpointer = ModelCheckpoint(filepath = 'my_furit_360_models.hdf5', verbose = 1, save_best_only = True)

history=model.fit_generator(generator=train_generator,

                   steps_per_epoch=1506//32,

                   epochs=50,

                   validation_data=validation_data,

                    callbacks=[checkpointer],

                   validation_steps=1000//32)
model.save("my_furit_360_50epochs.h5")

#save history with json 

import json



with open("my_furit_360.json","w") as f:

    json.dump(history.history,f)

    
# load json file

import codecs



with codecs.open("my_furit_360.json","r",encoding="utf-8") as file:

    hist=json.loads(file.read())

    

plt.plot(hist["acc"],"r")

plt.plot(hist["val_loss"],"b")

plt.legend()

plt.show()