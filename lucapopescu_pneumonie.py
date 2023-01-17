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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import cv2

import os

import random

from keras.models import Sequential

from keras.layers import Conv2D,Dense,Flatten,Dropout,MaxPooling2D

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

%matplotlib inline
img_size = (150 , 150)

batch_size = 10

no_of_epochs  = 10



train_datagen = ImageDataGenerator(rescale=1./255,

                                   rotation_range=15,

                                   shear_range=0.2,

                                   zoom_range=0.2

                                   )



test_datagen = ImageDataGenerator(rescale=1./255)



training_set = train_datagen.flow_from_directory('/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train',

                                                 target_size=img_size,

                                                 batch_size=10,

                                                 class_mode='binary')



test_set = test_datagen.flow_from_directory('/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/test',

                                            target_size=img_size,

                                            batch_size=10,

                                            class_mode='binary')



# Updated part --->

val_set = test_datagen.flow_from_directory('/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/val',

                                            target_size=img_size,

                                            batch_size=1,

                                            shuffle=False,

                                            class_mode='binary')

model = Sequential()

model.add(Conv2D(32 , (3,3) , input_shape=(150,150,3) , activation='relu'  ))

model.add(Conv2D(32 , (3,3) , activation='relu' ))

model.add(MaxPooling2D( pool_size=(2,2)  ))

model.add(Dropout(0.2))

model.add(Conv2D(64 , (3,3) , activation='relu' ))

model.add(Conv2D(64 , (3,3) , activation='relu' ))

model.add(MaxPooling2D( pool_size=(2,2)))

model.add(Dropout(0.2))

model.add(Conv2D(128 , (3,3) , activation='relu'))

model.add(Conv2D(128 , (3,3) , activation='relu'))

model.add(Flatten())

model.add(Dropout(0.2))

model.add(Dense( 32 , activation='relu' ))

model.add(Dense( 128 , activation='relu' ))

model.add(Dropout(0.2))

model.add(Dense( 32 , activation='relu' ))

model.add(Dense( 1 , activation='sigmoid' ))

model.compile(loss='binary_crossentropy' , optimizer='adam' , metrics=['accuracy'])
history = model.fit_generator(training_set,

                    steps_per_epoch=5216//batch_size,

                    epochs=10,

                    validation_data=test_set,

                    validation_steps=624//batch_size

                   )