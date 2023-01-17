# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from pathlib import Path



# Path to train directory 

train = Path('../input/brain-mri-images-for-brain-tumor-detection/brain_tumor_dataset/') 



#test = Path('../input/brain-mri-images-for-brain-tumor-detection') 
from keras.preprocessing.image import ImageDataGenerator



# create a data generator

datagen_train = ImageDataGenerator(rescale=1./255,  validation_split=0.2)
train_it = datagen_train.flow_from_directory(

    train, 

    subset='training' , color_mode ='grayscale', class_mode='binary'

)



val_it = datagen_train.flow_from_directory(

    train,

    subset='validation', color_mode ='grayscale',class_mode='binary'

)
from keras import layers

from keras import models

#from keras import regularizers



model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(256, 256, 1)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))
#compile model using accuracy to measure model performance

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
#train the model

model.fit_generator(train_it, steps_per_epoch=10,epochs=20)
val_loss, val_acc = model.evaluate_generator(val_it, steps=50, verbose =1)

print('test acc:', val_acc)