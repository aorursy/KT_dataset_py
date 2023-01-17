# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import tensorflow as tf
train_path='../input/chest-xray-pneumonia/chest_xray/train/'
test_path='../input/chest-xray-pneumonia/chest_xray/test/'
val_path='../input/chest-xray-pneumonia/chest_xray/val/'
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_gen=ImageDataGenerator(rescale=1/255)

train_generator=train_gen.flow_from_directory(train_path,target_size=(150,150),batch_size=32,class_mode='categorical')
test_gen=ImageDataGenerator(rescale=1/255)

test_generator=test_gen.flow_from_directory(test_path,target_size=(150,150),batch_size=32,class_mode='categorical')
val_gen=ImageDataGenerator(rescale=1/255)

val_generator=val_gen.flow_from_directory(val_path,target_size=(150,150),batch_size=32,class_mode='categorical')
model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256,activation='relu'),
    tf.keras.layers.Dense(2,activation='softmax')
        
])
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping
early_stop=EarlyStopping(patience=2,monitor='loss')
model.fit_generator(train_generator,validation_data=val_generator,epochs=20,callbacks=[early_stop])
