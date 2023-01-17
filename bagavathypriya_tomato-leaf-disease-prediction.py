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
import tensorflow as tf
tf.__version__
from tensorflow.keras.layers import Input, Lambda,Dense,Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
from glob import glob
Img_size=[224,224]
train_path="../input/tomato/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train"

train_path="../input/tomato/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/valid"
inception = InceptionV3(input_shape=Img_size+[3],weights='imagenet',include_top=False)

#include_top = False as we include the first and last layer
type(inception)
for layer in inception.layers:
    layer.trainable=False
folders=glob("../input/tomato/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train/*")
len(folders)
x=Flatten()(inception.output)
prediction=Dense(len(folders),activation='softmax')(x)
model=Model(inputs=inception.input,outputs=prediction)
model.summary()
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_data=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

test_data=ImageDataGenerator(rescale=1./255)
train_set=train_data.flow_from_directory('../input/tomato/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train',
                                        target_size=(224,224),batch_size=32,class_mode='categorical')
test_set=train_data.flow_from_directory('../input/tomato/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/valid',
                                        target_size=(224,224),batch_size=32,class_mode='categorical')
r=model.fit_generator(train_set,validation_data=test_set,epochs=10,steps_per_epoch=len(train_set),validation_steps=len(test_set))
import matplotlib.pyplot as plt
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')