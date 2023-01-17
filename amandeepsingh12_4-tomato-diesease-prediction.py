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

        file = os.path.join(dirname, filename)



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from tensorflow.keras.layers import Input, Dense, Flatten

from keras import Model

from keras.applications.vgg16 import VGG16

from keras.preprocessing import image

from keras.models import Sequential
image_size = [224, 224]
vgg = VGG16(input_shape = image_size + [3], weights = 'imagenet', include_top =  False)
for layer in vgg.layers:

    layer.trainable = False
from glob import glob

folders = glob('/kaggle/input/tomato/New Plant Diseases Dataset(Augmented)/train/*')
folders

x = Flatten()(vgg.output)
prediction = Dense(len(folders), activation = 'softmax')(x)
model = Model(inputs = vgg.input, outputs = prediction)
model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
from keras.preprocessing.image import ImageDataGenerator
train_data_gen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_data_gen = ImageDataGenerator(rescale = 1./255)
train_set = train_data_gen.flow_from_directory('/kaggle/input/tomato/New Plant Diseases Dataset(Augmented)/train/', target_size = (224,224), batch_size = 32, class_mode = 'categorical')
test_set = test_data_gen.flow_from_directory('/kaggle/input/tomato/New Plant Diseases Dataset(Augmented)/valid/', target_size = (224,224), batch_size = 32, class_mode = 'categorical')
mod = model.fit_generator(

  train_set,

  validation_data=test_set,

  epochs=10,

  steps_per_epoch=len(train_set),

  validation_steps=len(test_set)

)
import matplotlib.pyplot as plt

plt.plot(mod.history['loss'], label='train loss')

plt.plot(mod.history['val_loss'], label='val loss')

plt.legend()

plt.show()



plt.plot(mod.history['accuracy'], label='train accuracy')

plt.plot(mod.history['val_accuracy'], label='val_accuracy')

plt.legend()

plt.show()