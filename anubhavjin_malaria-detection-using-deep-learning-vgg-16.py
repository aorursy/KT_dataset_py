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
from keras.layers import Input,Dense,Lambda,Flatten
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from glob import glob
import matplotlib.pyplot as plt

IMAGE_SIZE=[224,224]
vgg=VGG16(input_shape=IMAGE_SIZE+[3],weights='imagenet',include_top=False)
for layer in vgg.layers:
    layer.trainable=False
folders=glob('/kaggle/input/cell-images-for-detecting-malaria/cell_images/cell_images/*')
x=Flatten()(vgg.output)
prediction = Dense(len(folders),activation='softmax')(x)
model =Model(inputs=vgg.input,outputs=prediction)
model.summary()
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
training_set=train_datagen.flow_from_directory('/kaggle/input/cell-images-for-detecting-malaria/cell_images/cell_images/',target_size=(224,224),batch_size=32,class_mode='categorical')
r=model.fit_generator(training_set,epochs=1,steps_per_epoch
                      =len(training_set))
r.history.keys()
plt.plot(r.history['accuracy'],label='train acc')
plt.legend()
plt.show()
