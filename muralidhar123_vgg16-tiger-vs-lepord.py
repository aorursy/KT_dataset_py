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
from keras.layers import Input,Lambda,Dense,Flatten

from keras.models import Model

from keras.applications.vgg16 import VGG16

from keras.applications.vgg16 import preprocess_input

from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from glob import glob

import warnings

warnings.filterwarnings('ignore')

IMAGE_SIZE = [224 , 224]
train_path = '/kaggle/input/compare-difference-bw-lepord-and-tiger/compare lepord and tiger/train/'

test_path = '/kaggle/input/compare-difference-bw-lepord-and-tiger/compare lepord and tiger/test'

vgg = VGG16(input_shape=IMAGE_SIZE + [3] , weights = 'imagenet' , include_top = False)
folder = '/kaggle/input/compare-difference-bw-lepord-and-tiger/compare lepord and tiger/train/'

print(len(folder))
for layers in vgg.layers:

    layers.trainable=False
x = Flatten()(vgg.output)

prediction = Dense(2,activation = 'softmax')(x)

model = Model(inputs=vgg.input , outputs = prediction)

model.summary()
from keras import optimizers

sgd = optimizers.SGD(lr=0.01,decay=1e-5,momentum=0.9)

model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])
datagen = ImageDataGenerator(rescale=1.0/255.0)

train_set = datagen.flow_from_directory('/kaggle/input/compare-difference-bw-lepord-and-tiger/compare lepord and tiger/train/',batch_size=100,target_size=(224,224))

test_set = datagen.flow_from_directory('/kaggle/input/compare-difference-bw-lepord-and-tiger/compare lepord and tiger/test',batch_size=100,target_size=(224,224))
# test_dagagen = ImageDataGenerator(

#     preprocessing_function = preprocess_input,

#     rotation_range = 40,

#     width_shift_range = 0.2,

#     height_shift_range = 0.2,

#     shear_range = 0.2,

#     zoom_range=0.2,

#     horizontal_flip=True,

#     fill_mode='nearest')
#!pip install datetime

import datetime

from datetime import datetime

from keras.callbacks import ModelCheckpoint , LearningRateScheduler

from keras.callbacks import ReduceLROnPlateau

checkpoint = ModelCheckpoint(filepath='mymodel.h5',verbose=1,save_best_only=True)

callbacks = [checkpoint]

start = datetime.now()

model.fit_generator(

    train_set,

    validation_data=test_set,

    epochs=10,

    steps_per_epoch=5,

    validation_steps=32,

    callbacks=callbacks,verbose=1)

duration = datetime.now() - start

print('training completed: ',duration)

from keras.models import load_model

import cv2

img = cv2.imread('/kaggle/input/compare-difference-bw-lepord-and-tiger/compare lepord and tiger/test/lepord/leopard.jpg')

model = load_model('mymodel.h5')

dim = (224,224)

img = cv2.resize(img , dim , interpolation=cv2.INTER_AREA)

x = image.img_to_array(img)

x = np.expand_dims(x , axis=0)

x = preprocess_input(x)

preds = model.predict(x)
preds