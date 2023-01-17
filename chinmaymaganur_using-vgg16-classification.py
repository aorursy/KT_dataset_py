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
from keras.layers import Flatten,Dense 
from keras.models import Model,Sequential
from keras.applications.vgg16 import VGG16,preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from glob import glob
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
train_dir='../input/dogs-cats-images/dataset/training_set'
test_dir='../input/dogs-cats-images/dataset/test_set'
help(VGG16)
vgg=VGG16(weights='imagenet', input_shape=[224,224,3],include_top=False)
#to not train the  model again
for layers in vgg.layers:
    layers.trainable=False
x=Flatten()(vgg.output)
prediction=Dense(2, activation='softmax')(x)
model=Model(vgg.input,prediction)

model.summary()
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
datagen=ImageDataGenerator(rescale=1.0/255.0)
train_datagen=datagen.flow_from_directory(train_dir,target_size=(224,224),batch_size=64)
test_datagen=datagen.flow_from_directory(test_dir,target_size=(224,224),batch_size=64)
fit_model=model.fit_generator(train_datagen,validation_data=test_datagen,epochs=5,steps_per_epoch=5)
hist=pd.DataFrame(fit_model.history)
hist['loss'].plot()
hist['val_loss'].plot()
from tensorflow.keras.preprocessing import image
def testing(dirpath):
    test_image=image.load_img(dirpath,target_size = (224, 224))
    test_image-image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    res=model.predict(test_image)
    print(res)
    if res[0][1]==1:
        print('DOG')
    else:
        print('cat')
train_datagen.class_indices
testing('../input/dogs-cats-images/dataset/test_set/dogs/dog.4001.jpg')
help(Model)