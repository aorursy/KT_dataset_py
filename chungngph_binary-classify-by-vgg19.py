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
import numpy as np
import tensorflow as tf
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.models import Model
from keras.preprocessing import image
import json
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import json
import glob
import keras.layers
from keras.optimizers import SGD
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense
from PIL import Image
import zipfile
import cv2
import zipfile


zip_files = ['test1', 'train']
# Will unzip the files so that you can see them..
for zip_file in zip_files:
    with zipfile.ZipFile("/kaggle/input/dogs-vs-cats/{}.zip".format(zip_file),"r") as z:
        z.extractall(".")
        print("{} unzipped".format(zip_file))
os.listdir('./test1/')
os.mkdir('./train/dogs')
os.mkdir('./train/cats')
os.system('mv ./train/cat* ./train/cats')
os.system('mv ./train/dog* ./train/dogs')
batch_size = 32

data = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2,
                          shear_range=0.15, horizontal_flip=True, fill_mode="nearest", validation_split=0.2)
traindata = data.flow_from_directory(directory="./train/", target_size=(224,224), batch_size=batch_size, subset='training',class_mode='binary')
testdata = data.flow_from_directory(directory="./train/", target_size=(224,224), batch_size=batch_size, subset='validation',class_mode='binary')
traindata.next()[1]
base_model = VGG19(weights='imagenet')
base_model.summary()
with tf.device('/cpu:0'):
    predictions = Dense(1, activation="sigmoid")(base_model.layers[-2].output)
    model = Model(inputs=base_model.input, outputs=predictions)
model.summary()
opt = SGD(lr=1e-2)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc'])
model.fit_generator(generator=traindata,
                    epochs=10,
                    validation_data=testdata,
                    steps_per_epoch=traindata.samples//batch_size,
                    validation_steps=testdata.samples//batch_size)
