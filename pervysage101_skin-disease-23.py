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
# Downloading dataset

!wget "https://www.dropbox.com/s/h5yora9j0onglw6/train.zip?dl=1"

!wget "https://www.dropbox.com/s/c94io61nmldcgv8/test.zip?dl=1"

!unzip -q train.zip?dl=1

!unzip -q test.zip?dl=1

!rm train.zip?dl=1

!rm test.zip?dl=1
!unzip -q train.zip?dl=1.1
!rm train.zip?dl=1

!rm test.zip?dl=1

# !rm test.zip?dl=1.2
import keras

from keras import backend as K

from keras.layers.core import Dense, Activation

from keras.optimizers import Adam

from keras.metrics import categorical_crossentropy

from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing import image

from keras.models import Model

from keras.applications import imagenet_utils

from keras.layers import Dense,GlobalAveragePooling2D

from keras.applications import MobileNet

from keras.applications.mobilenet import preprocess_input

import numpy as np

from IPython.display import Image

from keras.optimizers import Adam



from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Convolution2D, MaxPooling2D

from keras.layers.normalization import BatchNormalization



# !pip uninstall tensorflow

# !pip install tensorflow-gpu==1.14

import tensorflow

print(tensorflow.__version__)
# # from ../input/mri-model-file import model

# import os

# os.chdir("/kaggle/input/")

# import models

# print(os.listdir("."))

# model = models.mri_model_2()

# model.summary()

# os.chdir("/kaggle/working/")
# model = mri_model_2()

base = keras.applications.mobilenet.MobileNet(weights='imagenet')

last_layer=base.layers[-2].output

layers = Dropout(0.3)(last_layer)

layers = Dense(23,activation='softmax')(layers)

model = Model(base.input,outputs=layers)

# model.summary()
!ls ../input

# model = keras.models.load_model('../input/mobilenetepoch150/mobilenet-epoch147-val63.h5')

# model.summary()

for layer in model.layers:

    layer.trainable=False



for layer in model.layers[-16:]:

    layer.trainable = True

    

model.summary()
import keras

from keras import *

import time

from keras.callbacks import ModelCheckpoint
_DATASET_FILEPATH = "train/"

_TEST_FILEPATH = "test/"

_IMAGES_HEIGHT, _IMAGES_WIDTH = 224,224

_BATCH_SIZE = 100



train_generator = keras.preprocessing.image.ImageDataGenerator(

  rescale=1./255,

  shear_range=0.2,

  zoom_range=0.2,

  horizontal_flip=True,

  rotation_range=90,

)



train_flow = train_generator.flow_from_directory(

  directory=_DATASET_FILEPATH,

  target_size=(_IMAGES_HEIGHT, _IMAGES_WIDTH),

  batch_size=_BATCH_SIZE,

  class_mode='categorical',

)



# validation_generator = keras.preprocessing.image.ImageDataGenerator(

#     rescale=1./255)



# validation_flow = validation_generator.flow_from_directory(

#   directory=_VALIDATION_FILEPATH,

#   target_size=(_IMAGES_HEIGHT, _IMAGES_WIDTH),

#   batch_size=_BATCH_SIZE,

#   class_mode='categorical',

# )



test_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)



test_flow = test_generator.flow_from_directory(

  directory=_TEST_FILEPATH,

  target_size=(_IMAGES_HEIGHT, _IMAGES_WIDTH),

  batch_size=_BATCH_SIZE,

  class_mode='categorical',

)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.metrics_names
from keras.applications.resnet50 import ResNet50

from keras import Input

from keras.layers import Flatten,Dense

# resnet_model = ResNet50(input_shape=[80,80,3], include_top=False,weights='imagenet')

# # model.summary()

# last_layer = resnet_model.layers[-1].output

# x= Flatten(name='flatten')(last_layer)

# out = Dense(23, activation='softmax', name='output_layer')(x)

# model = Model(inputs=resnet_model.layers[0].output,outputs= out)



# for layer in model.layers:

#     layer.trainable = True



# model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])



# model.summary()
# from keras.models import load_model

# !ls ../input

# model = load_model("../input/resnet-weights-improvement-62-0.38.h5")

# loss,accuracy = model.evaluate_generator(generator=test_flow)

# print("[INFO] Model loss: "+str(loss)+"   Accuracy:"+str(accuracy))
!find  . -name 'resnet*' -exec rm {} \;
t = time.time()

logdir=""

tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

filepath="mobilenet-scratch-recent-3.h5"

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

h=model.fit_generator(generator=train_flow,validation_data=test_flow,epochs=30,callbacks=[checkpoint])

loss,accuracy = model.evaluate_generator(generator=test_flow)

print("[INFO] Model loss: "+str(loss)+"   Accuracy:"+str(accuracy)+"  Time: "+str(time.time()-t))

model.save("mobilenet_model_scratch.h5")

print("[INFO] Model Saved")
import matplotlib.pyplot as plt

import matplotlib.image as mpimg



h.history.keys()

plt.plot(h.history['accuracy'])

plt.plot(h.history['val_accuracy'])

plt.title("Accuracy in training and test set")

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend(["Train", "Test"], loc = 'upper_left')

plt.show()
!rm -r train

!rm -r test
# if tensorflow.test.gpu_device_name():

#     print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

# else:

#     print("Please install GPU version of TF")