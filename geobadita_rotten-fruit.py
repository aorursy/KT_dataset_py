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
import tensorflow as tf

import keras

import torch

import torchvision

import cv2

import numpy as np

import glob



train_fresh_apple_dir = '/kaggle/input/fruits-fresh-and-rotten-for-classification/dataset/dataset/train/freshapples'

train_rotten_apple_dir = '/kaggle/input/fruits-fresh-and-rotten-for-classification/dataset/dataset/train/rottenapples'

train_fresh_banana_dir = '/kaggle/input/fruits-fresh-and-rotten-for-classification/dataset/dataset/train/freshbanana'

train_rotten_banana_dir = '/kaggle/input/fruits-fresh-and-rotten-for-classification/dataset/dataset/train/rottenbanana'

train_fresh_orange_dir = '/kaggle/input/fruits-fresh-and-rotten-for-classification/dataset/dataset/train/freshoranges'

train_rotten_orange_dir = '/kaggle/input/fruits-fresh-and-rotten-for-classification/dataset/dataset/train/rottenoranges'



test_fresh_apple_dir = '/kaggle/input/fruits-fresh-and-rotten-for-classification/dataset/dataset/test/freshapples'

test_rotten_apple_dir = '/kaggle/input/fruits-fresh-and-rotten-for-classification/dataset/dataset/test/rottenapples'

test_fresh_banana_dir = '/kaggle/input/fruits-fresh-and-rotten-for-classification/dataset/dataset/test/freshbanana'

test_rotten_banana_dir = '/kaggle/input/fruits-fresh-and-rotten-for-classification/dataset/dataset/test/rottenbanana'

test_fresh_orange_dir = '/kaggle/input/fruits-fresh-and-rotten-for-classification/dataset/dataset/test/freshoranges'

test_rotten_orange_dir = '/kaggle/input/fruits-fresh-and-rotten-for-classification/dataset/dataset/test/rottenoranges'



train_fresh_apple_files = glob.glob(train_fresh_apple_dir + '/*')

train_rotten_apple_files = glob.glob(train_rotten_apple_dir + '/*')

train_fresh_banana_files = glob.glob(train_fresh_banana_dir + '/*')

train_rotten_banana_files = glob.glob(train_rotten_banana_dir + '/*')

train_fresh_orange_files = glob.glob(train_fresh_orange_dir + '/*')

train_rotten_orange_files = glob.glob(train_rotten_orange_dir + '/*')



print('train samples of fresh apple:', len(train_fresh_apple_files))

print('train samples of rotten apple:', len(train_rotten_apple_files))

print('train samples of apple:', len(train_fresh_apple_files) + len(train_rotten_apple_files))

print('train samples of fresh banana:', len(train_fresh_banana_files))

print('train samples of rotten banana:', len(train_rotten_banana_files))

print('train samples of banana:', len(train_fresh_banana_files) + len(train_rotten_banana_files))

print('train samples of fresh orange:', len(train_fresh_orange_files))

print('train samples of rotten orange:', len(train_rotten_orange_files))

print('train samples of orange:', len(train_fresh_orange_files) + len(train_rotten_orange_files))

print('total train samples:', 

      len(train_fresh_apple_files) + 

      len(train_rotten_apple_files) + 

      len(train_fresh_banana_files) + 

      len(train_rotten_banana_files) +

      len(train_fresh_orange_files) + 

      len(train_rotten_orange_files))



test_fresh_apple_files = glob.glob(test_fresh_apple_dir + '/*')

test_rotten_apple_files = glob.glob(test_rotten_apple_dir + '/*')

test_fresh_banana_files = glob.glob(test_fresh_banana_dir + '/*')

test_rotten_banana_files = glob.glob(test_rotten_banana_dir + '/*')

test_fresh_orange_files = glob.glob(test_fresh_orange_dir + '/*')

test_rotten_orange_files = glob.glob(test_rotten_orange_dir + '/*')



print('test samples of fresh apple:', len(test_fresh_apple_files))

print('test samples of rotten apple:', len(test_rotten_apple_files))

print('test samples of apple:', len(test_fresh_apple_files) + len(test_rotten_apple_files))

print('test samples of fresh banana:', len(test_fresh_banana_files))

print('test samples of rotten banana:', len(test_rotten_banana_files))

print('test samples of banana:', len(test_fresh_banana_files) + len(test_rotten_banana_files))

print('test samples of fresh orange:', len(test_fresh_orange_files))

print('test samples of rotten orange:', len(test_rotten_orange_files))

print('test samples of orange:', len(test_fresh_orange_files) + len(test_rotten_orange_files))

print('total test samples:', 

      len(test_fresh_apple_files) + 

      len(test_rotten_apple_files) + 

      len(test_fresh_banana_files) + 

      len(test_rotten_banana_files) +

      len(test_fresh_orange_files) + 

      len(test_rotten_orange_files))
from tensorflow.keras.applications import MobileNetV2

from tensorflow.keras.layers import Flatten

from tensorflow.keras.models import Model

import pandas as pd





input_shape = (224, 224, 3)

mobilenet_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=input_shape)

#output = mobilenet.layers[-1].output

#output = Flatten()(output)

#mobilenet_model = Model(mobilenet.input, output)



mobilenet_model.trainable = True

#fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer

#for layer in mobilenet_model.layers[:fine_tune_at]:

#    layer.trainable =  False



mobilenet_model.summary()
import os

from tensorflow.keras.layers import Conv2D

from tensorflow.keras.layers import MaxPooling2D

from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import Dropout

from tensorflow.keras.layers import InputLayer

from tensorflow.keras.layers import GlobalAveragePooling2D

from tensorflow.keras.models import Sequential

from tensorflow.keras import optimizers

from tensorflow.keras.preprocessing.image import ImageDataGenerator





train_datagen = ImageDataGenerator(rescale=1./255, 

                                   zoom_range=0.3, 

                                   rotation_range=50,

                                   width_shift_range=0.2, 

                                   height_shift_range=0.2, 

                                   shear_range=0.2, 

                                   horizontal_flip=True, 

                                   fill_mode='nearest')





val_datagen = ImageDataGenerator(rescale=1./255)



dataset_path = '/kaggle/input/fruits-fresh-and-rotten-for-classification/dataset/dataset'



train_set_path = os.path.join(dataset_path, 'train')



val_set_path = os.path.join(dataset_path, 'test')



BATCH_SIZE = 64

TARGET_SIZE = input_shape[:2]



train_generator = train_datagen.flow_from_directory(train_set_path,

                                                 target_size = TARGET_SIZE,

                                                 batch_size = BATCH_SIZE,

                                                 class_mode = 'categorical')



val_generator = val_datagen.flow_from_directory(val_set_path,

                                                target_size = TARGET_SIZE,

                                                batch_size = BATCH_SIZE,

                                                class_mode = 'categorical')





model = Sequential()

model.add(mobilenet_model)

model.add(GlobalAveragePooling2D())

model.add(Dense(units=6, activation='softmax'))



model.compile(loss='categorical_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-5),

              metrics=['accuracy'])

model.summary()

EPOCHS = 7

history = model.fit_generator(train_generator, 

                              steps_per_epoch=train_generator.n // BATCH_SIZE, 

                              epochs=EPOCHS,

                              validation_data=val_generator, 

                              validation_steps=val_generator.n // BATCH_SIZE, 

                              verbose=1)  
model.save('Fresh_Rotten_Fruits_MobileNetV2_Transfer_Learning.h5')
import matplotlib.pyplot as plt



f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

t = f.suptitle('Pre-trained MobileNetV2 Transfer Learn with Fine-Tuning & Image Augmentation Performance ', fontsize=12)

f.subplots_adjust(top=0.85, wspace=0.3)



epoch_list = list(range(1,8))

ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')

ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')

ax1.set_xticks(np.arange(0, 8, 1))

ax1.set_ylabel('Accuracy Value')

ax1.set_xlabel('Epoch #')

ax1.set_title('Accuracy')

l1 = ax1.legend(loc="best")



ax2.plot(epoch_list, history.history['loss'], label='Train Loss')

ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')

ax2.set_xticks(np.arange(0, 12, 1))

ax2.set_ylabel('Loss Value')

ax2.set_xlabel('Epoch #')

ax2.set_title('Loss')

l2 = ax2.legend(loc="best")
