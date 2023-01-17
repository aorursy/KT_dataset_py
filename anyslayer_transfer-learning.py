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
import os 
import keras
import torch
import tensorflow as tf
from IPython.display import Image, display
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D,Conv2D
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_size = 224

data_generator = ImageDataGenerator(featurewise_center=True,featurewise_std_normalization=True)
data_generatorv = ImageDataGenerator()

train_generator = data_generator.flow_from_directory(
                                        directory='../input/waste-classification-data/DATASET/TRAIN',
                                        target_size=(image_size, image_size),
                                        batch_size=10,
                                        color_mode='rgb',
                                        class_mode='categorical',
                                        shuffle=True)

validation_generator = data_generatorv.flow_from_directory(
                                        directory='../input/waste-classification-data/DATASET/TEST',
                                        target_size=(image_size, image_size),
                                        color_mode='rgb',
                                        class_mode='categorical',
                                        shuffle=True)
# from keras.applications import InceptionResNetV2
%load_ext tensorboard
num_classes = 2
my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
# my_new_model.add(InceptionResNetV2(include_top=False, pooling='avg', weights='imagenet',input_shape=(224,224,3)))

# my_new_model.add(Conv2D(50,kernel_size=(3,3),activation='relu'))myabs
# my_new_model.add(Dense(50,activation='relu'))


my_new_model.add(Dense(num_classes, activation='softmax'))
my_new_model.layers[0].trainable = False

my_new_model.compile(optimizer='adam', 
                     loss='categorical_crossentropy', 
                     metrics=['accuracy'])
import datetime
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

my_new_model.fit(train_generator,
                steps_per_epoch=80,
                epochs=50)
fit_stats = my_new_model.fit_generator(train_generator,
                                       steps_per_epoch=80,
                                       epochs=50,
                                       validation_data=validation_generator,
                                       validation_steps=1, callbacks=[tensorboard_callback]).history
import matplotlib.pylab as plt
plt.figure(figsize=(30,4))
plt.ylabel("Loss (training and validation)")
plt.xlabel("Training Steps")
plt.ylim([0,2])
plt.plot(fit_stats["loss"])
plt.plot(fit_stats["val_loss"])
plt.legend(['loss','val_loss'])

plt.figure(figsize=(30,4))
plt.ylabel("Accuracy (training and validation)")
plt.xlabel("Training Steps")
plt.ylim([0,1])
plt.plot(fit_stats["accuracy"])
plt.plot(fit_stats["val_accuracy"])
plt.legend(['acc', 'val_acc'])

%tensorboard --logdir logs/fit
for z,k in train_generator:
    print(z)
    count(z)
    
_, acc = my_new_model.evaluate_generator(validation_generator, steps=10, verbose=0)
print('Test Accuracy: %.3f' % (acc * 100))