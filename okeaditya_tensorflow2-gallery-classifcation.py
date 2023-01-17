# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import cv2

import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import tensorflow as tf

print(tf.__version__)

# tested on tensorflow v2.1.0

# Will not work on tensorflow v1
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import model_from_json
from tensorflow.keras import Model, Sequential

from tensorflow.keras import layers

from tensorflow.keras import optimizers

# from tensorflow.keras import ImageDatagenerator

from tensorflow.keras.callbacks import ModelCheckpoint
DATA_PATH = '/kaggle/input/mobile-gallery-image-classification-data/mobile_gallery_image_classification/mobile_gallery_image_classification'
print(os.listdir(DATA_PATH))
train_path = os.path.join(DATA_PATH, 'train')

test_path = os.path.join(DATA_PATH, 'test')

# print(os.listdir(train_path))

# print(os.listdir(test_path))
def show_samples_train(train_path, to_analyze):

    for folder_name in os.listdir(train_path):

        image_folder = os.path.join(train_path, folder_name)

        count = 0

        for image in os.listdir(image_folder):

            image_path = os.path.join(image_folder, image)

            if(count < to_analyze):

                image = cv2.imread(image_path)

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                plt.imshow(image)

                plt.title(folder_name)

                plt.xticks([])

                plt.yticks([])

                plt.show()

                count += 1

            else:

                break
def show_samples_test(test_path, to_analyze):

    count = 0

    for image in os.listdir(test_path):

        image_path = os.path.join(test_path, image)

        if(count < to_analyze):

                image = cv2.imread(image_path)

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                plt.imshow(image)

                plt.xticks([])

                plt.yticks([])

                plt.show()

                count += 1

        else:

            break

    
# Given the train folder it finds the distribution of the number of classes in it

def get_distribution_train(train_path, display=False):

    lengths = {}

    count = 0

    for folder in os.listdir(train_path):

        folder_path = os.path.join(train_path, folder)

        length = len(os.listdir(folder_path))

        count += length

        lengths[folder] = length

    

    if display is True:

        names = list(lengths.keys())

        values = list(lengths.values())

        plt.bar(range(len(lengths)), values, tick_label=names)

        plt.show()

    

    print("Total number of classes in training folder = %d"%(len(lengths)))

    print("Total number of images in training folder = %d"%(count))

    return lengths    
# to_analyze = 10 

# Analyzing 10 images per folder

show_samples_train(train_path, to_analyze=10)
# There anre only 7 images in the test folder will add more

# to_analyze = 8

show_samples_test(test_path, to_analyze=8)
images_per_folder = get_distribution_train(train_path, display=True)
train_datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2,

                                   height_shift_range=0.3, zoom_range=0.3,

                                   channel_shift_range=0.0,

                                   fill_mode='nearest', cval=0.0, horizontal_flip=True, vertical_flip=False, rescale=1/255.,

                                   data_format='channels_last', validation_split=0.3,

                                   dtype='float32')
val_datagen = ImageDataGenerator(rescale=1/255.,

                                   data_format='channels_last', validation_split=0.3,

                                   dtype='float32')
train_generator = train_datagen.flow_from_directory(train_path, target_size=(256,256), color_mode="rgb", 

                                                    class_mode="categorical", batch_size=64, subset="training")
val_generator = val_datagen.flow_from_directory(train_path, target_size=(256,256), color_mode="rgb", 

                                                    class_mode="categorical", batch_size=64, subset="validation")
def make_model(i_shape=(256,256,1), o_shape=10):

    model = Sequential()

    

    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), input_shape=i_shape, activation='elu'))

    model.add(layers.MaxPooling2D())

    model.add(layers.BatchNormalization())



    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='elu'))

    model.add(layers.MaxPooling2D())

    model.add(layers.BatchNormalization())



    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='elu'))

    model.add(layers.MaxPooling2D())

    model.add(layers.BatchNormalization())



    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='elu'))

    model.add(layers.MaxPooling2D())

    model.add(layers.BatchNormalization())

    model.add(layers.Dropout(rate=0.20))



    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation='elu'))

    model.add(layers.MaxPooling2D())

    model.add(layers.BatchNormalization())

    model.add(layers.Dropout(rate=0.20))



#     model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='elu'))

# #     model.add(Activation(activation=elu))

#     model.add(MaxPooling2D())

#     model.add(BatchNormalization())



    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dense(256, activation='elu'))

    model.add(layers.Dropout(rate=0.20))

    

    model.add(layers.Dense(128, activation='elu'))

    model.add(layers.Dropout(rate=0.20))

    model.add(layers.Dense(o_shape, activation='softmax'))

    

    return model

    
model = make_model(i_shape=(256,256,3), o_shape=6)
model.summary()
opt = optimizers.Adam(lr=0.000001)
filepath = 'mobile_gallery.h5'
chkpt = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,

                        save_best_only=True, save_weights_only=False, mode='auto', period=1) 

callbacks_l = [chkpt]
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
epochs = 20

batch_size = 32
# Note tf v2 will deprecate model.fit_generator we need to use model.fit instead

# history = model.fit(train_generator,

#                               steps_per_epoch= train_generator.samples // batch_size,

#                               epochs=epochs,

#                               validation_data=val_generator,

#                               validation_steps=val_generator.samples // batch_size, callbacks=callbacks_l)
# acc = history.history['acc']

# val_acc = history.history['val_acc']

# loss = history.history['loss']

# val_loss = history.history['val_loss']



# epochs = range(1, len(acc) + 1)



# plt.plot(epochs, acc, 'g', label='Training acc')

# plt.plot(epochs, val_acc, 'b', label='Validation acc')

# plt.title('Training and validation accuracy')

# plt.legend()

# plt.figure()



# plt.plot(epochs, loss, 'g', label='Training loss')

# plt.plot(epochs, val_loss, 'b', label='Validation loss')

# plt.title('Training and validation loss')

# plt.legend()

# plt.show()
# model_json = model.to_json()

# with open("model.json", "w") as json_file:

#     json_file.write(model_json)

# print("Saved model to disk")
# Plot some graphs

# Use tf_explain