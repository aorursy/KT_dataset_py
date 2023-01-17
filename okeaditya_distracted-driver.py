# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import cv2

import os

import glob

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import tensorflow as tf

print(tf.__version__)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from tensorflow.keras import layers

from tensorflow.keras import optimizers

from tensorflow.keras import Input

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization, GlobalAveragePooling2D, Dense

from tensorflow.keras.optimizers import Adam

# from keras import Activation

from tensorflow.keras.models import Model, Sequential
labels = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']



col = {'c0': 'safe driving',

'c1': 'texting - right',

'c2': 'talking on the phone - right',

'c3': 'texting - left',

'c4': 'talking on the phone - left',

'c5':'operating the radio',

'c6': 'drinking',

'c7': 'reaching behind',

'c8': 'hair and makeup',

'c9': 'talking to passenger'}
DATA_PATH = "/kaggle/input/state-farm-distracted-driver-detection/imgs"

train_path = os.path.join(DATA_PATH, "train")

test_path = os.path.join(DATA_PATH, "test")
def read_image(path):

    img = cv2.imread(path, cv2.IMREAD_COLOR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img
train_datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2,

                                   height_shift_range=0.3, zoom_range=0.3,

                                   channel_shift_range=0.0,

                                   fill_mode='nearest', cval=0.0, horizontal_flip=True, vertical_flip=False, rescale=1/255.,

                                   data_format='channels_last', validation_split=0.3,

                                   dtype='float32')
val_datagen = ImageDataGenerator(rescale=1/255.,

                                   data_format='channels_last', validation_split=0.3,

                                   dtype='float32')
train_generator = train_datagen.flow_from_directory(train_path, target_size=(256,256), color_mode="grayscale", 

                                                    class_mode="categorical", batch_size=64, subset="training")
val_generator = val_datagen.flow_from_directory(train_path, target_size=(256,256), color_mode="grayscale", 

                                                    class_mode="categorical", batch_size=64, subset="validation")
df = pd.read_csv("/kaggle/input/state-farm-distracted-driver-detection/driver_imgs_list.csv")
def make_model(input_shape=(256,256,1)):

    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=input_shape, activation='elu'))

#     model.add(Activation(activation='elu'))

    model.add(MaxPooling2D())

    model.add(BatchNormalization())



    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='elu'))

#     model.add(Activation(activation=elu))

    model.add(MaxPooling2D())

    model.add(BatchNormalization())



    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='elu'))

#     model.add(Activation(activation=elu))

    model.add(MaxPooling2D())

    model.add(BatchNormalization())



    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='elu'))

#     model.add(Activation(activation=elu))

    model.add(MaxPooling2D())

    model.add(BatchNormalization())



    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='elu'))

#     model.add(Activation(activation=elu))

    model.add(MaxPooling2D())

    model.add(BatchNormalization())



#     model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='elu'))

# #     model.add(Activation(activation=elu))

#     model.add(MaxPooling2D())

#     model.add(BatchNormalization())



    # model.add(Conv2D(filters=32, kernel_size=(3, 3), activation=elu))

    # model.add(Activation(activation=elu))

    # model.add(MaxPooling2D())

    # model.add(BatchNormalization())



    model.add(GlobalAveragePooling2D())

    model.add(Dense(3000, activation='elu'))

#     model.add(Activation(activation=elu))

    model.add(Dropout(rate=0.25))

    model.add(Dense(2000, activation='elu'))

#     model.add(Activation(activation='elu'))

    model.add(Dropout(rate=0.25))

    model.add(Dense(10, activation='softmax'))

    

    return (model)

    
model = make_model()

opt = Adam(lr=0.0001)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])    
model.summary()
epochs = 3

batch_size = 64
filepath = 'distracted_driver.h5'

# reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5,

#                              verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,

                             save_best_only=True, save_weights_only=False, mode='auto', period=1)

callback_l = [checkpoint]
history = model.fit(train_generator,

                              steps_per_epoch= train_generator.samples // batch_size,

                              epochs=epochs,

                              validation_data=val_generator,

                              validation_steps=val_generator.samples // batch_size, callbacks=callback_l)
model_json = model.to_json()

with open("model.json", "w") as json_file:

    json_file.write(model_json)

print("Saved model to disk")
tf.saved_model.save(model, "distracted_driver")
# loaded = tf.saved_model.load("distracted_driver")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()
tflite_model_file = 'distracted_driver.tflite'

with open(tflite_model_file, "wb") as f:

    f.write(tflite_model)
clss = {'0': 'safe driving',

'1': 'texting - right',

'2': 'talking on the phone - right',

'3': 'texting - left',

'4': 'talking on the phone - left',

'5':'operating the radio',

'6': 'drinking',

'7': 'reaching behind',

'8': 'hair and makeup',

'9': 'talking to passenger'}
# Checking working

interpreter = tf.lite.Interpreter(model_content=tflite_model)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()

output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

print(input_shape)
input_shape = (1,256,256,1)

# input_data = np.ones(input_shape, dtype=np.float32)

input_path = "/kaggle/input/state-farm-distracted-driver-detection/imgs/test/img_100001.jpg"

input_data = cv2.imread(input_path,0)

input_data = cv2.resize(input_data, (256, 256))

# input_data = cv2.cv2Color(input_data, cv2.COLORBGR2RGB)

input_data = (input_data / 255).astype(np.float32)

input_data = tf.expand_dims(input_data, -1)

input_data = tf.expand_dims(input_data, 0)

print(input_data.shape)

interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

tflite_results = interpreter.get_tensor(output_details[0]['index'])

cpred = np.argmax(tflite_results, axis=1)

print(cpred)

print(clss[str(cpred[0])])
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()
tflite_results = interpreter.get_tensor(output_details[0]['index'])
print(tflite_results)
cpred = np.argmax(tflite_results, axis=1)
print(cpred)
print(clss[str(cpred[0])])
! tar -zcvf distracted_driver.tar.gz "distracted_driver/"