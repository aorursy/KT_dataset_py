# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('../input/genderdetectionface'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#initialising NN
from tensorflow.keras.models import Sequential
#convolution layers
from tensorflow.keras.layers import Convolution2D
#pooling layers
from tensorflow.keras.layers import MaxPooling2D
#convert pool layer to vector form
from tensorflow.keras.layers import Flatten
#initialising  fully connected layer or hidden layer
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
import tensorflow as tf
from tensorflow.nn import local_response_normalization
from  tensorflow.keras.layers import LayerNormalization,BatchNormalization

'''model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=4096,activation="relu"))
model.add(tf.keras.layers.Dense(units=4096,activation="relu"))
model.add(tf.keras.layers.Dense(units=2, activation="softmax"))





from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
opt = Adam(lr=0.0001)
model.compile(optimizer=opt, loss=categorical_crossentropy, metrics=['accuracy'])


from keras.preprocessing.image import ImageDataGenerator 
train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        '../input/genderdetectionface/dataset1/train/',
        target_size=(224,224),
        batch_size=5,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        '../input/genderdetectionface/dataset1/test/',
        target_size=(224,224),
        batch_size=5,
        class_mode='categorical')



from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')
hist = model.fit_generator(steps_per_epoch=100,generator=training_set, validation_data= test_set, validation_steps=10,epochs=100,callbacks=[checkpoint,early])'''
#initialising cnn
classifier =  tf.keras.Sequential()
classifier.add(tf.keras.layers.Conv2D(filters=96,kernel_size=(5,5),strides=(2,2),padding='same', activation='relu', input_shape=(227,227,3)))
#classifier.add(tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),padding='same', activation='relu', input_shape=(64,64,3)))
classifier.add(tf.keras.layers.MaxPooling2D(pool_size=2))
#classifier.add(tf.keras.layers.LayerNormalization())
#classifier.add(tf.keras.layers.Dropout(0.2))
classifier.add(tf.keras.layers.Conv2D(filters=96,kernel_size=(1,1),activation='relu'))

classifier.add(tf.keras.layers.Conv2D(filters=256,kernel_size=(5,5),strides=(1,1),padding='same', activation='relu'))
#classifier.add(tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),padding='same', activation='relu'))
classifier.add(tf.keras.layers.MaxPooling2D(pool_size=2))
#classifier.add(tf.keras.layers.LayerNormalization())
#classifier.add(tf.keras.layers.Dropout(0.2))
classifier.add(tf.keras.layers.Conv2D(filters=256,kernel_size=(1,1),activation='relu'))


classifier.add(tf.keras.layers.Conv2D(filters=256,kernel_size=(3,3),strides=(1,1),padding='same', activation='relu'))
#classifier.add(tf.keras.layers.Conv2D(filters=256,kernel_size=(3,3), activation='relu'))
classifier.add(tf.keras.layers.MaxPooling2D(pool_size=2))
#classifier.add(tf.keras.layers.LayerNormalization())
classifier.add(tf.keras.layers.Conv2D(filters=256,kernel_size=(1,1),activation='relu'))


classifier.add(tf.keras.layers.Flatten())
#classifier.add(tf.keras.layers.Dropout(0.4))
#step 4 full conncetion step or hidden layer step
#128 number of hidden nodes
classifier.add(tf.keras.layers.Dense(4019, activation='relu'))
classifier.add(tf.keras.layers.Dropout(0.2))
classifier.add(tf.keras.layers.Dense(2000, activation='relu'))
classifier.add(tf.keras.layers.Dropout(0.2))
classifier.add(tf.keras.layers.Dense(1000, activation='relu'))

#MAKING OUPUT LAYER
classifier.add(tf.keras.layers.Dense(2, activation='softmax'))

from tensorflow.keras.optimizers import Adam
opt = Adam(lr=0.0001)
classifier.compile(optimizer = opt,loss = 'categorical_crossentropy',metrics = ['accuracy'])
'''from tensorflow.keras.optimizers import Adam
opt = Adam(lr=0.0001)
classifier.compile(optimizer = opt,loss = 'binary_crossentropy',metrics = ['accuracy'])'''

#fitting images to CNN for training
from keras.preprocessing.image import ImageDataGenerator 
train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        '../input/genderdetectionface/dataset1/train/',
        target_size=(227,227),
        batch_size=5,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        '../input/genderdetectionface/dataset1/test/',
        target_size=(227,227),
        batch_size=5,
        class_mode='categorical')



# Set callback functions to early stop training and save the best model so far
callbacks = [EarlyStopping(monitor='val_loss', patience=10),
             ModelCheckpoint(filepath='1.h5', monitor='val_loss', save_best_only=True)]
print(training_set.class_indices)

history = classifier.fit_generator(
        training_set,
        epochs=100,
        callbacks=callbacks,
        validation_data=test_set)
from matplotlib import pyplot as plt
plt.plot(history.history['accuracy'],'green',label='Accuracy')
plt.plot(history.history['loss'],'red',label='Loss')
plt.title('Training Accuracy & Loss')
plt.xlabel('Epoch')
plt.figure()
plt.plot(history.history['val_accuracy'],'green',label='Accuracy')
plt.plot(history.history['val_loss'],'red',label='Loss')
plt.title('Validation Accuracy & Loss')
plt.xlabel('Epoch')
plt.figure()
from matplotlib import pyplot as plt


import cv2
imgman = cv2.imread('../input/logokephoto/person.jfif')
plt.imshow(imgman)


imgman = cv2.resize(imgman, (227,227))
imgman = imgman.reshape(1,227,227,3)

pred = classifier.predict_classes(imgman)

if pred==0:
    print("men")
else:
    print("women")



imgwomen = cv2.imread('../input/logokephoto/11.jpg')
plt.imshow(imgwomen)

imgwomen= cv2.resize(imgwomen, (227,227))
imgwomen = imgwomen.reshape(1,227,227,3)

pred = classifier.predict_classes(imgwomen)
if pred==0:
  print("man")
else:
  print("women")

