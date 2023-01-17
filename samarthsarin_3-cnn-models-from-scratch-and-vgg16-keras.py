# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn

from keras.models import Sequential

from keras.layers import Convolution2D,MaxPooling2D,Dropout,Dense,Flatten,BatchNormalization,Conv2D

from keras.preprocessing.image import ImageDataGenerator

from keras.applications import VGG16

from keras.callbacks import ModelCheckpoint

%matplotlib inline

import glob

import cv2
infected = glob.glob('../input/cell_images/cell_images/Parasitized/*.png')

uninfected = glob.glob('../input/cell_images/cell_images/Uninfected/*.png')
print('Total number of Infected Cell Images are ',len(infected),' shape of first image is ',cv2.imread(infected[0]).shape)

print('Total number of Uninfected Cell Images are ',len(uninfected),' shape of the first image is ',cv2.imread(uninfected[0]).shape)
plt.figure(figsize=(12,5))

for i in range(1,5):

    plt.subplot(1,4,i)

    value = np.random.randint(100)

    image = cv2.imread(infected[value])

    plt.imshow(image)

    plt.title('Infected Image')

    plt.xticks([])

    plt.yticks([])
plt.figure(figsize=(12,5))

for i in range(1,5):

    plt.subplot(1,4,i)

    value = np.random.randint(100)

    image = cv2.imread(uninfected[value])

    plt.imshow(image)

    plt.title('Uninfected Image')

    plt.xticks([])

    plt.yticks([])
augmentor = ImageDataGenerator(rescale=1./255,zoom_range=0.2,shear_range=0.2,horizontal_flip=True,validation_split=0.2)
train_generator = augmentor.flow_from_directory('../input/cell_images/cell_images/',batch_size=64,

                                                target_size = (96,96),class_mode = 'binary',subset = 'training')

test_generator = augmentor.flow_from_directory('../input/cell_images/cell_images/',batch_size=64,target_size=(96,96),

                                              class_mode='binary',subset='validation')
model1 = Sequential()

model1.add(Convolution2D(32,(3,3),activation='relu',input_shape = (96,96,3)))

model1.add(BatchNormalization())

model1.add(MaxPooling2D(2,2))

model1.add(Dropout(0.2))

model1.add(Convolution2D(32,(3,3),activation='relu'))

model1.add(BatchNormalization())

model1.add(MaxPooling2D(2,2))

model1.add(Dropout(0.2))

model1.add(Convolution2D(64,(3,3),activation='relu'))

model1.add(BatchNormalization())

model1.add(MaxPooling2D(2,2))

model1.add(Dropout(0.2))

model1.add(Flatten())

model1.add(Dense(64,activation='relu'))

model1.add(Dropout(0.2))

model1.add(Dense(1,activation='sigmoid'))

model1.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model1.summary()
history_custom = model1.fit_generator(train_generator,steps_per_epoch=2000,

                              epochs = 5,validation_data=test_generator,validation_steps=64)
values  = history_custom.history

validation_loss = values['val_loss']

validation_acc = values['val_acc']

training_acc = values['acc']

training_loss = values['acc']

epochs = range(5)
plt.plot(epochs,training_loss,label = 'Training Loss')

plt.plot(epochs,validation_loss,label = 'Validation Loss')

plt.title('Epochs vs Loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
plt.plot(epochs,training_acc,label = 'Training Accuracy')

plt.plot(epochs,validation_acc,label = 'Validation Accuracy')

plt.title('Epochs vs Accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
model2 = Sequential()

model2.add(Convolution2D(32,(5,5),activation='relu',input_shape = (96,96,3)))

model2.add(BatchNormalization())

model2.add(MaxPooling2D(2,2))

model2.add(Dropout(0.2))

model2.add(Convolution2D(32,(5,5),activation='relu'))

model2.add(BatchNormalization())

model2.add(MaxPooling2D(2,2))

model2.add(Dropout(0.2))

model2.add(Convolution2D(64,(5,5),activation='relu'))

model2.add(BatchNormalization())

model2.add(MaxPooling2D(2,2))

model2.add(Dropout(0.2))

model2.add(Flatten())

model2.add(Dense(64,activation='relu'))

model2.add(Dropout(0.2))

model2.add(Dense(1,activation='sigmoid'))

model2.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model2.summary()
history_custom = model2.fit_generator(train_generator,steps_per_epoch=2000,

                              epochs = 5,validation_data=test_generator,validation_steps=64)
values  = history_custom.history

validation_loss = values['val_loss']

validation_acc = values['val_acc']

training_acc = values['acc']

training_loss = values['acc']

epochs = range(5)
plt.plot(epochs,training_loss,label = 'Training Loss')

plt.plot(epochs,validation_loss,label = 'Validation Loss')

plt.title('Epochs vs Loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
plt.plot(epochs,training_acc,label = 'Training Accuracy')

plt.plot(epochs,validation_acc,label = 'Validation Accuracy')

plt.title('Epochs vs Accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
model3 = Sequential()

model3.add(Convolution2D(32,(5,5),activation='tanh',input_shape = (96,96,3)))

model3.add(BatchNormalization())

model3.add(MaxPooling2D(2,2))

model3.add(Dropout(0.2))

model3.add(Convolution2D(32,(5,5),activation='tanh'))

model3.add(BatchNormalization())

model3.add(MaxPooling2D(2,2))

model3.add(Dropout(0.2))

model3.add(Convolution2D(64,(5,5),activation='tanh'))

model3.add(BatchNormalization())

model3.add(MaxPooling2D(2,2))

model3.add(Dropout(0.2))

model3.add(Flatten())

model3.add(Dense(64,activation='tanh'))

model3.add(Dropout(0.2))

model3.add(Dense(1,activation='sigmoid'))

model3.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model3.summary()
history_custom = model2.fit_generator(train_generator,steps_per_epoch=2000,

                              epochs = 5,validation_data=test_generator,validation_steps=64)
values  = history_custom.history

validation_loss = values['val_loss']

validation_acc = values['val_acc']

training_acc = values['acc']

training_loss = values['acc']

epochs = range(5)
plt.plot(epochs,training_loss,label = 'Training Loss')

plt.plot(epochs,validation_loss,label = 'Validation Loss')

plt.title('Epochs vs Loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
plt.plot(epochs,training_acc,label = 'Training Accuracy')

plt.plot(epochs,validation_acc,label = 'Validation Accuracy')

plt.title('Epochs vs Accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
vgg16 = VGG16(weights = 'imagenet',include_top = False,input_shape = (96,96,3))
vgg16.summary()
for layers in vgg16.layers[:-4]:

    layers.trainable = False
model = Sequential()

model.add(vgg16)

model.add(Flatten())

model.add(Dense(64,activation='relu'))

model.add(Dense(1,activation = 'sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
callback = ModelCheckpoint('model_vgg16.h5',monitor='val_acc',mode = 'max',save_best_only=True)

calls = [callback]
history = model.fit_generator(train_generator,

                              steps_per_epoch=2000,

                              epochs=5,

                              validation_data=test_generator,

                              validation_steps=64,

                              callbacks = calls)
values  = history.history

validation_loss = values['val_loss']

validation_acc = values['val_acc']

training_acc = values['acc']

training_loss = values['acc']

epochs = range(5)
plt.plot(epochs,training_loss,label = 'Training Loss')

plt.plot(epochs,validation_loss,label = 'Validation Loss')

plt.title('Epochs vs Loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
plt.plot(epochs,training_acc,label = 'Training Accuracy')

plt.plot(epochs,validation_acc,label = 'Validation Accuracy')

plt.title('Epochs vs Accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()