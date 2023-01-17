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
from keras.layers import Dense, Flatten, AveragePooling2D, Dropout

from keras.models import Model

from keras.applications.vgg16 import VGG16

from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator

import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt

%matplotlib inline



from skimage.segmentation import mark_boundaries

import lime

from lime import lime_image

import cv2
data_path = '/kaggle/input/chest-xray-covid19-pneumonia/Data'
baseModel1 = VGG16(input_shape=(300,300,3), weights='imagenet', include_top=False)

baseModel2 = tf.keras.applications.Xception(input_shape=(300,300,3), weights='imagenet', include_top=False)

baseModel3 = tf.keras.applications.InceptionV3(input_shape=(300,300,3), weights='imagenet', include_top=False)

basemodels = [baseModel1,baseModel2,baseModel3]

for i in range(3):

    for layer in basemodels[i].layers:

        layer.trainable = False
x = basemodels[0].output

x = AveragePooling2D()(x)

x = Flatten()(x)

x = Dense(128, activation='relu')(x)

x = Dropout(0.2)(x)

x = Dense(3, activation='softmax')(x)



model1 = Model(basemodels[0].input,x)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

model1.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
x = basemodels[1].output

x = Flatten()(x)

x = Dense(128, activation='relu')(x)

x = Dropout(0.2)(x)

x = Dense(64, activation='relu')(x)

x = Dropout(0.1)(x)

x = Dense(32, activation='relu')(x)

x = Dense(3, activation='softmax')(x)



model2 = Model(basemodels[1].input,x)

model2.compile(optimizer="adam",loss='categorical_crossentropy',metrics=['accuracy'])
x = basemodels[2].output

x = Flatten()(x)

x = Dense(128, activation='relu')(x)

x = Dropout(0.2)(x)

x = Dense(64, activation='relu')(x)

x = Dropout(0.2)(x)

x = Dense(3, activation='softmax')(x)



model3 = Model(basemodels[2].input,x)

model3.compile(optimizer="adam",loss='categorical_crossentropy',metrics=['accuracy'])
train_datagen = ImageDataGenerator(rescale = 1./255,

                                   samplewise_center=True,

                                   zoom_range = 0.2,

                                   rotation_range=15,

                                   width_shift_range=0.2,

                                   height_shift_range=0.2,

                                   shear_range=0.2,

                                   )



test_datagen = ImageDataGenerator(rescale = 1./255)



training_set = train_datagen.flow_from_directory(data_path + '/train',

                                                 target_size = (300,300),

                                                 batch_size = 16,

                                                 class_mode = 'categorical',

                                                 shuffle=True)



test_set = test_datagen.flow_from_directory(data_path + '/test',

                                            target_size = (300,300),

                                            batch_size = 16,

                                            class_mode = 'categorical',

                                            shuffle = False)
from tensorflow.keras.callbacks import ReduceLROnPlateau , ModelCheckpoint, EarlyStopping

lr_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.5, epsilon=0.0001, patience=5, verbose=1)

early = EarlyStopping(monitor='val_acc', mode="max", patience=10)
epochs = 50

filepath="VGG16_weights.h5"

checkpoint1 = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

history1 = model1.fit(training_set,validation_data=test_set,steps_per_epoch=20,callbacks=[lr_reduce,checkpoint1,early] ,

           epochs=epochs)
epochs = 50

filepath= "Xception_weights.h5"

checkpoint2 = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

history2 = model2.fit(training_set, validation_data=test_set,steps_per_epoch=20, callbacks=[lr_reduce,checkpoint2,early],

           epochs=epochs)
epochs = 50

filepath="InceptionV3_weights.h5"

checkpoint3 = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

history3 = model3.fit(training_set,validation_data=test_set,steps_per_epoch=20, callbacks=[lr_reduce,checkpoint3,early] ,

         epochs=epochs)
acc = history1.history['accuracy']

val_acc = history1.history['val_accuracy']

loss = history1.history['loss']

val_loss = history1.history['val_loss']

epochs=range(len(acc))
f, ax = plt.subplots(1)

ax.plot(epochs,acc,label='Trainin_acc',color='blue')

ax.plot(epochs,val_acc,label='Validation_acc',color='red')

ax.legend()

ax.set_ylim(ymin=0.5)

plt.savefig('VGG16_accuracy.png')


f, ax = plt.subplots(1)

ax.plot(epochs,loss,label='Training_loss',color='blue')

ax.plot(epochs,val_loss,label='Validation_acc',color='red')

ax.legend()

ax.set_ylim(ymax=2)

plt.savefig('VGG16_loss.png')
acc = history2.history['accuracy']

val_acc = history2.history['val_accuracy']

loss = history2.history['loss']

val_loss = history2.history['val_loss']

epochs=range(len(acc))
f, ax = plt.subplots(1)

ax.plot(epochs,acc,label='Trainin_acc',color='blue')

ax.plot(epochs,val_acc,label='Validation_acc',color='red')

ax.legend()

ax.set_ylim(ymin=0.5)

plt.savefig('Xception_accuracy.png')
f, ax = plt.subplots(1)

ax.plot(epochs,loss,label='Training_loss',color='blue')

ax.plot(epochs,val_loss,label='Validation_acc',color='red')

ax.legend()

ax.set_ylim(ymax=2)

plt.savefig('Xception_loss.png')
acc = history3.history['accuracy']

val_acc = history3.history['val_accuracy']

loss = history3.history['loss']

val_loss = history3.history['val_loss']

epochs=range(len(acc))
f, ax = plt.subplots(1)

ax.plot(epochs,acc,label='Trainin_acc',color='blue')

ax.plot(epochs,val_acc,label='Validation_acc',color='red')

ax.legend()

ax.set_ylim(ymin=0.5)

plt.savefig('InceptionV3_accuracy.png')
f, ax = plt.subplots(1)

ax.plot(epochs,loss,label='Training_loss',color='blue')

ax.plot(epochs,val_loss,label='Validation_acc',color='red')

ax.legend()

ax.set_ylim(ymax=2)

plt.savefig('InceptionV3_loss.png')