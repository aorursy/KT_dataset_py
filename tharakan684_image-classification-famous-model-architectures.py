import os

import keras

import cv2

import h5py

import matplotlib.image as mpimg

import matplotlib.pyplot as plt

from tensorflow.keras import regularizers

import tensorflow as tf

from tensorflow.keras.optimizers import RMSprop, Adam

import numpy as np

import random

from   tensorflow.keras.preprocessing.image import img_to_array, load_img

import os

from glob import glob

from keras.models import Model

from keras.layers import Input, Dense, GlobalMaxPool2D, GlobalAvgPool2D, Concatenate, Multiply, Dropout, Subtract

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from keras.preprocessing import image
model = tf.keras.models.Sequential([



tf.keras.layers.Conv2D(16, (3,3), activation='relu',padding = 'same', input_shape=(224, 224, 3)),

#tf.keras.layers.MaxPooling2D(2, 2),



tf.keras.layers.Conv2D(32, (3,3), activation='relu'),

tf.keras.layers.MaxPooling2D(2,2),



tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

tf.keras.layers.MaxPooling2D(2,2),



tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

tf.keras.layers.MaxPooling2D(2,2),



tf.keras.layers.Conv2D(128, (3,3), activation='relu'),

tf.keras.layers.MaxPooling2D(2,2),



tf.keras.layers.Flatten(),



tf.keras.layers.Dense(1024, activation='tanh'),

tf.keras.layers.Dropout(0.01),

tf.keras.layers.Dense(512, activation='tanh',kernel_regularizer=regularizers.l2(0.01)),

tf.keras.layers.Dropout(0.01),

#tf.keras.layers.Dense(128, activation='relu'),

tf.keras.layers.Dense(32, activation='tanh'),



tf.keras.layers.Dense(1, activation='sigmoid')

])

model.compile(loss='binary_crossentropy',

          optimizer=Adam(lr=0.001),

          metrics=['accuracy'])



model.summary()
from tensorflow.keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(

        rescale=1./255,

        horizontal_flip=True)

vali_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(

        '../input/urecamain/Train',

        target_size=(224, 224),

        batch_size=64,

        classes = ['Fire','Non-Fire'],

        shuffle = True,

        

        class_mode='binary')

validation_generator = vali_datagen.flow_from_directory(

        '../input/urecamain/Vali',

        target_size=(224, 224),

        batch_size=64,

        classes = ['Fire','Non-Fire'],

        shuffle = True,

        class_mode='binary')

test_generator = test_datagen.flow_from_directory(

        '../input/urecamain/Test',

        target_size=(224, 224),

        batch_size=64,

        classes = ['Fire','Non-Fire'],

        shuffle = True,

        class_mode='binary')

file_path = "./Final1.h5"

checkpoint = ModelCheckpoint(file_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

reduce_on_plateau = ReduceLROnPlateau(monitor="loss", mode="min", factor=0.1, patience=5, verbose=1)

es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=8)

callbacks_list = [ checkpoint, reduce_on_plateau, es]



curr_model_hist = model.fit(

      train_generator,

      callbacks=callbacks_list,

      epochs=100,

      validation_data=validation_generator,

      verbose=1)
import matplotlib.pyplot as plt

def plot_accuracy(y):

    if(y == True):

        plt.plot(curr_model_hist.history['accuracy'])

        plt.plot(curr_model_hist.history['val_accuracy'])

        plt.legend(['train', 'validation'], loc='lower right')

        plt.title('Accuracy plot - train vs validation')

        plt.xlabel('epoch')

        plt.ylabel('accuracy')

        plt.show()

    else:

        pass

    return



def plot_loss(y):

    if(y == True):

        plt.plot(curr_model_hist.history['loss'])

        plt.plot(curr_model_hist.history['val_loss'])

        plt.legend(['training', 'validation'], loc = 'upper right')

        plt.title('Loss plot - train vs vaidation')

        plt.xlabel('epoch')

        plt.ylabel('loss')

        plt.show()

    else:

        pass

    return





plot_accuracy(True)

plot_loss(True)
lossandacc = model.evaluate_generator(test_generator,verbose=1)

print(lossandacc)
from tensorflow.keras.applications import MobileNetV2

from tensorflow.keras.layers import Dense

from tensorflow.keras.models import Model



def get_model():

    base = MobileNetV2(input_shape=(224,224,3), include_top=False, \

                       pooling='max', weights='imagenet')

    base.trainable = False

    dense = Dense(1, activation='sigmoid', name='dense')(base.output)



    model = Model(inputs=base.inputs, outputs=dense, name='mobilenetv2')

    model.compile(loss='binary_crossentropy',

          optimizer=Adam(lr=0.001),

          metrics=['acc'])

    return model



mobilenet = get_model()



file_path = '/mobilenet.h5'

checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

reduce_on_plateau = ReduceLROnPlateau(monitor="loss", mode="min", factor=0.1, patience=5, verbose=1)

es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=8)

callbacks_list = [ checkpoint, reduce_on_plateau, es]

hist = mobilenet.fit(train_generator,

      callbacks=callbacks_list,

      epochs=100,

      validation_data=validation_generator,

      verbose=1)
lossandacc = mobilenet.evaluate_generator(test_generator,verbose=1)

print(lossandacc)


from tensorflow.keras.applications import InceptionResNetV2

from tensorflow.keras.layers import Dense

from tensorflow.keras.models import Model



def get_model():

    base = InceptionResNetV2(input_shape=(224,224,3), include_top=False, \

                       pooling='max', weights='imagenet')

    base.trainable = False

    dense = Dense(1, activation='sigmoid', name='dense')(base.output)



    model = Model(inputs=base.inputs, outputs=dense, name='InceptionResNetV2')

    model.compile(loss='binary_crossentropy',

          optimizer=Adam(lr=0.001),

          metrics=['acc'])

    return model



InceptionResNetV2 = get_model()



file_path = '/InceptionResNetV2.h5'

checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

reduce_on_plateau = ReduceLROnPlateau(monitor="loss", mode="min", factor=0.1, patience=5, verbose=1)

es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=8)

callbacks_list = [ checkpoint, reduce_on_plateau, es]

hist = InceptionResNetV2.fit(train_generator,

      callbacks=callbacks_list,

      epochs=100,

      validation_data=validation_generator,

      verbose=1)

lossandacc = InceptionResNetV2.evaluate_generator(test_generator,verbose=1)

print(lossandacc)
from tensorflow.keras.applications import ResNet152

from tensorflow.keras.layers import Dense

from tensorflow.keras.models import Model



def get_model():

    base = ResNet152(input_shape=(224,224,3), include_top=False, \

                       pooling='max', weights='imagenet')

    base.trainable = False

    dense = Dense(1, activation='sigmoid', name='dense')(base.output)



    model = Model(inputs=base.inputs, outputs=dense, name='ResNet152')

    model.compile(loss='binary_crossentropy',

          optimizer=Adam(lr=0.001),

          metrics=['acc'])

    return model



ResNet152 = get_model()



file_path = '/ResNet152.h5'

checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

reduce_on_plateau = ReduceLROnPlateau(monitor="loss", mode="min", factor=0.1, patience=5, verbose=1)

es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=8)

callbacks_list = [ checkpoint, reduce_on_plateau, es]

hist = ResNet152.fit(train_generator,

      callbacks=callbacks_list,

      epochs=100,

      validation_data=validation_generator,

      verbose=1)
lossandacc = ResNet152.evaluate_generator(test_generator,verbose=1)

print(lossandacc)
from tensorflow.keras.applications import VGG19

from tensorflow.keras.layers import Dense

from tensorflow.keras.models import Model



def get_model():

    base = VGG19(input_shape=(224,224,3), include_top=False, \

                       pooling='max', weights='imagenet')

    base.trainable = False

    dense = Dense(1, activation='sigmoid', name='dense')(base.output)



    model = Model(inputs=base.inputs, outputs=dense, name='VGG19')

    model.compile(loss='binary_crossentropy',

          optimizer=Adam(lr=0.001),

          metrics=['acc'])

    return model



VGG19 = get_model()



file_path = '/VGG19.h5'

checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

reduce_on_plateau = ReduceLROnPlateau(monitor="loss", mode="min", factor=0.1, patience=5, verbose=1)

es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=8)

callbacks_list = [ checkpoint, reduce_on_plateau, es]

hist = VGG19.fit(train_generator,

      callbacks=callbacks_list,

      epochs=100,

      validation_data=validation_generator,

      verbose=1)
lossandacc = VGG19.evaluate_generator(test_generator,verbose=1)

print(lossandacc)
from tensorflow.keras.applications import ResNet50

from tensorflow.keras.layers import Dense

from tensorflow.keras.models import Model



def get_model():

    base = ResNet50(input_shape=(224,224,3), include_top=False, \

                       pooling='max', weights='imagenet')

    base.trainable = False

    dense = Dense(1, activation='sigmoid', name='dense')(base.output)



    model = Model(inputs=base.inputs, outputs=dense, name='ResNet50')

    model.compile(loss='binary_crossentropy',

          optimizer=Adam(lr=0.001),

          metrics=['acc'])

    return model



ResNet50 = get_model()



file_path = '/EfficientNetB7.h5'

checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

reduce_on_plateau = ReduceLROnPlateau(monitor="loss", mode="min", factor=0.1, patience=5, verbose=1)

es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=8)

callbacks_list = [ checkpoint, reduce_on_plateau, es]

hist = ResNet50.fit(train_generator,

      callbacks=callbacks_list,

      epochs=100,

      validation_data=validation_generator,

      verbose=1)
lossandacc = ResNet50.evaluate_generator(test_generator,verbose=1)

print(lossandacc)
from IPython.display import FileLink

FileLink('./Final1.h5')