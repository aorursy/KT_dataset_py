import tensorflow as tf

from tensorflow.python import keras



print('Tensorflow Version: ', tf.__version__)

print('Keras Version: ', keras.__version__)
import os

import numpy as np

import keras

from keras import models

from keras import layers

from keras.preprocessing.image import ImageDataGenerator

from keras import regularizers

from keras import optimizers

from keras.callbacks import ModelCheckpoint



pneumoniaNetModel=models.Sequential()



pneumoniaNetModel.add(layers.SeparableConv2D(32, 3, activation='relu', input_shape=(150,150,3)))

pneumoniaNetModel.add(layers.SeparableConv2D(64, 3, activation='relu'))

pneumoniaNetModel.add(layers.MaxPooling2D(2))



pneumoniaNetModel.add(layers.SeparableConv2D(64, 3, activation='relu'))

pneumoniaNetModel.add(layers.SeparableConv2D(128, 3, activation='relu'))

pneumoniaNetModel.add(layers.MaxPooling2D(2))



pneumoniaNetModel.add(layers.SeparableConv2D(64, 3, activation='relu'))

pneumoniaNetModel.add(layers.SeparableConv2D(128, 3, activation='relu'))

pneumoniaNetModel.add(layers.GlobalAveragePooling2D())



pneumoniaNetModel.add(layers.Dense(32, activation='relu'))

pneumoniaNetModel.add(layers.Dense(2, activation='softmax'))



pneumoniaNetModel.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['categorical_accuracy'])

pneumoniaNetModel.summary()



filepath="PneumoniaNet8.h5"

checkpoint = ModelCheckpoint(filepath, save_best_only=True)

callbacks_list = [checkpoint]
image_height = 150

image_width = 150

batch_size = 8

no_of_epochs  = 60

number_of_training_samples=5216

number_of_validation_samples=16

number_of_test_samples=624
train_dir='/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train'

validation_dir='/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/val'

test_dir='/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/test'
train_datagen = ImageDataGenerator(rescale=1./255,

                                   rotation_range=40,

                                   width_shift_range=0.2,

                                   height_shift_range=0.2,

                                   shear_range=0.2,

                                   zoom_range=0.2

                                   )



validation_datagen = ImageDataGenerator(rescale=1./255)  



test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(train_dir,target_size=(image_width, image_height),batch_size=batch_size)

validation_set = validation_datagen.flow_from_directory(validation_dir,target_size=(image_width, image_height),batch_size=batch_size,shuffle=False)

test_set = test_datagen.flow_from_directory(test_dir,target_size=(image_width, image_height),batch_size=batch_size,shuffle=False)
import math



history = pneumoniaNetModel.fit_generator(

      training_set,

      steps_per_epoch=math.ceil(number_of_training_samples//batch_size),

      epochs=no_of_epochs,

      callbacks=callbacks_list,

      validation_data=validation_set,

      validation_steps=math.ceil(number_of_validation_samples//batch_size))
import matplotlib.pyplot as plt

%matplotlib inline



acc=history.history['categorical_accuracy']

val_acc=history.history['val_categorical_accuracy']

loss=history.history['loss']

val_loss=history.history['val_loss']

epochs=range(1,len(acc)+1)

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()



plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
from tensorflow.python.keras.models import load_model



best_model = load_model('PneumoniaNet8.h5')
steps_test=int(number_of_test_samples/batch_size)

result = best_model.evaluate_generator(test_set, steps=steps_test,verbose=1)

print("Test-set classification accuracy: {0:.2%}".format(result[1]))