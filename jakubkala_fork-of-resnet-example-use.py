import tensorflow as tf



class ResnetIdentityBlock(tf.keras.Model):

  def __init__(self, kernel_size, filters, batch_normalization=True, conv_first=False):

    super(ResnetIdentityBlock, self).__init__(name='')

    

    self.residual_layers = []

    

    for i in range(len(filters)):

        

        if conv_first:

            setattr(self, 'conv' + str(i+1), tf.keras.layers.Conv2D(filters[i], kernel_size, padding='same'))

            self.residual_layers.append('conv' + str(i+1))



            if batch_normalization:

                setattr(self, 'bn' + str(i+1), tf.keras.layers.BatchNormalization())

                self.residual_layers.append('bn' + str(i+1))

        

        else:

            if batch_normalization:

                setattr(self, 'bn' + str(i+1), tf.keras.layers.BatchNormalization())

                self.residual_layers.append('bn' + str(i+1))

            

            setattr(self, 'conv' + str(i+1), tf.keras.layers.Conv2D(filters[i], kernel_size, padding='same'))

            self.residual_layers.append('conv' + str(i+1))



            

            

  def call(self, input_tensor, training=False):

    

    x = input_tensor

    

    for layer in self.residual_layers:

        if isinstance(layer, tf.keras.layers.Conv2D):

            x = getattr(self, layer)(x)

        else: 

            x = getattr(self, layer)(x, training=False)

        x = tf.nn.relu(x)

        

    x += input_tensor

    return tf.nn.relu(x)
from keras.datasets import cifar10



(x_train,y_train),(x_test,y_test) = cifar10.load_data() 



x_train = x_train /255

x_test = x_test / 255
from keras.utils.np_utils import to_categorical   



y_train = to_categorical(y_train, num_classes=10)

y_test = to_categorical(y_test, num_classes=10)
import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator



#from src.scripts.residuallayer import ResnetIdentityBlock
datagen = ImageDataGenerator(

        # set input mean to 0 over the dataset

        featurewise_center=False,

        # set each sample mean to 0

        samplewise_center=False,

        # divide inputs by std of dataset

        featurewise_std_normalization=False,

        # divide each input by its std

        samplewise_std_normalization=False,

        # apply ZCA whitening

        zca_whitening=False,

        # epsilon for ZCA whitening

        zca_epsilon=1e-06,

        # randomly rotate images in the range (deg 0 to 180)

        rotation_range=5,

        # randomly shift images horizontally

        width_shift_range=0.1,

        # randomly shift images vertically

        height_shift_range=0.1,

        # set range for random shear

        shear_range=0.,

        # set range for random zoom

        zoom_range=0.,

        # set range for random channel shifts

        channel_shift_range=0.,

        # set mode for filling points outside the input boundaries

        fill_mode='nearest',

        # value used for fill_mode = "constant"

        cval=0.,

        # randomly flip images

        horizontal_flip=True,

        # randomly flip images

        vertical_flip=False,

        # set rescaling factor (applied before any other transformation)

        rescale=None,

        # set function that will be applied on each input

        preprocessing_function=None,

        # image data format, either "channels_first" or "channels_last"

        data_format=None,

        # fraction of images reserved for validation (strictly between 0 and 1)

        validation_split=0)
NUM_CLASSES = 10

INPUT_SHAPE = (32, 32, 3)



NUM_EPOCHS = 100

learning_rate = 1e-4

BATCH_SIZE=128
model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=INPUT_SHAPE))



model.add(ResnetIdentityBlock((3,3), filters=(128, 128)))

model.add(ResnetIdentityBlock((3,3), filters=(128, 128)))

model.add(ResnetIdentityBlock((3,3), filters=(128, 128)))

model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same'))



model.add(ResnetIdentityBlock((2,2), filters=(64, 64)))

model.add(ResnetIdentityBlock((2,2), filters=(64, 64)))

model.add(ResnetIdentityBlock((2,2), filters=(64, 64)))



model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Activation('relu'))

model.add(tf.keras.layers.AveragePooling2D(pool_size=8))

model.add(tf.keras.layers.Flatten())



model.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))
model.summary()



datagen.fit(x_train)



adam = tf.keras.optimizers.Adam(learning_rate)



model.compile(optimizer=adam,

              loss='categorical_crossentropy',

              metrics=['accuracy'])



model.fit_generator(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),

                    validation_data=(x_test, y_test),

                    epochs=NUM_EPOCHS)
history2 = model.history.history



# summarize history for accuracy

plt.plot(history2['accuracy'])

plt.plot(history2['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()



# summarize history for loss

plt.plot(history2['loss'])

plt.plot(history2['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', input_shape=INPUT_SHAPE))



model.add(ResnetIdentityBlock((3,3), filters=(256, 256)))

model.add(ResnetIdentityBlock((3,3), filters=(256, 256)))

model.add(ResnetIdentityBlock((3,3), filters=(256, 256)))

model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same'))





model.add(ResnetIdentityBlock((3,3), filters=(128, 128)))

model.add(ResnetIdentityBlock((3,3), filters=(128, 128)))

model.add(ResnetIdentityBlock((3,3), filters=(128, 128)))

model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same'))



model.add(ResnetIdentityBlock((2,2), filters=(64, 64)))

model.add(ResnetIdentityBlock((2,2), filters=(64, 64)))

model.add(ResnetIdentityBlock((2,2), filters=(64, 64)))



model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Activation('relu'))

model.add(tf.keras.layers.AveragePooling2D(pool_size=8))

model.add(tf.keras.layers.Flatten())



model.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))
model.summary()



datagen.fit(x_train)



adam = tf.keras.optimizers.Adam(learning_rate)



model.compile(optimizer=adam,

              loss='categorical_crossentropy',

              metrics=['accuracy'])



model.fit_generator(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),

                    validation_data=(x_test, y_test),

                    epochs=NUM_EPOCHS)
history1 = model.history.history



# summarize history for accuracy

plt.plot(history1['accuracy'])

plt.plot(history1['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()



# summarize history for loss

plt.plot(history1['loss'])

plt.plot(history1['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=INPUT_SHAPE))



model.add(ResnetIdentityBlock((2,2), filters=(64, 64)))

model.add(ResnetIdentityBlock((2,2), filters=(64, 64)))

model.add(ResnetIdentityBlock((2,2), filters=(64, 64)))



model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Activation('relu'))

model.add(tf.keras.layers.AveragePooling2D(pool_size=8))

model.add(tf.keras.layers.Flatten())



model.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))
model.summary()



datagen.fit(x_train)



adam = tf.keras.optimizers.Adam(lr=learning_rate)



model.compile(optimizer=adam,

              loss='categorical_crossentropy',

              metrics=['accuracy'])



model.fit_generator(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),

                    validation_data=(x_test, y_test),

                    epochs=NUM_EPOCHS)
history3 = model.history.history



# summarize history for accuracy

plt.plot(history3['accuracy'])

plt.plot(history3['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()



# summarize history for loss

plt.plot(history3['loss'])

plt.plot(history3['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
# save history

import json



with open('history1.txt', 'w') as file:

     file.write(json.dumps(str(history1)))



with open('history2.txt', 'w') as file:

     file.write(json.dumps(str(history2)))

        

with open('history3.txt', 'w') as file:

     file.write(json.dumps(str(history3)))