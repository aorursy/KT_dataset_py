import numpy as np

import keras

import matplotlib.pyplot as plt

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, UpSampling2D, GlobalAveragePooling2D

from tensorflow.python.keras.layers import Conv2D, MaxPooling2D

from tensorflow.python.keras.layers.normalization import BatchNormalization

from tensorflow.python.keras.utils import np_utils

from tensorflow.python.keras.datasets import cifar100

from tensorflow.keras.applications.resnet_v2 import ResNet152V2, preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import os

from keras import backend as K
num_classes = 100



(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_val, y_val = x_train[40000:50000,:], y_train[40000:50000]

x_train, y_train = x_train[:50000,:], y_train[:50000]



x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)

x_val = x_val.reshape(x_val.shape[0], 32, 32, 3)

x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)



x_val = preprocess_input(x_val)

x_train = preprocess_input(x_train)

x_test = preprocess_input(x_test)
y_val = np_utils.to_categorical(y_val, num_classes)

y_train = np_utils.to_categorical(y_train, num_classes)

y_test = np_utils.to_categorical(y_test, num_classes)





filepath = '/kaggle/working/bestmodel.h5'



checkpoint = ModelCheckpoint(filepath=filepath,

                             monitor='val_accuracy',

                             verbose=1,

                             save_best_only=True)



stop =  EarlyStopping(monitor='val_accuracy', min_delta=0.001, 

                      verbose=0, mode='auto', baseline=None, 

                      restore_best_weights=False)





callbacks = [checkpoint, stop]



datagen = ImageDataGenerator(

        featurewise_center=False,

        samplewise_center=False,

        featurewise_std_normalization=False,

        samplewise_std_normalization=False,

        zca_whitening=False,

        rotation_range=0,

        width_shift_range=0.1,

        height_shift_range=0.1,

        horizontal_flip=True,

        vertical_flip=False)



datagen.fit(x_train)
resnet_model = ResNet152V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3), classes = 100)



for layer in resnet_model.layers:

    if isinstance(layer, BatchNormalization):

        layer.trainable = True

    else:

        layer.trainable = False



model = Sequential()

model.add(UpSampling2D())

model.add(UpSampling2D())

model.add(UpSampling2D())

model.add(resnet_model)

model.add(GlobalAveragePooling2D())

model.add(Dense(256, activation='relu'))

model.add(Dropout(.25))

model.add(BatchNormalization())

model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])



H = model.fit_generator(datagen.flow(x_train, y_train,

                                  batch_size=64),

                                  steps_per_epoch=x_train.shape[0] // 64,

                                  epochs=20,

                                  verbose=1,

                                  validation_data=(x_val, y_val),

                                  callbacks=callbacks)
plt.figure(figsize=(14, 4))



plt.subplot(1,2,1)

plt.plot(H.history['loss'], 'bo', label='Training loss')

plt.plot(H.history['val_loss'], 'b', label='Validation loss')

plt.title('Training and Validation Loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.subplot(1,2,2)

plt.plot(H.history['accuracy'], 'bo', 

         label='Accuracy', c='orange')

plt.plot(H.history['val_accuracy'], 'b', 

         label='Validation accuracy', c='orange')

plt.title('Training and Validation Accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()
score = model.evaluate(x_test, y_test, verbose=0)

print(score)
model.save_weights('mymodelweights17.h5')
from keras.models import load_model



model.save('mymodel17.h5')