import numpy as np



import tensorflow as tf

import time

import os
!pip install -q keras
from keras_applications.resnet import ResNet50

from keras.models import Model, load_model

from keras.layers import Dense, GlobalAveragePooling2D, Input, MaxPooling2D, Dropout, Flatten, Conv2D, BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adadelta

import keras

import math, os, sys

import matplotlib.pyplot as plt





def get_model():

    input_tensor = Input(shape=(300, 300, 3))  # this assumes K.image_data_format() == 'channels_last'

    print(input_tensor)

    # create the base pre-trained model

    base_model = ResNet50(input_tensor=input_tensor, include_top=False, weights='imagenet',

        backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)



    for layer in base_model.layers:

        layer.trainable = False



    x = base_model.output

    print(x)

    x = Conv2D(64, (3, 3), activation='sigmoid')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Dense(128, activation='relu')(x)

    x = BatchNormalization()(x)

    x = Dropout(0.5)(x)

    x = GlobalAveragePooling2D(data_format='channels_last')(x)

    x = Dense(num_classes, activation='softmax')(x)



    updatedModel = Model(base_model.input, x)



    return updatedModel





def compile_model(compiledModel):

    compiledModel.compile(loss=keras.losses.categorical_crossentropy,

                          optimizer=keras.optimizers.Adadelta(),

                          metrics=['accuracy'])





def modelFitGenerator(fitModel):

    num_train_samples = sum([len(files) for r, d, files in os.walk(train_data_dir)])

    num_valid_samples = sum([len(files) for r, d, files in os.walk(validation_data_dir)])



    num_train_steps = math.floor(num_train_samples / batch_size)

    num_valid_steps = math.floor(num_valid_samples / batch_size)



    train_datagen = ImageDataGenerator(

        rotation_range=15,

        width_shift_range=.15,

        height_shift_range=.15,

        shear_range=0.15,

        zoom_range=0.15,

        channel_shift_range=1,

        horizontal_flip=True,

        vertical_flip=False)



    test_datagen = ImageDataGenerator(

        rotation_range=15,

        width_shift_range=.15,

        height_shift_range=.15,

        shear_range=0.15,

        zoom_range=0.15,

        channel_shift_range=1,

        horizontal_flip=True,

        vertical_flip=False)



    train_generator = train_datagen.flow_from_directory(

        train_data_dir,

        target_size=image_size,

        batch_size=32,

        class_mode='categorical', shuffle=True,



    )



    validation_generator = test_datagen.flow_from_directory(

        validation_data_dir,

        target_size=image_size,

        batch_size=10,

        class_mode='categorical', shuffle=True

    )



    print("start history model")

    val_checkpoint = keras.callbacks.ModelCheckpoint('models/model.h5', 'val_loss', 1, True)

    cur_checkpoint = keras.callbacks.ModelCheckpoint('models/model_current.h5')

    # don't need to use ReduceLROnOPlateau 

    lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=3, cooldown=1, verbose=2,

                                                    min_lr = 0.0001)



    history = fitModel.fit_generator(

        train_generator,

        steps_per_epoch=num_train_steps,

        epochs=nb_epoch,

        validation_data=validation_generator,

        validation_steps=1,

        callbacks=[val_checkpoint, cur_checkpoint])



    printGraph(history)



    fitModel.save('models/model_last.h5')





def printGraph(history):

    plt.plot(history.history['acc'])

    plt.plot(history.history['val_acc'])

    plt.title('model accuracy')

    plt.ylabel('accuracy')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.show()

    # summarize history for loss

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('model loss')

    plt.ylabel('loss')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.show()











def main():

    model = get_model()

    compile_model(model)

    modelFitGenerator(model)





if __name__ == '__main__':

    # constants

    image_size = (300, 300)

    train_data_dir = 'data_mix_300/train'

    validation_data_dir = 'data_mix_300/valid'

    nb_epoch = 300

    batch_size =32

    num_classes = 10

    validation_steps = 1

    main()