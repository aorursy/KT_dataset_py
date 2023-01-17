# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import os

import shutil



root_train_dir = "/kaggle/input/indian-currency-notes/indian_currency_new/training"

#root_test_dir = "/home/xiaoyzhu/notebooks/currency_detector/data/test"

root_validation_dir = "/kaggle/input/indian-currency-notes/indian_currency_new/training"

#root_visualizaion_dir = "/home/xiaoyzhu/notebooks/currency_detector/data/visualization"

saved_model_file_name = "currency_detector_mobilenet.h5"

tensorboard_dir = "/home/xiaoyzhu/notebooks/currency_detector/data/tensorboard"
import keras

from keras.layers import Dense, Flatten, Activation, BatchNormalization, Dropout

from keras import applications

from keras import optimizers

from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras.layers import Dense

from keras.models import Model

from keras.preprocessing.image import ImageDataGenerator



img_width, img_height = 224,224

train_data_dir = root_train_dir

validation_data_dir = root_validation_dir

nb_train_samples = 1672

nb_validation_samples = 560

train_steps = 70 # 1672 training samples/batch size of 32 = 52 steps. We are doing heavy data processing so put 500 here

validation_steps = 15 # 560 validation samples/batch size of 32 = 10 steps. We put 20 for validation steps

batch_size = 32

epochs = 100



def build_model():

    # constructing the model

    model = applications.mobilenet.MobileNet(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3),

                                  pooling='avg')



    # only train the last 2 layers

    for layer in model.layers[:-10]:

        layer.trainable = False



    # Adding custom Layers

    x = model.output

    # x = Flatten()(x)

    predictions = Dense(8, activation="softmax")(x)



    # creating the final model

    model_final = Model(inputs=model.input, outputs=predictions)

    

    return model_final



model_final = build_model()

# compile the model

model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=0.001), metrics=["accuracy"])



# Initiate the train and test generators with data Augumentation

train_datagen = ImageDataGenerator(

    rescale=1. / 255,

    fill_mode="nearest",

    zoom_range=0.3,

    width_shift_range = 0.2,

    height_shift_range = 0.2,

    shear_range = 0.2,

    rotation_range=180)



validation_datagen = ImageDataGenerator(

    rescale=1. / 255,

    fill_mode="nearest",

    zoom_range=0.3,

    rotation_range=30)



train_generator = train_datagen.flow_from_directory(

    train_data_dir,

    # save_to_dir = root_visualizaion_dir,

    target_size=(img_height, img_width),

    batch_size=batch_size,

    class_mode="categorical")



validation_generator = validation_datagen.flow_from_directory(

    validation_data_dir,

    target_size=(img_height, img_width),

    class_mode="categorical")



# Save the model according to the conditions

checkpoint = ModelCheckpoint("currency_detector_test.h5", monitor='val_loss', verbose=1, save_best_only=True,

                             save_weights_only=False,

                             mode='auto', period=1)

early = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')



# Train the model

model_final.fit_generator(

    train_generator,

    steps_per_epoch = train_steps,

    epochs=epochs,

    validation_data=validation_generator,

    validation_steps = validation_steps,

    workers=16,

    callbacks=[checkpoint, early])