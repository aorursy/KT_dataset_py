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
import keras

print(keras.__version__)
from keras.applications.xception import Xception 

from keras.layers import *

from keras.optimizers import *

from keras.applications import *

from keras.models import Model

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint, EarlyStopping

# hyper parameters for model

nb_classes = 4  # number of classes

img_width, img_height = 299, 299  # change based on the shape/structure of your images

batch_size = 4  # try 4, 8, 16, 32, 64, 128, 256 dependent on CPU/GPU memory capacity (powers of 2 values).

nb_epoch = 1  # number of iteration the algorithm gets trained.

learn_rate = 1e-3  # sgd learning rate

momentum = .9  # sgd momentum to avoid local minimum



def train(train_data_dir, validation_data_dir, model_path):

    # Pre-Trained CNN Model using imagenet dataset for pre-trained weights

    base_model = Xception(input_shape=(img_width, img_height, 3), weights='imagenet', include_top=False)



    # Top Model Block

    x = base_model.output

    x = GlobalAveragePooling2D()(x)

    predictions = Dense(nb_classes, activation='softmax')(x)



    # add your top layer block to your base model

    model = Model(base_model.input, predictions)

    print(model.summary())

    

    datagen = ImageDataGenerator(rescale=1. / 255)



    train_generator = datagen.flow_from_directory(train_data_dir,

                                                        target_size=(img_width, img_height),

                                                        batch_size=batch_size,

                                                        class_mode='categorical')

   

    validation_generator = datagen.flow_from_directory(validation_data_dir,

                                                                  target_size=(img_width, img_height),

                                                                  batch_size=batch_size,

                                                                  class_mode='categorical')



    # verbose

    print("\nStarting to Fine Tune Model\n")



    # compile the model with a SGD/momentum optimizer

    # and a very slow learning rate.

    model.compile(optimizer='nadam',

                  loss='categorical_crossentropy',

                  metrics=['accuracy'])



    # save weights of best training epoch: monitor either val_loss or val_acc

    final_weights_path = os.path.join(os.path.abspath(model_path), 'model_weights.h5')

    callbacks_list = [

        ModelCheckpoint(final_weights_path, monitor='val_acc', verbose=1, save_best_only=True),

        EarlyStopping(monitor='val_loss', patience=5, verbose=0)

    ]

  

    # fine-tune the model

    history = model.fit_generator(train_generator,

                        samples_per_epoch=train_generator.samples,

                        nb_epoch=nb_epoch,

                        validation_data=validation_generator,

                        nb_val_samples=validation_generator.samples,

                        callbacks=callbacks_list)



    # save model

    model_json = model.to_json()

    with open(os.path.join(os.path.abspath(model_path), 'model.json'), 'w') as json_file:

        json_file.write(model_json)

    return history

import matplotlib.pyplot as plt

def show_history(history, title):

    plt.title(title)

    plt.plot(history.history['acc'])

    plt.plot(history.history['val_acc'])

    plt.ylabel('accuracy')

    plt.xlabel('epoch')

    plt.legend(['train_accuracy', 'validation_accuracy'], loc='best')

    plt.savefig('Accuracy')

import os

train_dir = '../input/dokurek/data/train'

validation_dir = '../input/dokurek/data/validation'

model_dir = '.'

history = train(train_dir, validation_dir, model_dir)

show_history(history, 'BS=4, LR=1e-3, NEP=1')