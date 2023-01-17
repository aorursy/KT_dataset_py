# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os        

import numpy as np # linear algebra

import matplotlib.pyplot as plt

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from tensorflow import keras



import os

for dirname, _, filenames in os.walk('../input/dogs-cats-images/dataset/training_set'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



def train_cats_dogs():

    # Please write your code only where you are indicated.

    # please do not remove # model fitting inline comments.



    DESIRED_ACCURACY = 0.95



    class myCallback(tf.keras.callbacks.Callback):

        def on_epoch_end(self, epoch, logs={}):

            if(logs.get('acc')>DESIRED_ACCURACY):

                print("\nReached 95% accuracy so cancelling training!")

                self.model.stop_training = True



    callbacks = myCallback()

    

    # This Code Block should Define and Compile the Model

    model = tf.keras.models.Sequential([

        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),

    tf.keras.layers.MaxPooling2D(2, 2),

    # The second convolution

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    # The third convolution

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    # Flatten the results to feed into a DNN

    tf.keras.layers.Flatten(),

    # 512 neuron hidden layer

    tf.keras.layers.Dense(512, activation='relu'),

    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')

    tf.keras.layers.Dense(1, activation='sigmoid')

    ])



    from tensorflow.keras.optimizers import RMSprop



    model.compile(loss='binary_crossentropy',

              optimizer=RMSprop(lr=0.001),

              metrics=['acc'])

        



    # This code block should create an instance of an ImageDataGenerator called train_datagen 

    # And a train_generator by calling train_datagen.flow_from_directory



    from tensorflow.keras.preprocessing.image import ImageDataGenerator



    train_datagen = ImageDataGenerator(rescale=1/255)



    train_generator = train_datagen.flow_from_directory(

        '../input/dogs-cats-images/dataset/training_set',  # This is the source directory for training images

        target_size=(150, 150),  # All images will be resized to 150x150

        batch_size=25,

        # Since we use binary_crossentropy loss, we need binary labels

        class_mode='binary')

        # Your Code Here)

    # Expected output: 'Found 80 images belonging to 2 classes'

    # This code block should call model.fit_generator and train for

    # a number of epochs.

    # model fitting

    history = model.fit_generator(

      train_generator,

      steps_per_epoch=320,  

      epochs=10,

      verbose=1, callbacks=[callbacks])

        # Your Code Here)

    # model fitting

    return history.history['acc'][-1]
train_cats_dogs()