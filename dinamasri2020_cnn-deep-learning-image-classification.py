# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import os        

import numpy as np # linear algebra

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import PIL

import PIL.Image

from tensorflow import keras

import tensorflow_datasets as tfds

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory







# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
## Trains a model to classify images of 3 classes: cat, dog, and panda

def train_test_animals():

    

    

    # Defines & compiles the model

    model = tf.keras.models.Sequential([

        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),

    tf.keras.layers.MaxPooling2D(2, 2),

    keras.layers.Dropout(rate=0.2), #adding dropout regularization throughout the model to deal with overfitting

    # The second convolution

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    keras.layers.Dropout(rate=0.15),

    # The third convolution

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    keras.layers.Dropout(rate=0.1),

    # Flatten the results to feed into a DNN

    tf.keras.layers.Flatten(),

    # 512 neuron hidden layer

    tf.keras.layers.Dense(512, activation='relu'),

    keras.layers.Dropout(rate=0.1),

    # 3 output neuron for the 3 classes of Animal Images

    tf.keras.layers.Dense(3, activation='softmax')

    ])



    from tensorflow.keras.optimizers import RMSprop



    model.compile(loss='categorical_crossentropy',

              optimizer="adam",

              metrics=['acc'])

        



    # Creates an instance of an ImageDataGenerator called train_datagen, and a train_generator, train_datagen.flow_from_directory



    from tensorflow.keras.preprocessing.image import ImageDataGenerator



    #splits data into training and testing(validation) sets

    train_datagen =ImageDataGenerator(rescale=1./255, validation_split=0.25)

    

    import matplotlib.pyplot as plt



   

    #training data

    train_generator = train_datagen.flow_from_directory(

        '/kaggle/input/animal-image-datasetdog-cat-and-panda/animals/animals',  # Source directory

        target_size=(150, 150),  # Resizes images

        batch_size=15,

        class_mode='categorical',subset = 'training')

    



    epochs = 15

    #Testing data

    validation_generator = train_datagen.flow_from_directory(

    '/kaggle/input/animal-image-datasetdog-cat-and-panda/animals/animals',

    target_size=(150, 150),

    batch_size=15,

    class_mode='categorical',

    subset='validation') # set as validation data

       

    #Model fitting for a number of epochs

    history = model.fit_generator(

      train_generator,

      steps_per_epoch=150,

      epochs=epochs,

      validation_data = validation_generator,

      validation_steps = 50,

      verbose=1)

    

        

    

    acc = history.history['acc']

    val_acc = history.history['val_acc']



    loss = history.history['loss']

    val_loss = history.history['val_loss']



    #This code is used to plot the training and validation accuracy

    epochs_range = range(epochs)



    plt.figure(figsize=(8, 8))

    plt.subplot(1, 2, 1)

    plt.plot(epochs_range, acc, label='Training Accuracy')

    plt.plot(epochs_range, val_acc, label='Validation Accuracy')

    plt.legend(loc='lower right')

    plt.title('Training and Validation Accuracy')



    plt.subplot(1, 2, 2)

    plt.plot(epochs_range, loss, label='Training Loss')

    plt.plot(epochs_range, val_loss, label='Validation Loss')

    plt.legend(loc='upper right')

    plt.title('Training and Validation Loss')

    plt.show()

 

    # returns accuracy of training

    print("Training Accuracy:"), print(history.history['val_acc'][-1])

    print("Testing Accuracy:"), print (history.history['acc'][-1])



    
train_test_animals()