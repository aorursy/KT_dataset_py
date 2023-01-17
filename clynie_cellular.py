# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.image as mpimg



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

print(os.listdir('/kaggle/input/'))

# for dirname, _, filenames in os.walk('/kaggle/input/'):

#     print(dirname)

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



print(os.listdir('/kaggle/input/train/HUVEC-06/'))
# img = mpimg.imread('/kaggle/input/train/HUVEC-06/Plate1/O16_s2_w6.png')

# plt.imshow(img)

import csv

df = pd.read_csv('/kaggle/input/train.csv')

df.head()

num = np.random.random_integers(3000)

path = '/kaggle/input/train/' + df['experiment'][num] + '/Plate' + str(df['plate'][num]) + '/'

_ = os.listdir(path)

img = mpimg.imread(path + _[np.random.random_integers(len(_))])

# img[np.where(img<0.1)] = 0.1

print(img.min(),img.max())

plt.imshow(img)
# ! pip install --upgrade tensorflow==2.0.0-beta1

# ! pip install --upgrade tf-nightly-2.0-preview

import tensorflow as tf

from tensorflow.keras import layers

from tensorflow.keras import Model

from tensorflow.keras.preprocessing.image import ImageDataGenerator





print(tf.__version__)
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(16, (3,3), activation='relu', 

                           # activity_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01), 

                           input_shape=(300,300,3)),

    # tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(256, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(512, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512,activation='relu'),

    tf.keras.layers.Dense(33,activation='softmax')

])



history = model.compile(

                        optimizer=tf.keras.optimizers.Adam(lr=1e-3), 

                        # optimizer=tf.keras.optimizers.Adadelta(lr=1e-0),

                        # optimizer=tf.train.RMSPropOptimizer(learning_rate=0.01),

                        # loss='categorical_crossentropy', metrics=['acc'])

                        loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

model.summary()
TRAINING_DIR = '/kaggle/input/train/'





train_datagen = ImageDataGenerator(

    # rescale = 1./255,

    rotation_range=90,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    vertical_flip=True,

    fill_mode='nearest'

)



train_generator = train_datagen.flow_from_directory(TRAINING_DIR, 

                                                    batch_size=500,

                                                    class_mode='sparse',

                                                    target_size=(300,300)

)



VALIDATION_DIR = '/kaggle/input/test/'

validation_datagen = ImageDataGenerator(

    # rescale = 1./255,

    rotation_range=90,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    vertical_flip=True,

    fill_mode='nearest'

)



validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,

                                                    batch_size=50,

                                                    class_mode='sparse',

                                                    target_size=(300,300)

)



history = model.fit_generator(train_generator,

                              steps_per_epoch=2,

                              epochs=3, 

                              verbose=1, 

                              validation_steps=1,

                              validation_data=validation_generator

)
#-----------------------------------------------------------

# Retrieve a list of list results on training and test data

# sets for each training epoch

#-----------------------------------------------------------

acc=history.history['sparse_categorical_accuracy']

val_acc=history.history['val_sparse_categorical_accuracy']

loss=history.history['loss']

val_loss=history.history['val_sparse_categorical_accuracy']



epochs=range(len(acc)) # Get number of epochs



#------------------------------------------------

# Plot training and validation accuracy per epoch

#------------------------------------------------

plt.plot(epochs, acc, 'r', "Training Accuracy")

plt.plot(epochs, val_acc, 'b', "Validation Accuracy")

plt.title('Training and validation accuracy')

plt.grid('True')

plt.figure()



#------------------------------------------------

# Plot training and validation loss per epoch

#------------------------------------------------

plt.plot(epochs, loss, 'r', "Training Loss")

plt.plot(epochs, val_loss, 'b', "Validation Loss")

plt.grid('True')

plt.figure()