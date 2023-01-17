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
covid_path = '../input/covid2019/Dataset/Validation/Positives/F4341CE7-73C9-45C6-99C8-8567A5484B63.jpeg'

normal_path = '../input/covid2019/Dataset/Validation/Negatives/pneumocystis-jiroveci-pneumonia-4-PA.png'
import matplotlib.image as mpimg

import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow as tf



img = mpimg.imread(covid_path)

#imgplot = plt.imshow(img, cmap = 'gray')

imgplot = plt.imshow(img)
img = mpimg.imread(normal_path)

imgplot = plt.imshow(img)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
batch_size = 5

epochs = 12

IMG_HEIGHT = 150

IMG_WIDTH = 150



image_gen_train = ImageDataGenerator(

                    rescale=1./255,

                    width_shift_range=.15,

                    height_shift_range=.15,

                    zoom_range=0.5

                    )



train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data

validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

test_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our test data
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,

                                                           directory='../input/covid2019/Dataset/Train',

                                                           shuffle=True,

                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),

                                                           class_mode='binary')



val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,

                                                              directory='../input/covid2019/Dataset/Validation',

                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),

                                                              class_mode='binary')



test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,

                                                              directory='../input/covid2019/Dataset/Test',

                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),class_mode='binary')
sample_training_images, _ = next(train_data_gen)
def plotImages(images_arr):

    fig, axes = plt.subplots(1, 5, figsize=(20,20))

    axes = axes.flatten()

    for img, ax in zip( images_arr, axes):

        ax.imshow(img)

        ax.axis('off')

    plt.tight_layout()

    plt.show()
model = Sequential([

    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),

    MaxPooling2D(),

    Conv2D(32, 3, padding='same', activation='relu'),

    MaxPooling2D(),

    Conv2D(64, 3, padding='same', activation='relu'),

    MaxPooling2D(),

    Flatten(),

    Dense(512, activation='relu'),

    Dense(1)

])
from keras import optimizers

model.compile(optimizer='adam',

              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),

              metrics=['accuracy'])
model.summary()
history = model.fit(

    train_data_gen,

    steps_per_epoch=6,

    epochs=epochs,

    validation_data=val_data_gen,

    validation_steps=6

)
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



loss=history.history['loss']

val_loss=history.history['val_loss']



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
# Save the entire model as a SavedModel

model.save('covid_edwin_new.cnn') 
pred_Y = model.predict(test_data_gen,

                       #batch_size = 5,

                       verbose = True)

print(pred_Y[:15])