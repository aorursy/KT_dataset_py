# Install Tensorflow 2.0 Alpha

!pip install tensorflow-gpu==2.0.0-alpha0

# Install Split-folders to split the dataset into train and validation set

!pip install split-folders
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import pathlib

import os

import random as rnd

import tensorflow as tf



from tensorflow import keras



%matplotlib inline
import split_folders

orig_path = '../input/cell_images/cell_images/'

output_path = '../dataset/'

split_folders.ratio(orig_path, output=output_path, seed=1, ratio=(.9, .1))
from tensorflow.keras.preprocessing.image import ImageDataGenerator



IMAGE_TARGE_SIZE = 64

train_gen = ImageDataGenerator(rescale = 1. / 255, rotation_range = 0.05, shear_range = 0.1, horizontal_flip = True, vertical_flip = True, fill_mode = 'wrap')

val_gen = ImageDataGenerator(rescale = 1. / 255)



train_set = train_gen.flow_from_directory(output_path + 'train/', target_size = (IMAGE_TARGE_SIZE, IMAGE_TARGE_SIZE), batch_size = 100, class_mode = 'binary')

val_set = val_gen.flow_from_directory(output_path + 'val/', target_size=(IMAGE_TARGE_SIZE, IMAGE_TARGE_SIZE), batch_size=100, class_mode='binary', shuffle = False)
parasitized_dir = output_path + 'train/Parasitized/'

uninfected_dir = output_path + 'train/Uninfected/'

parasitized_images = [parasitized_dir + x for x in os.listdir(parasitized_dir)]

uninfected_images = [uninfected_dir + x for x in os.listdir(uninfected_dir)]



def draw_image(ax, imgPath, xlabel, fontsize = 20):

    ax.imshow(imread(imgPath))

    ax.set_xticks([])

    ax.set_yticks([])

    ax.set_xlabel(xlabel)

    ax.xaxis.label.set_fontsize(fontsize)
from matplotlib.image import imread



parasitized_sample = rnd.sample(parasitized_images, 8)

uninfected_sample = rnd.sample(uninfected_images, 8)



fig, ax = plt.subplots(4, 4)

fig.set_size_inches(25, 25)



k = 0

for i in range(0, 4, 2):

    for j in range(4):

        draw_image(ax[i, j], parasitized_sample[k], 'Parasitized')

        draw_image(ax[i + 1, j], uninfected_sample[k], 'Uninfected')

        k = k + 1

        

plt.show()
fig, (ax1, ax2) = plt.subplots(2, 1)

fig.set_size_inches(30, 30)



draw_image(ax1, parasitized_sample[0], 'Parasitized', fontsize = 30)

draw_image(ax2, uninfected_sample[0], 'Uninfected', fontsize = 30)

plt.show()
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Dropout, Flatten, Dense



model = tf.keras.Sequential([

    Conv2D(filters=19, kernel_size=(3, 3), activation='relu', input_shape=(IMAGE_TARGE_SIZE, IMAGE_TARGE_SIZE, 3), padding='same'),

    AveragePooling2D(pool_size = (2, 2)),

    

    Conv2D(32, (3, 3), strides = (1, 1), padding = 'same', activation = 'relu'),

    AveragePooling2D(pool_size = (2, 2)),

    

    Flatten(),

    Dense(96, activation = 'relu'),

    Dropout(0.2),

    

    Dense(65, activation = 'relu'),

    Dropout(0.2),

    

    Dense(16, activation = 'relu'),

    Dropout(0.2),



    Dense(1, activation = 'sigmoid')

])

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
history = model.fit(train_set, epochs = 18, batch_size = 250, validation_data = val_set)
# Plot training & validation accuracy values

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
from sklearn.metrics import confusion_matrix

import seaborn as sn



y_pred = model.predict_classes(val_set)

y_true = val_set.classes



cm = confusion_matrix(y_true, y_pred)

df_cm = pd.DataFrame(cm)

plt.figure(figsize = (5, 4))

sn.heatmap(df_cm, annot=True, fmt='g')
from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred))
val_images = [output_path + 'val/' + x for x in val_set.filenames]

wrongly_predicted_imgs = dict()

for i, _ in enumerate(y_pred):

    if y_true[i] != y_pred[i]:

        label = 'Parasitized' if y_true[i] == 0 else 'Uninfected'

        label = label + ' - ' + ('Parasitized' if y_pred[i] == False else 'Uninfected')

        wrongly_predicted_imgs[val_images[i]] = label



cols = 5

rows = len(wrongly_predicted_imgs) // cols

fig, ax = plt.subplots(rows, cols)

fig.set_size_inches(25, 200)



k = 0

key_list = list(wrongly_predicted_imgs.keys())

for i in range(rows):

    for j in range(cols):

        key = key_list[k]

        draw_image(ax[i, j], key, wrongly_predicted_imgs[key], fontsize = 18)

        k = k + 1

        if (k >= len(wrongly_predicted_imgs)):

            break

        

plt.show()