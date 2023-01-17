# Importing the required libraries

import os

import numpy as np

import pandas as pd 

import random

import cv2

import matplotlib.pyplot as plt

%matplotlib inline

import keras.backend as K

from keras.models import Model, Sequential

from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization

from keras.layers import Conv2D, SeparableConv2D, MaxPool2D, LeakyReLU, Activation

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

import tensorflow as tf
seed = 232

tf.random.set_seed(

    seed

)
input_path = '../input/chest-xray-pneumonia//chest_xray/chest_xray/'



fig, ax = plt.subplots(2, 3, figsize=(15, 7))

ax = ax.ravel()

plt.tight_layout()



for i, _set in enumerate(['train', 'val', 'test']):

    set_path = input_path+_set

    ax[i].imshow(plt.imread(set_path+'/NORMAL/'+os.listdir(set_path+'/NORMAL') [0]), cmap='gray')

    ax[i].set_title('Set: {}, Condition: Normal'.format(_set))

    ax[i+3].imshow(plt.imread(set_path+'/PNEUMONIA/'+os.listdir(set_path+'/PNEUMONIA') [0]), cmap='gray')

    ax[i+3].set_title('Set: {}, Condition: Pneumonia'.format(_set))
normal_example = os.listdir(f'{input_path}train/NORMAL')[0]

pneumonia_example = os.listdir(f'{input_path}train/PNEUMONIA')[0]
normal_img = plt.imread(f'{input_path}train/NORMAL/{normal_example}')

plt.imshow(normal_img)
pneumonia_img = plt.imread(f'{input_path}train/PNEUMONIA/{pneumonia_example}')

plt.imshow(pneumonia_img)
# Distribution of our datasets

for _set in ['train', 'val', 'test']:

    n_normal = len(os.listdir(input_path + _set + '/NORMAL'))

    n_infect = len(os.listdir(input_path + _set + '/PNEUMONIA'))

    print('Set: {}, normal images: {}, pneumonia images: {}'.format(_set, n_normal, n_infect))
def process_data(img_dims, batch_size):

    # Data generation objects

    train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, vertical_flip=True)

    test_val_datagen = ImageDataGenerator(rescale=1./255)

    

    # This is fed to the network in the specified batch sizes and image dimensions

    train_gen = train_datagen.flow_from_directory(

    directory=input_path+'train', 

    target_size=(img_dims, img_dims), 

    batch_size=batch_size, 

    class_mode='binary', 

    shuffle=True)



    test_gen = test_val_datagen.flow_from_directory(

    directory=input_path+'test', 

    target_size=(img_dims, img_dims), 

    batch_size=batch_size, 

    class_mode='binary', 

    shuffle=True)

    

    # I will be making predictions off of the test set in one batch size

    # This is useful to be able to get the confusion matrix

    test_data = []

    test_labels = []



    for cond in ['/NORMAL/', '/PNEUMONIA/']:

        for img in (os.listdir(input_path + 'test' + cond)):

            img = plt.imread(input_path+'test'+cond+img)

            img = cv2.resize(img, (img_dims, img_dims))

            img = np.dstack([img, img, img])

            img = img.astype('float32') / 255

            if cond=='/NORMAL/':

                label = 0

            elif cond=='/PNEUMONIA/':

                label = 1

            test_data.append(img)

            test_labels.append(label)

        

    test_data = np.array(test_data)

    test_labels = np.array(test_labels)

    

    return train_gen, test_gen, test_data, test_labels
# Hyperparameters

img_dims = 150

epochs = 10

batch_size = 32



# Getting the data

train_gen, test_gen, test_data, test_labels = process_data(img_dims, batch_size)
# Input layer

inputs = Input(shape=(img_dims, img_dims, 3))



# First conv block

x = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(inputs)

x = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(x)

x = MaxPool2D(pool_size=(2, 2))(x)



# Second conv block

x = SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)

x = SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)

x = BatchNormalization()(x)

x = MaxPool2D(pool_size=(2, 2))(x)



# Third conv block

x = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)

x = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)

x = BatchNormalization()(x)

x = MaxPool2D(pool_size=(2, 2))(x)



# Fourth conv block

x = SeparableConv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)

x = SeparableConv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)

x = BatchNormalization()(x)

x = MaxPool2D(pool_size=(2, 2))(x)

x = Dropout(rate=0.2)(x)



# Fifth conv block

x = SeparableConv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)

x = SeparableConv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)

x = BatchNormalization()(x)

x = MaxPool2D(pool_size=(2, 2))(x)

x = Dropout(rate=0.2)(x)



# FC layer

x = Flatten()(x)

x = Dense(units=512, activation='relu')(x)

x = Dropout(rate=0.7)(x)

x = Dense(units=128, activation='relu')(x)

x = Dropout(rate=0.5)(x)

x = Dense(units=64, activation='relu')(x)

x = Dropout(rate=0.3)(x)



# Output layer

output = Dense(units=1, activation='sigmoid')(x)



# Creating model and compiling

model = Model(inputs=inputs, outputs=output)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



# Callbacks

checkpoint = ModelCheckpoint(filepath='best_weights.hdf5', save_best_only=True, save_weights_only=True)

lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=2, mode='max')

early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=1, mode='min')

# Traning Model

hist = model.fit_generator(

           train_gen, steps_per_epoch=train_gen.samples // batch_size, 

           epochs=epochs, validation_data=test_gen, 

           validation_steps=test_gen.samples // batch_size, callbacks=[checkpoint, lr_reduce])
fig, ax = plt.subplots(1, 2, figsize=(10, 3))

ax = ax.ravel()



for i, met in enumerate(['accuracy', 'loss']):

    ax[i].plot(hist.history[met])

    ax[i].plot(hist.history['val_' + met])

    ax[i].set_title('Model {}'.format(met))

    ax[i].set_xlabel('epochs')

    ax[i].set_ylabel(met)

    ax[i].legend(['train', 'val'])
from sklearn.metrics import accuracy_score, confusion_matrix



preds = model.predict(test_data)



acc = accuracy_score(test_labels, np.round(preds))*100

cm = confusion_matrix(test_labels, np.round(preds))

tn, fp, fn, tp = cm.ravel()



print('CONFUSION MATRIX ------------------')

print(cm)



print('\nTEST METRICS ----------------------')

precision = tp/(tp+fp)*100

recall = tp/(tp+fn)*100

print('Accuracy: {}%'.format(acc))

print('Precision: {}%'.format(precision))

print('Recall: {}%'.format(recall))

print('F1-score: {}'.format(2*precision*recall/(precision+recall)))



print('\nTRAIN METRIC ----------------------')

print('Train acc: {}'.format(np.round((hist.history['accuracy'][-1])*100, 2)))