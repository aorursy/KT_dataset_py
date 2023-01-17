import os

import numpy as np

import pandas as pd 

import random

import cv2

import matplotlib.pyplot as plt



import tensorflow as tf

import tensorflow.keras.backend as K

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization

from tensorflow.keras.layers import Conv2D, SeparableConv2D, MaxPool2D, LeakyReLU, Activation

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping



from os import listdir

from os.path import isfile, join

from PIL import Image

import glob





seed = 2019

np.random.seed(seed)

%matplotlib inline
tf.__version__
Training_kaggle = True
if  Training_kaggle==True:

    dirname = '/kaggle/input'

    train_path = os.path.join(dirname, 'chest-xray-pneumonia/chest_xray/chest_xray/train')

    train_nrml_pth = os.path.join(train_path, 'NORMAL')

    train_pnm_pth = os.path.join(train_path, 'PNEUMONIA')

    test_path = os.path.join(dirname, 'chest-xray-pneumonia/chest_xray/chest_xray/test')

    test_nrml_pth = os.path.join(test_path, 'NORMAL')

    test_pnm_pth = os.path.join(test_path, 'PNEUMONIA')

    val_path = os.path.join(dirname, 'chest-xray-pneumonia/chest_xray/chest_xray/test')

    val_nrml_pth = os.path.join(val_path, 'NORMAL')

    val_pnm_pth = os.path.join(val_path, 'PNEUMONIA')

else:

    dirname = ''

    train_path = os.path.join(dirname, '../input/chest_xray/train/')

    train_nrml_pth = os.path.join(train_path, 'NORMAL')

    train_pnm_pth = os.path.join(train_path, 'PNEUMONIA')

    test_path = os.path.join(dirname, '../input/chest_xray/test/')

    test_nrml_pth = os.path.join(test_path, 'NORMAL')

    test_pnm_pth = os.path.join(test_path, 'PNEUMONIA')

    val_path = os.path.join(dirname, '../input/chest_xray/val/')

    val_nrml_pth = os.path.join(val_path, 'NORMAL')

    val_pnm_pth = os.path.join(val_path, 'PNEUMONIA')
train_nrml_pth
def plot_imgs(item_dir, num_imgs=6):

    all_item_dirs = os.listdir(item_dir)

    item_files = [os.path.join(item_dir, file) for file in all_item_dirs][:num_imgs]



    plt.figure(figsize=(20, 12))

    for idx, img_path in enumerate(item_files):

        plt.subplot(2, 3, idx+1)



        img = plt.imread(img_path)

        plt.imshow(img, cmap='gray')



    plt.tight_layout()
plot_imgs(train_nrml_pth)
plot_imgs(train_pnm_pth)
def Images_details_Print_data(data, path):

    print(" ====== Images in: ", path)    

    for k, v in data.items():

        print("%s:\t%s" % (k, v))



def Images_details(path):

    files = [f for f in glob.glob(path + "**/*.*", recursive=True)]

    data = {}

    data['images_count'] = len(files)

    data['min_width'] = 10**100  # No image will be bigger than that

    data['max_width'] = 0

    data['min_height'] = 10**100  # No image will be bigger than that

    data['max_height'] = 0





    for f in files:

        im = Image.open(f)

        width, height = im.size

        data['min_width'] = min(width, data['min_width'])

        data['max_width'] = max(width, data['max_height'])

        data['min_height'] = min(height, data['min_height'])

        data['max_height'] = max(height, data['max_height'])



    Images_details_Print_data(data, path)

Images_details(train_nrml_pth)

Images_details(train_pnm_pth)
Images_details(test_nrml_pth)

Images_details(test_pnm_pth)
Images_details(val_nrml_pth)

Images_details(val_pnm_pth)
if  Training_kaggle==True:

    input_path = "/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/"

else:

    input_path = "../input/chest_xray/"

    

for _set in ['train', 'test', 'val']:

    nrml = len(os.listdir(input_path + _set + '/NORMAL'))

    pnm = len(os.listdir(input_path + _set + '/PNEUMONIA'))

    print('{}, Normal images: {}, Pneumonia images: {}'.format(_set, nrml, pnm))
def process_data(img_dims, batch_size):

    # Data generation objects

    train_datagen = ImageDataGenerator(

        rescale = 1./255,

      #  featurewise_center=True,

      #  featurewise_std_normalization=True,

        zoom_range = 0.3,

        vertical_flip = True)

    

    test_datagen = ImageDataGenerator(

      #  featurewise_center=True,

      #  featurewise_std_normalization=True,

        rescale=1./255)

    

    # This is fed to the network in the specified batch sizes and image dimensions

    train_gen = train_datagen.flow_from_directory(

    directory = train_path, 

    target_size = (img_dims, img_dims), 

    batch_size = batch_size, 

    class_mode = 'binary', 

    shuffle=True)



    test_gen = test_datagen.flow_from_directory(

    directory=test_path, 

    target_size=(img_dims, img_dims), 

    batch_size=batch_size, 

    class_mode='binary', 

    shuffle=True)

    

    # I will be making predictions off of the test set in one batch size

    # This is useful to be able to get the confusion matrix

    test_data = []

    test_labels = []



    for cond in ['/NORMAL/', '/PNEUMONIA/']:

        for img in (os.listdir(test_path + cond)):

            img = plt.imread(test_path + cond + img)

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

epochs = 20

batch_size = 32



# Getting the data

train_gen, test_gen, test_data, test_labels = process_data(img_dims, batch_size)
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
model.summary()
history = model.fit_generator(

           train_gen, steps_per_epoch=train_gen.samples // batch_size, 

           epochs=epochs, 

           validation_data=test_gen, 

           validation_steps=test_gen.samples // batch_size,

           callbacks=[checkpoint, lr_reduce])
history.history
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

print('Train acc: {}'.format(np.round((history.history['accuracy'][-1])*100, 2)))