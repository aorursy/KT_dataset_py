# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random

import cv2

import matplotlib.pyplot as plt



# Dl framwork - tensorflow, keras a backend 

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





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

dir_name = '/kaggle/input/'

# for dirname, _, filenames in os.walk(dir_name):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))

    

# Any results you write to the current directory are saved as output.
config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56}, log_device_placement=True ) 

sess = tf.compat.v1.Session(config=config) 
# Input data file

train_path = os.path.join(dir_name, 'chest-xray-pneumonia/chest_xray/train')

train_normal_path = os.path.join(train_path, 'NORMAL')

train_pneumonia_path = os.path.join(train_path, 'PNEUMONIA')



val_path = os.path.join(dir_name, 'chest-xray-pneumonia/chest_xray/test')

val_normal_path = os.path.join(val_path, 'NORMAL')

val_pneumonia_path = os.path.join(val_path, 'PNEUMONIA')



test_path = os.path.join(dir_name, 'chest-xray-pneumonia/chest_xray/test')

test_normal_path = os.path.join(test_path, 'NORMAL')

test_pneumonia_path = os.path.join(test_path, 'PNEUMONIA')
# # Plot the Images:

def plot_images(item_dir, n=6):

    all_item_dir = os.listdir(item_dir)

    item_files = [os.path.join(item_dir, file) for file in all_item_dir][:n]

    

    plt.figure(figsize=(35, 10))

    for idx, img_path in enumerate(item_files):

        plt.subplot(2, n, idx+1)

        img = plt.imread(img_path)

        plt.imshow(img, cmap='gray')

        plt.axis('off')

    

    plt.tight_layout()
plot_images(train_normal_path, 5)

plot_images(train_pneumonia_path, 5)
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
Images_details(train_normal_path)

Images_details(train_pneumonia_path)
input_path = "/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/"

    

for _set in ['train', 'test', 'val']:

    nrml = len(os.listdir(input_path + _set + '/NORMAL'))

    pnm = len(os.listdir(input_path + _set + '/PNEUMONIA'))

    print('{}, Normal images: {}, Pneumonia images: {}'.format(_set, nrml, pnm))


def process_data(img_dims, batch_size):

    # Data generation objects - thorugh rescalling, veticle flip, zoom range

    train_datagen = ImageDataGenerator(

                        rescale = 1./255,

                      # featurewise_center=True,

                      # featurewise_std_normalization=True,

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

    

    # Making predictions off of the test set in one batch size

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
#### Fitting the model

history = model.fit_generator(

           train_gen, steps_per_epoch=train_gen.samples // batch_size, 

           epochs=epochs, 

           validation_data=test_gen, 

           validation_steps=test_gen.samples // batch_size,

           callbacks=[checkpoint, lr_reduce])
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
# confution matrix

from sklearn.metrics import accuracy_score, confusion_matrix

from mlxtend.plotting import plot_confusion_matrix



preds = model.predict(test_data)



accuracy = accuracy_score(test_labels, np.round(preds))*100

conf_mat = confusion_matrix(test_labels, np.round(preds))

true_negative, false_postive, false_negative, true_posiitve = conf_mat.ravel()



plt.title("CONFUSION MATRIX")

plot_confusion_matrix(conf_mat,figsize=(12,8), hide_ticks=True,cmap=plt.cm.Blues)

plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)

plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)

plt.show()
print('\n','-'*20,' TEST METRICS', '-'*20)

precision = true_posiitve / (true_posiitve + false_postive) * 100

recall = true_posiitve / (true_posiitve + false_negative) * 100

print('\tAccuracy: {}%'.format(accuracy))

print('\tPrecision: {}%'.format(precision))

print('\tRecall: {}%'.format(recall))

print('\tF1-score: {}'.format(2*precision*recall/(precision+recall)))
# Save the model

model.save('xray-pneumona-depthwise-convolution.h5')