%pylab --no-import-all

%matplotlib inline

import numpy as np

import pandas as pd

import dicom

import os

import scipy.ndimage



images_path = '../input/sample_images/'
def get_3d_data(path):

    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]

    slices.sort(key = lambda x: int(x.InstanceNumber))

    return np.stack([s.pixel_array for s in slices])
patients = os.listdir(images_path)

patients.sort()



sample_image = get_3d_data(images_path + patients[0])

sample_image.shape
#the images have the unavailable pixel set to -2000, changing them to 0 makes the picture clearer

sample_image[sample_image == -2000] = 0
#same plane as the original data, cut at the Z axis

pylab.imshow(sample_image[100], cmap=pylab.cm.bone)

pylab.show()
#cut at the Y axis

pylab.imshow(sample_image[:, 100, :], cmap=pylab.cm.bone)

pylab.show()
#cut at the X axis

pylab.imshow(sample_image[:, :, 100], cmap=pylab.cm.bone)

pylab.show()
#can also do some operations on the images directly, mostly useful for exploration

pylab.imshow(np.average(sample_image, 0), cmap=pylab.cm.bone)

pylab.show()
#remaping the image to 1 standard deviation of the average and clipping it to 0-1

img_std = np.std(sample_image)

img_avg = np.average(sample_image)

std_image = np.clip((sample_image - img_avg + img_std) / (img_std * 2), 0, 1)
#same cut as before, a bit easier to spot the features

pylab.imshow(std_image[100], cmap=pylab.cm.bone)

pylab.show()
# load training labels

labels_csv = pd.read_csv(images_path + '../stage1_labels.csv', index_col='id')



# Remove the (single) unlabbeled patient from our list

patients = labels_csv.ix[patients].dropna().index



# And finally get the training labels

train_labels = labels_csv.ix[patients].cancer.astype(np.float16).as_matrix()

train_labels = train_labels.reshape([len(train_labels), 1])
# Loads, resizes and processes the image

def process_image(path):

    img = get_3d_data(path)

    img[img == -2000] = 0

    img = scipy.ndimage.zoom(img.astype(np.float), 0.25)

    img_std = np.std(img)

    img_avg = np.average(img)

    return np.clip((img - img_avg + img_std) / (img_std * 2), 0, 1).astype(np.float16)
train_features = np.zeros([len(patients), 1, 128, 128, 128], np.float16)

for i in range(len(patients)):

    f = process_image(images_path + patients[i])

    f = np.concatenate([f, np.zeros([128 - f.shape[0], 128, 128], np.float16)]) # Pads the image

    f = f.reshape([1, 128, 128, 128]) # add an extra dimension for the color channel

    train_features[i] = f

train_features.shape
# This is a 5 minute CNN model roughly based on VGG, don't try to find any deep insights here, it's mostly random

import keras



nn = keras.models.Sequential([

        keras.layers.convolutional.Convolution3D(32, 3, 3, 3, border_mode='same', activation='relu', input_shape=train_features.shape[1:], dim_ordering='th'),

        keras.layers.convolutional.MaxPooling3D((2, 2, 2), (2, 2, 2), dim_ordering='th'),

        keras.layers.convolutional.Convolution3D(32, 3, 3, 3, border_mode='same', activation='relu', dim_ordering='th'),

        keras.layers.convolutional.MaxPooling3D((2, 2, 2), (2, 2, 2), dim_ordering='th'),

        keras.layers.convolutional.Convolution3D(64, 3, 3, 3, border_mode='same', activation='relu', dim_ordering='th'),

        keras.layers.convolutional.MaxPooling3D((2, 2, 2), (2, 2, 2), dim_ordering='th'),

        keras.layers.convolutional.Convolution3D(64, 3, 3, 3, border_mode='same', activation='relu', dim_ordering='th'),

        keras.layers.convolutional.MaxPooling3D((2, 2, 2), (2, 2, 2), dim_ordering='th'),

        keras.layers.convolutional.Convolution3D(128, 3, 3, 3, border_mode='same', activation='relu', dim_ordering='th'),

        keras.layers.convolutional.MaxPooling3D((2, 2, 2), (2, 2, 2), dim_ordering='th'),

        keras.layers.convolutional.Convolution3D(256, 3, 3, 3, border_mode='same', activation='relu', dim_ordering='th'),

        keras.layers.convolutional.AveragePooling3D((4, 4, 4), dim_ordering='th'),

        keras.layers.core.Flatten(),

        keras.layers.core.Dense(32, activation='relu'),

        keras.layers.BatchNormalization(),

        keras.layers.core.Dense(1, activation='sigmoid')

    ])

nn.compile(optimizer=keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
# Finally train the CNN

# nn.fit(train_features, train_labels, batch_size=1, validation_split=0.1, nb_epoch=1)