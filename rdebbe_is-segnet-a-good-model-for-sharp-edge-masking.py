# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

import gzip

import os

from os.path import basename

import glob

import cv2

import random

from PIL import Image

#import scipy.ndimage

from scipy import ndimage

import matplotlib.pyplot as plt

%matplotlib inline



from scipy.misc import imresize

from skimage.transform import resize



from sklearn.model_selection import train_test_split



from keras import models



from keras.models import Model

from keras.layers.core import Activation, Reshape, Permute

from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D

from keras.layers.normalization import BatchNormalization

from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Concatenate

#from keras.layers import Dense, Conv2D, Input, MaxPool2D, UpSampling2D, Concatenate, Conv2DTranspose

from keras.optimizers import Adam

from keras.optimizers import SGD

from keras.optimizers import RMSprop

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping

from keras.callbacks import ReduceLROnPlateau, TensorBoard, Callback

from keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator

from keras.losses import binary_crossentropy



from keras import backend as K



K.set_image_dim_ordering('tf') # Theano dimension ordering in this code



from keras import __version__ as keras_version

import sys

print(sys.version)

print(sys.path)

print('Keras version: {}'.format(keras_version))

print('openCV version: ', cv2.__version__)

INPUT_PATH = '../input/'

input_size = 256

dims = [input_size, input_size]    #height X width

img_rows = dims[0]

img_cols = dims[1]

n_labels = 2
train = sorted(glob.glob(INPUT_PATH + 'train/*.jpg'))

masks = sorted(glob.glob(INPUT_PATH + 'train_masks/*.gif'))

test  = sorted(glob.glob(INPUT_PATH + 'test/*.jpg'))

print('Number of training images: ', len(train), ' Number of corresponding masks: ', len(masks), ' Number of test images: ', len(test))



meta = pd.read_csv(INPUT_PATH + 'metadata.csv')

mask_df = pd.read_csv(INPUT_PATH + 'train_masks.csv')

ids_train = mask_df['img'].map(lambda s: s.split('_')[0]).unique()

print('Length of ids_train ', len(ids_train))
smooth = 1.



def dice_coef(y_true, y_pred):

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)



def dice_coef_np(y_true,y_pred):

    y_true_f = y_true.flatten()

    y_pred_f = y_pred.flatten()

    intersection = np.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)



def dice_coef_loss(y_true, y_pred):

    return -dice_coef(y_true, y_pred)



def dice_loss(y_true, y_pred):

    smooth = 1.

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)





def bce_dice_loss(y_true, y_pred):

    return binary_crossentropy(y_true, y_pred) + (1 - dice_loss(y_true, y_pred))
def label_map(labels):

    label_map = np.zeros([img_rows, img_cols, n_labels])    

    #print('label_map shape ', label_map.shape)

    for r in range(img_rows):

        for c in range(img_cols):

            #label_map[r, c, labels[r][c]] = 1

            label_map[r, c, labels[r, c]] = 1

    return label_map



def build_model(img_w, img_h, filters):

    n_labels = 2



    kernel = 3



    encoding_layers = [

        Conv2D(64, (kernel, kernel), input_shape=(img_h, img_w, 3), padding='same'),

        BatchNormalization(),

        Activation('relu'),

        Convolution2D(64, (kernel, kernel), padding='same'),

        BatchNormalization(),

        Activation('relu'),

        MaxPooling2D(),



        Convolution2D(128, (kernel, kernel), padding='same'),

        BatchNormalization(),

        Activation('relu'),

        Convolution2D(128, (kernel, kernel), padding='same'),

        BatchNormalization(),

        Activation('relu'),

        MaxPooling2D(),



        Convolution2D(256, (kernel, kernel), padding='same'),

        BatchNormalization(),

        Activation('relu'),

        Convolution2D(256, (kernel, kernel), padding='same'),

        BatchNormalization(),

        Activation('relu'),

        Convolution2D(256, (kernel, kernel), padding='same'),

        BatchNormalization(),

        Activation('relu'),

        MaxPooling2D(),



        Convolution2D(512, (kernel, kernel), padding='same'),

        BatchNormalization(),

        Activation('relu'),

        Convolution2D(512, (kernel, kernel), padding='same'),

        BatchNormalization(),

        Activation('relu'),

        Convolution2D(512, (kernel, kernel), padding='same'),

        BatchNormalization(),

        Activation('relu'),

        MaxPooling2D(),



        Convolution2D(512, (kernel, kernel), padding='same'),

        BatchNormalization(),

        Activation('relu'),

        Convolution2D(512, (kernel, kernel), padding='same'),

        BatchNormalization(),

        Activation('relu'),

        Convolution2D(512, (kernel, kernel), padding='same'),

        BatchNormalization(),

        Activation('relu'),

        MaxPooling2D(),

    ]



    autoencoder = models.Sequential()

    autoencoder.encoding_layers = encoding_layers



    for l in autoencoder.encoding_layers:

        autoencoder.add(l)



    decoding_layers = [

        UpSampling2D(),

        Convolution2D(512, (kernel, kernel), padding='same'),

        BatchNormalization(),

        Activation('relu'),

        Convolution2D(512, (kernel, kernel), padding='same'),

        BatchNormalization(),

        Activation('relu'),

        Convolution2D(512, (kernel, kernel), padding='same'),

        BatchNormalization(),

        Activation('relu'),



        UpSampling2D(),

        Convolution2D(512, (kernel, kernel), padding='same'),

        BatchNormalization(),

        Activation('relu'),

        Convolution2D(512, (kernel, kernel), padding='same'),

        BatchNormalization(),

        Activation('relu'),

        Convolution2D(256, (kernel, kernel), padding='same'),

        BatchNormalization(),

        Activation('relu'),



        UpSampling2D(),

        Convolution2D(256, (kernel, kernel), padding='same'),

        BatchNormalization(),

        Activation('relu'),

        Convolution2D(256, (kernel, kernel), padding='same'),

        BatchNormalization(),

        Activation('relu'),

        Convolution2D(128, (kernel, kernel), padding='same'),

        BatchNormalization(),

        Activation('relu'),



        UpSampling2D(),

        Convolution2D(128, (kernel, kernel), padding='same'),

        BatchNormalization(),

        Activation('relu'),

        Convolution2D(64, (kernel, kernel), padding='same'),

        BatchNormalization(),

        Activation('relu'),



        UpSampling2D(),

        Convolution2D(64, (kernel, kernel), padding='same'),

        BatchNormalization(),

        Activation('relu'),

        Convolution2D(n_labels, (1, 1), padding='valid', activation="sigmoid"),

        BatchNormalization(),

    ]

    autoencoder.decoding_layers = decoding_layers

    for l in autoencoder.decoding_layers:

        autoencoder.add(l)



    autoencoder.add(Reshape((n_labels, img_h * img_w)))

    autoencoder.add(Permute((2, 1)))

    autoencoder.add(Activation('softmax'))



    #with open('model_5l.json', 'w') as outfile:

    #    outfile.write(json.dumps(json.loads(autoencoder.to_json()), indent=2))

    

    return autoencoder
# split the train set into train and validation sets:

train_images, validation_images = train_test_split(train, train_size=0.8, test_size=0.2)

print('Split into training set with ', len(train_images), ' images and validation set with  ', len(validation_images), ' images')
# batch generator for training

def data_gen_small( images, batch_size):

        print('entered data_gen_small batch size: ', batch_size, 'size of input: ', len(images))

    

        while True:

            #

            # use all data sequentially

            #

            for start in range(0, len(images), batch_size):

                x_batch = []

                y_batch = []

                end = min(start + batch_size, len(images))

                ix = images[start:end] 

                imgs = []

                labels = []

                for i in ix:

                    img = cv2.imread(i)

                    img = cv2.resize(img, (input_size, input_size))

                    mask_filename = basename(i)

                    no_extension = os.path.splitext(mask_filename)[0]

                    correct_mask = INPUT_PATH + 'train_masks/' + no_extension + '_mask.gif' 

                    original_mask = Image.open(correct_mask).convert('L')

                    resized_mask = imresize(original_mask, dims+[3])

                    array_mask = resized_mask / 255

                    gt = np.clip(array_mask, 0, 1)

                    gt = np.array(gt, np.int)

                    x_batch.append(img)

                    y_batch.append(label_map(gt))

                x_batch = np.array(x_batch, np.float32) / 255

                y_batch = np.array(y_batch, np.float32) 

                y_batch = np.array(y_batch).reshape((batch_size, img_rows * img_cols, n_labels))

                yield x_batch, y_batch



            

# create an instance of a training generator:

train_gen = data_gen_small( train_images, 1) 

img, msk = next(train_gen) 

print('shape of image batch: ', img.shape, ' shape of mask batch: ', msk.shape)

# create an instance of a validation generator:

validation_gen = data_gen_small( validation_images, 2) 

imgv, mskv = next(validation_gen)

print('shape of validation batch: ', imgv.shape, ' shape of validation mask batch: ', mskv.shape)

model = build_model(input_size, input_size, 10)



optimizer = SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False)

model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

print( 'Compilation: OK')

#model.summary()
# to read the complete train and validation sets with batches of size 5 one would use:

# steps_per_epoch = 814

# validation_steps = 204

# I run for 50 epochs

# here I am limited by kaggle's limit of 1200 sec of cpu.

#

model.fit_generator(train_gen, steps_per_epoch=14, epochs=7, validation_data=validation_gen, validation_steps=20)
predicted_mask = model.predict(img)

print('shape of prediction: ', predicted_mask.shape)
predicted_mask = predicted_mask.reshape((1, input_size, input_size, 2))

plt.imshow(predicted_mask[0][:, :, 0])

plt.show()
labeled = np.argmax(predicted_mask[0], axis=-1)

plt.imshow(labeled)
plt.imshow(img[0])