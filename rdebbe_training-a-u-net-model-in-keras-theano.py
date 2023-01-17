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



import numpy as np

import gzip

import os

from os.path import basename

import glob

import time

import cv2

import pandas as pd

import random

from PIL import Image

#import scipy.ndimage

from scipy import ndimage

import matplotlib.pyplot as plt

%matplotlib inline



from scipy.misc import imresize

from skimage.transform import resize



from sklearn.model_selection import train_test_split

from keras.models import Model

from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Concatenate

#from keras.layers import Dense, Conv2D, Input, MaxPool2D, UpSampling2D, Concatenate, Conv2DTranspose

from keras.optimizers import Adam

from keras.optimizers import SGD

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping

from keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator

from keras import backend as K



K.set_image_dim_ordering('th') # Theano dimension ordering in this code

INPUT_PATH = '../input/'

dims = [128, 128] 

img_rows = dims[0]

img_cols = dims[1]

train = sorted(glob.glob(INPUT_PATH + 'train/*.jpg'))

masks = sorted(glob.glob(INPUT_PATH + 'train_masks/*.gif'))

test  = sorted(glob.glob(INPUT_PATH + 'test/*.jpg'))

print('Number of training images: ', len(train), ' Number of corresponding masks: ', len(masks), ' Number of test images: ', len(test))



meta = pd.read_csv(INPUT_PATH + 'metadata.csv')

mask_df = pd.read_csv(INPUT_PATH + 'train_masks.csv')

ids_train = mask_df['img'].map(lambda s: s.split('_')[0]).unique()

print('Length of ids_train ', len(ids_train))
mask_df.head()

#'''rle_mask is the run-length encoded version of the training set masks the input for the encoder has to be binary; zeroes and ones.

#I read the encoding output as the pixel number of the first 1 (in the flattened mask) followed by the count of consecutive (uninterupted) series of pixels with value 

#'''
image = cv2.imread(INPUT_PATH + 'train/00087a6bd4dc_01.jpg')

plt.imshow(image)

plt.show()
img = Image.open(INPUT_PATH + 'train_masks/00087a6bd4dc_01_mask.gif').convert('RGB')

plt.imshow(img)

plt.show()
# from ecobill:



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

# this is the U-Net model I encountered in the LUNA16 Challenge

def get_unet():

    inputs = Input((3,img_rows, img_cols))

    conv1 = Conv2D(32, (3, 3), padding="same", activation='relu')(inputs)    

    conv1 = Conv2D(32, (3, 3), padding="same", activation='relu')(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)



    conv2 = Conv2D(64, (3, 3), padding="same", activation='relu')(pool1)

    conv2 = Conv2D(64, (3, 3), padding="same", activation='relu')(conv2)    

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)



    conv3 = Conv2D(128, (3, 3), padding="same", activation='relu')(pool2)

    conv3 = Conv2D(128, (3, 3), padding="same", activation='relu')(conv3)

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)



    conv4 = Conv2D(256, (3, 3), padding="same", activation='relu')(pool3)

    conv4 = Conv2D(256, (3, 3), padding="same", activation='relu')(conv4)

    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)



    conv5 = Conv2D(512, (3, 3), padding="same", activation='relu')(pool4)

    conv5 = Conv2D(512, (3, 3), padding="same", activation='relu')(conv5)



    up6 = Concatenate(axis=1)([UpSampling2D(size=(2, 2))(conv5), conv4])

    #      Concatenate(axis=3)([residual, upconv])

    conv6 = Conv2D(256, (3, 3), padding="same", activation='relu')(up6)

    conv6 = Conv2D(256, (3, 3), padding="same", activation='relu')(conv6)



    up7 = Concatenate(axis=1)([UpSampling2D(size=(2, 2))(conv6), conv3])

    conv7 = Conv2D(128, (3, 3), padding="same", activation='relu')(up7)

    conv7 = Conv2D(128, (3, 3), padding="same", activation='relu')(conv7)



    up8 = Concatenate(axis=1)([UpSampling2D(size=(2, 2))(conv7), conv2])

    conv8 = Conv2D(64, (3, 3), padding="same", activation='relu')(up8)

    conv8 = Conv2D(64, (3, 3), padding="same", activation='relu')(conv8)



    up9 = Concatenate(axis=1)([UpSampling2D(size=(2, 2))(conv8), conv1])

    conv9 = Conv2D(32, (3, 3), padding="same", activation='relu')(up9)

    conv9 = Conv2D(32, (3, 3), padding="same", activation='relu')(conv9)



    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)   #9



    model = Model(inputs=inputs, outputs=conv10)

    #      `Model(inputs=/input_19, outputs=sigmoid.0)`



    #model.compile(optimizer=Adam(lr=1.0e-5), loss=dice_coef_loss, metrics=[dice_coef])  #LUNA16

    model.compile(optimizer=Adam(5e-4), loss='binary_crossentropy', metrics=[dice_coef]) #ecobill



    return model

# split the train set into train and validation:

train_images, validation_images = train_test_split(train, train_size=0.8, test_size=0.2)

print('Split into training set with ', len(train_images), ' images and validation set with  ', len(validation_images), ' images')
#use loading functions from: ecobill



# utility function to convert greyscale images to rgb

def grey2rgb(img):

    new_img = []

    for i in range(img.shape[0]):

        for j in range(img.shape[1]):

            new_img.append(list(img[i][j])*3)

    new_img = np.array(new_img).reshape(img.shape[0], img.shape[1], 3)

    return new_img





# generator that we will use to read the data from the directory

def data_gen_small(data_dir, masks, images, batch_size, dims):

        """

        data_dir: where the actual images are kept

        mask_dir: where the actual masks are kept

        images: the filenames of the images we want to generate batches from

        batch_size: self explanatory

        dims: the dimensions in which we want to rescale our images

        

        Image.resize(size, resample=0)



        Returns a resized copy of this image.

        Parameters:	



        size – The requested size in pixels, as a 2-tuple: (width, height).

        resample – An optional resampling filter. This can be one of PIL.Image.NEAREST, 

        PIL.Image.BOX, PIL.Image.BILINEAR, PIL.Image.HAMMING, PIL.Image.BICUBIC or PIL.Image.LANCZOS. 

        If omitted, or if the image has mode “1” or “P”, it is set PIL.Image.NEAREST

        """

        while True:

            ix = np.random.choice(np.arange(len(images)), batch_size)

            imgs = []

            labels = []

            for i in ix:

                # images

                original_img = cv2.imread(images[i])

                resized_img = imresize(original_img, dims+[3]) #this looks like TensorFlow ordering 

                array_img = resized_img / 255   

                array_img = array_img.swapaxes(0,2)

                imgs.append(array_img)

                #imgs is a numpy array with dim: (batch size X 128 X 128 X 3)

                

                # masks

                mask_filename = basename(images[i])

                no_extension = os.path.splitext(mask_filename)[0]

                correct_mask = INPUT_PATH + 'train_masks/' + no_extension + '_mask.gif' 

                original_mask = Image.open(correct_mask).convert('L')

                data = np.asarray( original_mask, dtype="int32" )

                resized_mask = imresize(original_mask, dims+[3])

                array_mask = resized_mask / 255

                labels.append(array_mask)

            imgs = np.array(imgs)

            labels = np.array(labels)

            relabel = labels.reshape(-1, dims[0], dims[1], 1)

            yield imgs, relabel.swapaxes(1, 3)



train_gen = data_gen_small(INPUT_PATH + 'train/', masks, train_images, 2, dims) 

img, msk = next(train_gen)

print('Size of batch: ', len(img))

print('shape of img ', img.shape, 'number dimensions: ', img[0].ndim)

print('shape of msk ', msk.shape, 'number dimensions: ', msk[0].ndim)

newshape = img[0].swapaxes(0,2)

plt.imshow(newshape)

plt.show()



#try resize up 



resized_img = imresize(img[0], [1280, 1918]+[3])

print('resized up: ', resized_img.shape)

newshape = resized_img.swapaxes(0,1)

print('resized swapaxes: ', newshape.shape)

print('resized swapaxes shape[-1]: ', newshape.shape[-1])



plt.imshow(newshape)

plt.show()



newshape = msk.swapaxes(1,3)

print(newshape.shape)

plt.imshow(grey2rgb(newshape[0]), alpha=0.5)

plt.show()
# create an instance of a validation generator:

validation_gen = data_gen_small(INPUT_PATH + 'train/', masks, validation_images, 4, dims) 
# define and compile the model

model = get_unet()

model.summary()
# fit the model and check dice_coef on validation data at end of each epoch

model.fit_generator(train_gen, steps_per_epoch=50, epochs=1, validation_data=validation_gen, validation_steps=50)
# lets look at one of the predicted masks

img, msk = next(validation_gen)

predicted_mask = model.predict(img)

predicted_mask.shape
newshape = predicted_mask.swapaxes(1,3)

print('newshape shape ', newshape.shape)

grey = grey2rgb(newshape[3])

print('grey shape ', grey.shape)

plt.imshow(grey, alpha = 0.5)

plt.show()
# the corresponding image is:

newshape = img[3].swapaxes(0,2)

plt.imshow(newshape)

plt.show()