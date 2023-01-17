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
# Folder List



!ls ../input/airbus-ship-detection
# Load necessary libraries



from matplotlib import pyplot as plt

import tensorflow as tf



# from keras.preprocessing.image import load_img



import keras.backend as K

from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping



from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization



!pip install Pillow

!pip install scipy==1.1.0

from scipy.misc.pilutil import imread # 지금은 deprecated 되어 기존의 imread에서 scipy.misc.pilutil 의 imread 함수로 대체한다

from skimage.transform import resize



from sklearn.model_selection import train_test_split



# Load truncated iamges https://www.kaggle.com/c/airbus-ship-detection/discussion/62574#latest-445141

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
# Variables



IMG_WIDTH = 768

IMG_HEIGHT = 768

IMG_CHANNELS = 3

TARGET_WIDTH = 128

TARGET_HEIGHT = 128

batch_size=10

image_shape=(768, 768)
# loading masks (RLE Encoded)



masks = pd.read_csv("../input/airbus-ship-detection/train_ship_segmentations_v2.csv")

sub_masks = pd.read_csv("../input/airbus-ship-detection/sample_submission_v2.csv") # for submission
# Pre-defined Functions

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode



# nomask -> default vector

no_mask = np.zeros(image_shape[0]*image_shape[1], dtype=np.uint8)



def rle_encode(img):

    '''

    img: numpy array, 1 - mask, 0 - background

    Returns run length as string formated

    '''

    pixels = img.flatten()

    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    runs[1::2] -= runs[::2]

    rle = ' '.join(str(x) for x in runs)

    return rle



def rle_decode(mask_rle, shape=image_shape):

    '''

    mask_rle: run-length as string formated (start length)

    shape: (height,width) of array to return 

    Returns numpy array, 1 - mask, 0 - background



    '''

    if pd.isnull(mask_rle):

        img = no_mask

        return img.reshape(shape).T

    s = mask_rle.split()

    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]



    starts -= 1

    ends = starts + lengths

    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):

        img[lo:hi] = 1

    return img.reshape(shape).T
# file_name to image vector (for model input)



def get_image(image_name):

    img = imread('../input/airbus-ship-detection/train_v2/'+image_name)[:,:,:IMG_CHANNELS]

    img = resize(img, (TARGET_WIDTH, TARGET_HEIGHT), mode='constant', preserve_range=True)

    return img



def get_test_image(image_name):

    img = imread('../input/airbus-ship-detection/test_v2/'+image_name)[:,:,:IMG_CHANNELS]

    img = resize(img, (TARGET_WIDTH, TARGET_HEIGHT), mode='constant', preserve_range=True)

    return img

    

# RLE Code to Mask Vector

def get_mask(code):

    img = rle_decode(code)

    img = resize(img, (TARGET_WIDTH, TARGET_HEIGHT, 1), mode='constant', preserve_range=True)

    return img



# U-Net Building



inputs = Input((TARGET_WIDTH , TARGET_HEIGHT, IMG_CHANNELS))



# 128



down1 = Conv2D(64, (3, 3), padding='same')(inputs)

down1 = BatchNormalization()(down1)

down1 = Activation('relu')(down1)

down1 = Conv2D(64, (3, 3), padding='same')(down1)

down1 = BatchNormalization()(down1)

down1 = Activation('relu')(down1)

down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)

# 64



down2 = Conv2D(128, (3, 3), padding='same')(down1_pool)

down2 = BatchNormalization()(down2)

down2 = Activation('relu')(down2)

down2 = Conv2D(128, (3, 3), padding='same')(down2)

down2 = BatchNormalization()(down2)

down2 = Activation('relu')(down2)

down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)

# 32



down3 = Conv2D(256, (3, 3), padding='same')(down2_pool)

down3 = BatchNormalization()(down3)

down3 = Activation('relu')(down3)

down3 = Conv2D(256, (3, 3), padding='same')(down3)

down3 = BatchNormalization()(down3)

down3 = Activation('relu')(down3)

down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)

# 16





center = Conv2D(512, (3, 3), padding='same')(down3_pool)

center = BatchNormalization()(center)

center = Activation('relu')(center)

center = Conv2D(512, (3, 3), padding='same')(center)

center = BatchNormalization()(center)

center = Activation('relu')(center)

# center





up3 = UpSampling2D((2, 2))(center)

up3 = concatenate([down3, up3], axis=3)

up3 = Conv2D(256, (3, 3), padding='same')(up3)

up3 = BatchNormalization()(up3)

up3 = Activation('relu')(up3)

up3 = Conv2D(256, (3, 3), padding='same')(up3)

up3 = BatchNormalization()(up3)

up3 = Activation('relu')(up3)

up3 = Conv2D(256, (3, 3), padding='same')(up3)

up3 = BatchNormalization()(up3)

up3 = Activation('relu')(up3)

# 32



up2 = UpSampling2D((2, 2))(up3)

up2 = concatenate([down2, up2], axis=3)

up2 = Conv2D(128, (3, 3), padding='same')(up2)

up2 = BatchNormalization()(up2)

up2 = Activation('relu')(up2)

up2 = Conv2D(128, (3, 3), padding='same')(up2)

up2 = BatchNormalization()(up2)

up2 = Activation('relu')(up2)

up2 = Conv2D(128, (3, 3), padding='same')(up2)

up2 = BatchNormalization()(up2)

up2 = Activation('relu')(up2)

# 64



up1 = UpSampling2D((2, 2))(up2)

up1 = concatenate([down1, up1], axis=3)

up1 = Conv2D(64, (3, 3), padding='same')(up1)

up1 = BatchNormalization()(up1)

up1 = Activation('relu')(up1)

up1 = Conv2D(64, (3, 3), padding='same')(up1)

up1 = BatchNormalization()(up1)

up1 = Activation('relu')(up1)

up1 = Conv2D(64, (3, 3), padding='same')(up1)

up1 = BatchNormalization()(up1)

up1 = Activation('relu')(up1)

# 128



outputs = Conv2D(1, (1, 1), activation='sigmoid')(up1)



model = Model(inputs=inputs, outputs=outputs)
# Data split for train, validate



train_df, validate_df = train_test_split(masks) 
# Compile model



from tensorflow.keras.optimizers import RMSprop



opt = RMSprop(lr=0.0001, decay=1e-6)



model.compile(loss='binary_crossentropy',

              optimizer=opt,

              metrics=['acc'])
# automating pre-processing process functions

# Creating Generators



def create_image_generator(precess_batch_size, data_df):

    while True:

        for k, group_df in data_df.groupby(np.arange(data_df.shape[0])//precess_batch_size):

            imgs = []

            labels = []

            for index, row in group_df.iterrows():

                # images

                original_img = get_image(row.ImageId) / 255.0

                # masks

                mask = get_mask(row.EncodedPixels) / 255.0

                

                imgs.append(original_img)

                labels.append(mask)

                

            imgs = np.array(imgs)

            labels = np.array(labels)

            yield imgs, labels

            

            

def create_test_generator(precess_batch_size):

    while True:

        for k, ix in sub_masks.groupby(np.arange(sub_masks.shape[0])//precess_batch_size):

            imgs = []

            labels = []

            for index, row in ix.iterrows():

                original_img = get_test_image(row.ImageId) / 255.0

                imgs.append(original_img)

                

            imgs = np.array(imgs)

            yield imgs

            

train_generator = create_image_generator(batch_size, train_df)

validate_generator = create_image_generator(batch_size, validate_df)

test_generator = create_test_generator(batch_size)
train_steps=np.ceil(float(train_df.shape[0]) / float(batch_size)).astype(int) # np.ceil -> gausse function

validate_steps=np.ceil(float(validate_df.shape[0]) / float(batch_size)).astype(int)

test_steps = np.ceil(float(sub_masks.shape[0]) / float(batch_size)).astype(int)



# start model fitting



history = model.fit_generator(

    train_generator, 

    steps_per_epoch=train_steps,

    validation_data=validate_generator,

    validation_steps=validate_steps,

    epochs=1

)
# Start Prediction



predict_mask = model.predict_generator(test_generator, steps=test_steps)
# For Submission



for index, row in sub_masks.iterrows():

    predict = predict_mask[index]

    resized_predict =  resize(predict, (IMG_WIDTH, IMG_HEIGHT)) * 255

    mask = resized_predict > 0.5

    sub_masks.at[index,'EncodedPixels'] = rle_encode(mask)

    

sub_masks.to_csv("submission.csv", index=False)