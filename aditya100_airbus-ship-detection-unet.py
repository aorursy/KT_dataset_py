# Importing Libraries



# Provides OS realted functions

import os 



# For Image Manipulation

from skimage.data import imread

from skimage.morphology import label



# For Data Manipulation and Analysis

import pandas as pd

import numpy as np



# Deep Learning Library

from keras.models import *

from keras.layers import *

from keras.optimizers import *



# Other utilities

import random

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import train_test_split
# Directories Paths

input_dir = '../input/'

train_dir = '../input/train_v2/'

test_dir = '../input/test_v2/'
# Reading Training Data

train_df = pd.read_csv(input_dir+'train_ship_segmentations_v2.csv')



train_df.head()
train_df.shape
# Removing Bug Images

train_df = train_df[train_df['ImageId'] != '6384c3e78.jpg']
train_df.shape
# Removing 10000 non-ship Images

def area_isnull(x):

    if x == x:

        return 0

    else:

        return 1
train_df['isnan'] = train_df['EncodedPixels'].apply(area_isnull)
train_df = train_df.sort_values('isnan', ascending=False)

train_df = train_df.iloc[100000:]
train_df['isnan'].value_counts()
# Helper Functions

def rle_to_mask(rle_list, SHAPE):

    tmp_flat = np.zeros(SHAPE[0]*SHAPE[1])

    if len(rle_list) == 1:

        mask = np.reshape(tmp_flat, SHAPE).T

    else:

        strt = rle_list[::2]

        length = rle_list[1::2]

        for i,v in zip(strt,length):

            tmp_flat[(int(i)-1):(int(i)-1)+int(v)] = 255

        mask = np.reshape(tmp_flat, SHAPE).T

    return mask
def calc_area_for_rle(rle_str):

    rle_list = [int(x) if x.isdigit() else x for x in str(rle_str).split()]

    if len(rle_list) == 1:

        return 0

    else:

        area = np.sum(rle_list[1::2])

        return area
train_df['area'] = train_df["EncodedPixels"].apply(calc_area_for_rle)
train_df_isship = train_df[train_df['area'] > 0]
train_df_isship.shape
train_df_smallarea = train_df_isship['area'][train_df_isship['area'] < 10]
train_df_smallarea.shape
train_df_smallarea.shape[0]/train_df_isship.shape[0]
train_gp = train_df.groupby('ImageId').sum()

train_gp = train_gp.reset_index()
train_gp.head()
def calc_class(area):

    area = area / (768*768)

    if area == 0:

        return 0

    elif area < 0.005:

        return 1

    elif area < 0.015:

        return 2

    elif area < 0.025:

        return 3

    elif area < 0.035:

        return 4

    elif area < 0.045:

        return 5

    else:

        return 6
train_gp['class'] = train_gp['area'].apply(calc_class)
train_gp['class'].value_counts()
train, val = train_test_split(train_gp, test_size=0.01, stratify=train_gp['class'].tolist())
train_isship_list = train['ImageId'][train['isnan']==0].tolist()

train_isship_list = random.sample(train_isship_list, len(train_isship_list))

train_nanship_list = train['ImageId'][train['isnan']==1].tolist()

train_nanship_list = random.sample(train_nanship_list, len(train_nanship_list))

# data generator



def mygenerator(isship_list, nanship_list, batch_size, cap_num):

    train_img_names_nanship = isship_list[:cap_num]

    train_img_names_isship = nanship_list[:cap_num]

    k = 0

    while True:

        if k+batch_size//2 >= cap_num:

            k = 0

        batch_img_names_nan = train_img_names_nanship[k:k+batch_size//2]

        batch_img_names_is = train_img_names_isship[k:k+batch_size//2]

        batch_img = []

        batch_mask = []

        for name in batch_img_names_nan:

            tmp_img = imread(train_dir + name)

            batch_img.append(tmp_img)

            mask_list = train_df['EncodedPixels'][train_df['ImageId'] == name].tolist()

            one_mask = np.zeros((768, 768, 1))

            for item in mask_list:

                rle_list = str(item).split()

                tmp_mask = rle_to_mask(rle_list, (768, 768))

                one_mask[:,:,0] += tmp_mask

            batch_mask.append(one_mask)

        for name in batch_img_names_is:

            tmp_img = imread(train_dir + name)

            batch_img.append(tmp_img)

            mask_list = train_df['EncodedPixels'][train_df['ImageId'] == name].tolist()

            one_mask = np.zeros((768, 768, 1))

            for item in mask_list:

                rle_list = str(item).split()

                tmp_mask = rle_to_mask(rle_list, (768, 768))

                one_mask[:,:,0] += tmp_mask

            batch_mask.append(one_mask)

        img = np.stack(batch_img, axis=0)

        mask = np.stack(batch_mask, axis=0)

        img = img / 255.0

        mask = mask / 255.0

        k += batch_size//2

        yield img, mask

BATCH_SIZE = 2

CAP_NUM = min(len(train_isship_list),len(train_nanship_list))

datagen = mygenerator(train_isship_list, train_nanship_list, batch_size=BATCH_SIZE, cap_num=CAP_NUM)
inputs = Input(shape=(768,768,3))

conv0 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)

conv0 = BatchNormalization()(conv0)

conv0 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv0)

conv0 = BatchNormalization()(conv0)



comp0 = AveragePooling2D((6,6))(conv0)

conv1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(comp0)

conv1 = BatchNormalization()(conv1)

conv1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)

conv1 = BatchNormalization()(conv1)

conv1 = Dropout(0.4)(conv1)



pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)

conv2 = BatchNormalization()(conv2)

conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)

conv2 = BatchNormalization()(conv2)

conv2 = Dropout(0.4)(conv2)



pool2 = MaxPooling2D(pool_size=(2,2))(conv2)

conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)

conv3 = BatchNormalization()(conv3)

conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)

conv3 = BatchNormalization()(conv3)

conv3 = Dropout(0.4)(conv3)



pool3 = MaxPooling2D(pool_size=(2,2))(conv3)

conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)

conv4 = BatchNormalization()(conv4)

conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)

conv4 = BatchNormalization()(conv4)

conv4 = Dropout(0.4)(conv4)



pool4 = MaxPooling2D(pool_size=(2,2))(conv4)

conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)

conv5 = BatchNormalization()(conv5)

conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

conv5 = BatchNormalization()(conv5)



upcv6 = UpSampling2D(size=(2,2))(conv5)

upcv6 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(upcv6)

upcv6 = BatchNormalization()(upcv6)

mrge6 = concatenate([conv4, upcv6], axis=3)

conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mrge6)

conv6 = BatchNormalization()(conv6)

conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

conv6 = BatchNormalization()(conv6)



upcv7 = UpSampling2D(size=(2,2))(conv6)

upcv7 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(upcv7)

upcv7 = BatchNormalization()(upcv7)

mrge7 = concatenate([conv3, upcv7], axis=3)

conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mrge7)

conv7 = BatchNormalization()(conv7)

conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

conv7 = BatchNormalization()(conv7)



upcv8 = UpSampling2D(size=(2,2))(conv7)

upcv8 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(upcv8)

upcv8 = BatchNormalization()(upcv8)

mrge8 = concatenate([conv2, upcv8], axis=3)

conv8 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mrge8)

conv8 = BatchNormalization()(conv8)

conv8 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

conv8 = BatchNormalization()(conv8)



upcv9 = UpSampling2D(size=(2,2))(conv8)

upcv9 = Conv2D(16, 2, activation='relu', padding='same', kernel_initializer='he_normal')(upcv9)

upcv9 = BatchNormalization()(upcv9)

mrge9 = concatenate([conv1, upcv9], axis=3)

conv9 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mrge9)

conv9 = BatchNormalization()(conv9)

conv9 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

conv9 = BatchNormalization()(conv9)



dcmp10 = UpSampling2D((6,6), interpolation='bilinear')(conv9)

mrge10 = concatenate([dcmp10, conv0], axis=3)

conv10 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mrge10)

conv10 = BatchNormalization()(conv10)

conv10 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv10)

conv10 = BatchNormalization()(conv10)

conv11 = Conv2D(1, 1, activation='sigmoid')(conv10)



model = Model(inputs=inputs, outputs=conv11)

model.compile(optimizer='adam', loss='binary_crossentropy')
history = model.fit_generator(datagen, steps_per_epoch = 500, epochs = 10)