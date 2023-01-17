import os

import sys

import random

import warnings



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt



from tqdm import tqdm



from keras.models import Model, load_model

from keras.layers import Input

from keras.layers.core import Dropout, Lambda

from keras.layers.convolutional import Conv2D, Conv2DTranspose

from keras.layers.pooling import MaxPooling2D

from keras.layers.merge import concatenate

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras import backend as K

from keras.optimizers import Adam

import tensorflow as tf



from skimage.io import imread

from skimage.transform import resize
%matplotlib inline
TRAIN_PATH = '../input/train/'

TEST_PATH = '../input/test/'



seed = 42

random.seed = seed

np.random.seed = seed



tot_num = 5635

IMG_HEIGHT = 128

IMG_WIDTH = 128



files = os.listdir(TRAIN_PATH)

masks_list = []

imgs_list = []



for f in files:

    if 'mask' in f:

        masks_list.append(f)

    else:

        imgs_list.append(f)



masks_list = sorted(masks_list)

imgs_list = sorted(imgs_list)
X_train = np.zeros((tot_num, IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)

Y_train = np.zeros((tot_num, IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
Y_train_one = []

X_train_one = []

Y_train_zero = []

X_train_zero = []



for i, file in tqdm(enumerate(imgs_list), total=len(imgs_list)):

    img_path = file

    mask_path = img_path[:-4] + '_mask.tif'

   

    mask = imread(TRAIN_PATH + mask_path)

    mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

    

    img = imread(TRAIN_PATH + img_path)

    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

    

    if mask.any() == False:

        Y_train_zero.append(mask)

        X_train_zero.append(img)

    else:

        Y_train_one.append(mask)

        X_train_one.append(img)
# fig, ax = plt.subplots(5, 2, figsize=(10, 50))

# for i in range(5):

#     ax[i, 0].imshow(X_train_zero[i])

#     ax[i, 1].imshow(Y_train_zero[i])

#     print(np.unique(Y_train_zero[i]))
# fig, ax = plt.subplots(5, 2, figsize=(10, 50))

# for i in range():

#     ax[i, 0].imshow(X_train_one[i], 'gray')

#     Y_train_one = np.array(Y_train_one, dtype='bool')

# #     Y_train_one[i][Y_train_one]=1

#     ax[i, 1].imshow(Y_train_one[i], 'gray')

#     print(np.unique(Y_train_one[i]))
X_train_one = np.array(X_train_one)

Y_train_one = np.array(Y_train_one)

X_train_zero = np.array(X_train_zero)

Y_train_zero = np.array(Y_train_zero)
X_train = []

Y_train = []
def augmentation(imgs, masks):

    for img, mask in zip(imgs, masks):

        img_lr = np.fliplr(img)

        mask_lr = np.fliplr(mask)

        img_up = np.flipud(img)

        mask_up = np.flipud(mask)

        #img_lr_up = np.flipud(img_lr)

        #mask_lr_up = np.flipud(mask_lr)

        #img_up_lr = np.fliplr(img_up)

        #mask_up_lr = np.fliplr(mask_up)

        X_train.append(img)

        Y_train.append(mask)

        X_train.append(img_lr)

        Y_train.append(mask_lr)

        X_train.append(img_up)

        Y_train.append(mask_up)

        #X_train.append(img_lr_up)

        #Y_train.append(mask_lr_up)

        #X_train.append(img_up_lr)

        #Y_train.append(mask_up_lr)

        

augmentation(X_train_one, Y_train_one)
# for img, mask in zip(X_train_zero, Y_train_zero):

#     X_train.append(img)

#     Y_train.append(mask)
X_train = np.array(X_train)

Y_train = np.array(Y_train)
# X_train_all = np.concatenate((X_train, X_train_zero[:1000]), axis=0)

# Y_train_all = np.concatenate((Y_train, Y_train_zero[:1000]), axis=0)
X_train_ax = X_train[:,:,:,np.newaxis]/255.

Y_train_ax = Y_train[:,:,:,np.newaxis]/255.
def to_one(x):

    if x==0:

        return 0

    else:

        return 1



to_one = np.vectorize(to_one)



Y_train_ax = to_one(Y_train_ax)
smooth = 1.



def dice_coef(y_true, y_pred):

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)





def dice_coef_loss(y_true, y_pred):

    return -dice_coef(y_true, y_pred)
IMG_CHANNELS = 1
!pip3 install git+https://github.com/qubvel/segmentation_models

from segmentation_models import Unet



# model = Unet('densenet121',encorder_weights='imagenet',freeze_encorder=True)
# model = Unet('densenet121',encorder_weights='imagenet',freeze_encorder=True)
# N = 1



# base_model = Unet(backbone_name='resnet34', encoder_weights='imagenet')



# inp = Input(shape=(None, None, N))

# l1 = Conv2D(3, (1, 1))(inp) # map N channels data to 3 channels

# out = base_model(l1)



# model = Model(inp, out, name=base_model.name)

# model.compile(optimizer=Adam(lr = 1e-5), loss=dice_coef_loss, metrics=[dice_coef])
inputs = Input((IMG_HEIGHT, IMG_WIDTH, 1))



conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)

conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)

pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)



conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)

conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)

pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)



conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)

conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)

pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)



conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)

conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)

pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)



conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)

conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)



up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)

conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)

conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)



up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)

conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)

conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)



up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)

conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)

conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)



up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)

conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)

conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)



conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)



model = Model(inputs=[inputs], outputs=[conv10])

model.compile(optimizer=Adam(lr = 1e-5), loss=dice_coef_loss, metrics=[dice_coef])
results = model.fit(X_train_ax, Y_train_ax, validation_split=0.1, batch_size=8, epochs=18)
import re



sub = pd.read_csv("../input/sample_submission.csv")

test_list = os.listdir("../input/test")



print("The number of test data : ", len(test_list))



# Sort the test set in ascending order.

reg = re.compile("[0-9]+")



temp1 = list(map(lambda x: reg.match(x).group(), test_list)) 

temp1 = list(map(int, temp1))



test_list = [x for _,x in sorted(zip(temp1, test_list))]



test_list[:15]



import cv2

import re



X_test = np.empty((len(test_list), IMG_HEIGHT, IMG_WIDTH), dtype = 'float32')

for i, item in enumerate(test_list):

    image = cv2.imread("../input/test/" + item, 0)

    image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation = cv2.INTER_AREA)

    X_test[i] = image

X_test = X_test[:,:,:,np.newaxis] / 255



y_pred = model.predict(X_test)



def run_length_enc(label):

    from itertools import chain

    x = label.transpose().flatten()

    y = np.where(x > 0)[0]

    if len(y) < 10:  # consider as empty

        return ''

    z = np.where(np.diff(y) > 1)[0]

    start = np.insert(y[z+1], 0, y[0])

    end = np.append(y[z], y[-1])

    length = end - start

    res = [[s+1, l+1] for s, l in zip(list(start), list(length))]

    res = list(chain.from_iterable(res))

    return ' '.join([str(r) for r in res])



rles = []

for i in range(X_test.shape[0]):

    img = y_pred[i, :, :, 0]

    img = img > 0.5

    img = resize(img, (420, 580), preserve_range=True)

    rle = run_length_enc(img)

    rles.append(rle)

    if i % 100 == 0:

            print('{}/{}'.format(i, X_test.shape[0]), end = "\r")

            

sub['pixels'] = rles

sub.to_csv("submission.csv", index = False)