import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

from tensorflow.keras import applications

from tensorflow.keras.models import Sequential, Model, load_model

from tensorflow.keras.losses import BinaryCrossentropy, binary_crossentropy

from tensorflow.keras.layers import Dropout, Flatten, Dense, Conv2D, Conv2DTranspose, MaxPooling2D, Input, RepeatVector, Reshape, concatenate, UpSampling2D

from tensorflow.keras import optimizers

from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras import backend as K

from tqdm.notebook import tqdm

from sklearn.model_selection import train_test_split

import requests

from PIL import Image

from io import BytesIO

import cv2

import pandas as pd

import numpy as np

import os
!pip install googledrivedownloader

from google_drive_downloader import GoogleDriveDownloader as gdd

gdd.download_file_from_google_drive(file_id='1KVfrTao_0XzUWLI4-JgGZJOLE0ddjwto', dest_path='./data/file.zip', unzip=True)
lst   = os.listdir('/kaggle/working/data/images/')

mask = []

img = []

for filename in lst:

    if filename.endswith('.jpg'):

        img.append('/kaggle/working/data/images/' + filename)

    if filename.endswith('.png'):

        mask.append('/kaggle/working/data/images/' + filename)



img.sort()

mask.sort()

#img = img[:1000]

#masks = mask[:1000]



df = pd.DataFrame({'Filepath_Image':img, 'Filepath_Mask':mask})
img = img[:1000]

mask = mask[:1000]
x = np.zeros((1000, 256, 256, 1), dtype=np.float32)

y = np.zeros((1000, 256, 256, 1), dtype=np.int8)



for i in tqdm(range(len(img))):

    imx = cv2.imread(img[i], 0)

    imx = cv2.resize(imx, (256, 256))

    imx = imx / 255.0

    x[i,:,:, 0] = imx

    

for i in tqdm(range(len(mask))):

    msk = cv2.imread(mask[i], 0)

    msk = (msk!=2)*1.0

    msk = cv2.resize(msk, (256, 256))

    msk = 1.0*(msk[:,:]>0.2)

    #msk = msk / 3.0

    y[i,:,:,0] = msk
fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(10,12))



ax[0, 0].imshow(x[0, :, :, 0], cmap='gray', vmin=0, vmax=1)

ax[0, 0].set_xticks([]), ax[0, 0].set_yticks([])

ax[0, 1].imshow(y[0, :, :, 0], cmap='gray', vmin=0, vmax=1)

ax[0, 1].set_xticks([]), ax[0, 1].set_yticks([])

ax[1, 0].imshow(x[100, :, :, 0], cmap='gray', vmin=0, vmax=1)

ax[1, 0].set_xticks([]), ax[1, 0].set_yticks([])

ax[1, 1].imshow(y[100, :, :, 0], cmap='gray', vmin=0, vmax=1)

ax[1, 1].set_xticks([]), ax[1, 1].set_yticks([])

ax[2, 0].imshow(x[800, :, :, 0], cmap='gray', vmin=0, vmax=1)

ax[2, 0].set_xticks([]), ax[2, 0].set_yticks([])

ax[2, 1].imshow(y[800, :, :, 0], cmap='gray', vmin=0, vmax=1)

ax[2, 1].set_xticks([]), ax[2, 1].set_yticks([])

plt.show()
def iou_score(x, y, smooth = 1e-6):

    intersection = tf.reduce_sum(x * y)

    union = tf.reduce_sum(x + y) - intersection

    return (intersection + smooth) / (union + smooth)

def iou_loss(x,y, smooth = 1e-6):

    intersection = tf.reduce_sum(x * y)

    union = tf.reduce_sum(x + y) - intersection

    return 1 - ((intersection + smooth) / (union + smooth))





def dice_coef(y_true, y_pred, smooth=1):

    """

    Dice = (2*|X & Y|)/ (|X|+ |Y|)

         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))

    ref: https://arxiv.org/pdf/1606.04797v1.pdf

    """

    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)

    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)



def dice_coef_loss(y_true, y_pred):

    return 1-dice_coef(y_true, y_pred)



def jaccard_distance_loss(y_true, y_pred, smooth=100):

    """

    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)

            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

    

    The jaccard distance loss is usefull for unbalanced datasets. This has been

    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing

    gradient.

    

    Ref: https://en.wikipedia.org/wiki/Jaccard_index

    

    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96

    @author: wassname

    """

    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)

    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return (1 - jac) * smooth



def DiceBCELoss(targets, inputs, smooth=1e-6):    

       

    #flatten label and prediction tensors

    inputs = K.flatten(inputs)

    targets = K.flatten(targets)

    

    BCE = 0.5# binary_crossentropy(targets, inputs)

    #intersection = K.sum((targets * inputs))

    intersection = K.sum(K.abs(targets * inputs), axis=-1)

    dice_loss = 1 - (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)

    Dice_BCE = BCE + dice_loss

    

    return Dice_BCE
def unet(pretrained_weights = None, input_size = (256,256,1)):

    inputs = Input(input_size)

    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)

    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)

    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)

    drop4 = Dropout(0.5)(conv4)

    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)



    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)

    drop5 = Dropout(0.5)(conv5)



    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))

    merge6 = concatenate([drop4,up6], axis = 3)

    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)

    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)



    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))

    merge7 = concatenate([conv3,up7], axis = 3)

    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)

    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)



    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))

    merge8 = concatenate([conv2,up8], axis = 3)

    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)

    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)



    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))

    merge9 = concatenate([conv1,up9], axis = 3)

    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)

    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    

    model = Model(inputs = inputs, outputs = conv10)



    model.compile(optimizer = optimizers.SGD(lr=1e-4, momentum=0.9), loss = jaccard_distance_loss, metrics = [iou_score, dice_coef])

    

    model.summary()



    if(pretrained_weights):

        model.load_weights(pretrained_weights)



    return model

model = unet()
#model_checkpoint = ModelCheckpoint('unet_m.hdf5', monitor='loss', verbose=1, save_best_only=True)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'loss', factor = 0.2, patience = 1, verbose = 1, min_delta=0.05, min_lr = 1e-8)

model.fit(x, y, batch_size = 16 , epochs = 30, callbacks=[ModelCheckpoint('modelx.model', monitor='acc'), reduce_lr])
to_predict = np.asarray([x[0], x[100], x[700]])

out = model.predict(to_predict)
fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(10,10))



ax[0, 0].imshow(to_predict[0,:,:,0], cmap='gray', vmin=0, vmax=1)

ax[0, 0].set_xticks([]), ax[0, 0].set_yticks([])

ax[0, 1].imshow(out[0,:,:,0], cmap='gray', vmin=0, vmax=1)

ax[0, 1].set_xticks([]), ax[0, 1].set_yticks([])



ax[1, 0].imshow(to_predict[1,:,:,0], cmap='gray', vmin=0, vmax=1)

ax[1, 0].set_xticks([]), ax[1, 0].set_yticks([])

ax[1, 1].imshow(out[1,:,:,0], cmap='gray', vmin=0, vmax=1)

ax[1, 1].set_xticks([]), ax[1, 1].set_yticks([])



ax[2, 0].imshow(to_predict[2,:,:,0], cmap='gray', vmin=0, vmax=1)

ax[2, 0].set_xticks([]), ax[2, 0].set_yticks([])

ax[2, 1].imshow(out[2,:,:,0], cmap='gray', vmin=0, vmax=1)

ax[2, 1].set_xticks([]), ax[2, 1].set_yticks([])

plt.show()