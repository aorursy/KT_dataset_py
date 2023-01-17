from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
import os

import re

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import cv2

from tqdm import tqdm

from glob import glob

from PIL import Image

from skimage.transform import resize

from sklearn.model_selection import train_test_split, KFold



import tensorflow as tf

import tensorflow.keras

from tensorflow.keras import backend as K

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



K.set_image_data_format('channels_last')
img_path = "../input/isic2018/ISIC2018_Task1-2_Training_Input/ISIC2018_Task1-2_Training_Input/"

mask_path = '../input/isic2018/ISIC2018_Task1_Training_GroundTruth/ISIC2018_Task1_Training_GroundTruth/'
width = 128

height = 128

channels = 3
train_img = glob(img_path + '*.jpg')

train_mask = [i.replace(img_path, mask_path).replace('.jpg', '_segmentation.png') for i in train_img]



        

print(train_img[:2],"\n" ,train_mask[:2])
# It contains 2594 training samples

img_files   = np.zeros([2594, height, width, channels])

mask_files   = np.zeros([2594, height, width])



print('Reading ISIC 2018')

for idx, (img_path, mask_path) in tqdm(enumerate(zip(train_img, train_mask))):

    img = cv2.imread(img_path)

    img = np.double(cv2.resize(img,(width,height)))

    img = img / 255

    img_files[idx, :,:,:] = img



    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    mask = cv2.resize(mask,(width,height))

    mask = mask / 255

    mask[mask > 0.5] = 1

    mask[mask <= 0.5] = 0

    mask_files[idx, :,:] = mask    

         

print('Reading ISIC 2018 finished')
# Display the first image and mask of the first subject.

image1 = np.array(Image.open(train_img[0]))

image1_mask = np.array(Image.open(train_mask[0]))

image1_mask = np.ma.masked_where(image1_mask == 0, image1_mask)



fig, ax = plt.subplots(1,3,figsize = (16,12))

ax[0].imshow(image1, cmap = 'gray')



ax[1].imshow(image1_mask, cmap = 'gray')



ax[2].imshow(image1, cmap = 'gray', interpolation = 'none')

ax[2].imshow(image1_mask, cmap = 'jet', interpolation = 'none', alpha = 0.7)
from tensorflow.keras.models import Model, load_model

from tensorflow.keras import Input

from tensorflow.keras.layers import Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate,add

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
def dice_coef(y_true, y_pred):

    smooth = 0.0

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)



def iou(y_true, y_pred):

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum ( y_true_f * y_pred_f)

    union = K.sum ( y_true_f + y_pred_f - y_true_f * y_pred_f)

    return intersection/union





def dice_coef_loss(y_true, y_pred):

    return -dice_coef(y_true, y_pred)

from tensorflow.keras.applications.vgg16 import VGG16

from tensorflow.keras.layers import *



def segnet(input_size=(512, 512, 1)):

    

    encoder = VGG16(include_top=False, weights='imagenet', input_shape=input_size)

    

    # Encoding layer

    inp = Input(input_size)

    x = encoder.get_layer(name='block1_conv1')(inp)

    x = BatchNormalization()(x)

    x = encoder.get_layer(name='block1_conv2')(x)

    x = BatchNormalization()(x)

    x = MaxPooling2D()(x)

    

    x = encoder.get_layer(name='block2_conv1')(x)

    x = BatchNormalization()(x)

    x = encoder.get_layer(name='block2_conv2')(x)

    x = BatchNormalization()(x)

    x = MaxPooling2D()(x)



    x = encoder.get_layer(name='block3_conv1')(x)

    x = BatchNormalization()(x)

    x = encoder.get_layer(name='block3_conv2')(x)

    x = BatchNormalization()(x)

    x = encoder.get_layer(name='block3_conv3')(x)

    x = BatchNormalization()(x)

    x = MaxPooling2D()(x)



    x = encoder.get_layer(name='block4_conv1')(x)

    x = BatchNormalization()(x)

    x = encoder.get_layer(name='block4_conv2')(x)

    x = BatchNormalization()(x)

    x = encoder.get_layer(name='block4_conv3')(x)

    x = BatchNormalization()(x)

    x = MaxPooling2D()(x)

    

    x = encoder.get_layer(name='block5_conv1')(x)

    x = BatchNormalization()(x)

    x = encoder.get_layer(name='block5_conv2')(x)

    x = BatchNormalization()(x)

    x = encoder.get_layer(name='block5_conv3')(x)

    x = BatchNormalization()(x)

    x = MaxPooling2D()(x)

    

    # Decoding Layer 

    x = UpSampling2D()(x)

    x = Conv2D(512, (3, 3), padding='same', name='deconv1')(x)

    x = Activation('relu')(x)

    x = BatchNormalization()(x)

    x = Conv2D(512, (3, 3), padding='same', name='deconv2')(x)

    x = Activation('relu')(x)

    x = BatchNormalization()(x)

    x = Conv2D(512, (3, 3), padding='same', name='deconv3')(x)

    x = Activation('relu')(x)

    x = BatchNormalization()(x)

    

    x = UpSampling2D()(x)

    x = Conv2D(512, (3, 3), padding='same', name='deconv4')(x)

    x = Activation('relu')(x)

    x = BatchNormalization()(x)

    x = Conv2D(512, (3, 3), padding='same', name='deconv5')(x)

    x = Activation('relu')(x)

    x = BatchNormalization()(x)

    x = Conv2D(256, (3, 3), padding='same', name='deconv6')(x)

    x = Activation('relu')(x)

    x = BatchNormalization()(x)



    x = UpSampling2D()(x)

    x = Conv2D(256, (3, 3), padding='same', name='deconv7')(x)

    x = Activation('relu')(x)

    x = BatchNormalization()(x)

    x = Conv2D(256, (3, 3), padding='same', name='deconv8')(x)

    x = Activation('relu')(x)

    x = BatchNormalization()(x)

    x = Conv2D(128, (3, 3), padding='same', name='deconv9')(x)

    x = Activation('relu')(x)

    x = BatchNormalization()(x)



    x = UpSampling2D()(x)

    x = Conv2D(128, (3, 3), padding='same', name='deconv10')(x)

    x = Activation('relu')(x)

    x = BatchNormalization()(x)

    x = Conv2D(64, (3, 3), padding='same', name='deconv11')(x)

    x = Activation('relu')(x)

    x = BatchNormalization()(x)

    

    x = UpSampling2D()(x)

    x = Conv2D(64, (3, 3), padding='same', name='deconv12')(x)

    x = Activation('relu')(x)

    x = BatchNormalization()(x)

    x = Conv2D(1, (3, 3), padding='same', name='deconv13')(x)

    x = Activation('sigmoid')(x)

    

    pred = Reshape((input_size[0],input_size[1]))(x)

    

    return Model(inputs=inp, outputs=pred)
kf = KFold(n_splits = 5, shuffle=False)



histories = []

losses = []

accuracies = []

dicecoefs = []

ious = []



EPOCHS = 120

BATCH_SIZE = 16



for k, (train_index, test_index) in enumerate(kf.split(img_files, mask_files)):

    X_train = img_files[train_index]

    y_train = mask_files[train_index]

    X_test = img_files[test_index]

    y_test = mask_files[test_index]

    

    model = segnet(input_size=(height,width, channels))

    model.compile(optimizer=Adam(lr=5e-6), loss=dice_coef_loss, \

                      metrics=[iou, dice_coef, 'binary_accuracy'])



    model_checkpoint = ModelCheckpoint(str(k+1) + '_unet_skin_seg.hdf5', 

                                       verbose=1, 

                                       save_best_only=True)



    history = model.fit(X_train,

                        y_train,

                        epochs=EPOCHS,

                        callbacks=[model_checkpoint],

                        validation_data = (X_test, y_test))

    

    model = load_model(str(k+1) + '_unet_skin_seg.hdf5', custom_objects={'dice_coef_loss': dice_coef_loss, 'iou': iou, 'dice_coef': dice_coef})

    

    results = model.evaluate(X_test, y_test)

    results = dict(zip(model.metrics_names,results))

    

    histories.append(history)

    accuracies.append(results['binary_accuracy'])

    losses.append(results['loss'])

    dicecoefs.append(results['dice_coef'])

    ious.append(results['iou'])
import pickle



for h, history in enumerate(histories):



    keys = history.history.keys()

    fig, axs = plt.subplots(1, len(keys)//2, figsize = (25, 5))

    fig.suptitle('No. ' + str(h+1) + ' Fold Results', fontsize=30)



    for k, key in enumerate(list(keys)[:len(keys)//2]):

        training = history.history[key]

        validation = history.history['val_' + key]



        epoch_count = range(1, len(training) + 1)



        axs[k].plot(epoch_count, training, 'r--')

        axs[k].plot(epoch_count, validation, 'b-')

        axs[k].legend(['Training ' + key, 'Validation ' + key])

    

    with open(str(h+1) + '_skin_trainHistoryDict', 'wb') as file_pi:

        pickle.dump(history.history, file_pi)
print('accuracies : ', accuracies)

print('losses : ', losses)

print('dicecoefs : ', dicecoefs)

print('ious : ', ious)



print('-----------------------------------------------------------------------------')

print('-----------------------------------------------------------------------------')



print('average accuracy : ', np.mean(np.array(accuracies)))

print('average loss : ', np.mean(np.array(losses)))

print('average dicecoefs : ', np.mean(np.array(dicecoefs)))

print('average ious : ', np.mean(np.array(ious)))

print()



print('standard deviation of accuracy : ', np.std(np.array(accuracies)))

print('standard deviation of loss : ', np.std(np.array(losses)))

print('standard deviation of dicecoefs : ', np.std(np.array(dicecoefs)))

print('standard deviation of ious : ', np.std(np.array(ious)))
selector = np.argmin(abs(np.array(ious) - np.mean(ious)))

model = load_model(str(selector+1) + '_unet_skin_seg.hdf5', custom_objects={'dice_coef_loss': dice_coef_loss, 'iou': iou, 'dice_coef': dice_coef})
for i in range(20):

    index=np.random.randint(0,len(img_files))

    print(i+1, index)

    img = cv2.imread(img_files[index])

    img = cv2.resize(img, (height, width))

    img = img[np.newaxis, :, :, :]

    img = img / 255

    pred = model.predict(img)



    plt.figure(figsize=(12,12))

    plt.subplot(1,3,1)

    plt.imshow(cv2.resize(cv2.imread(train_img[index]), (height, width)))

    plt.title('Original Image')

    plt.subplot(1,3,2)

    plt.imshow(np.squeeze(cv2.resize(cv2.imread(mask_files[index]), (height, width))))

    plt.title('Original Mask')

    plt.subplot(1,3,3)

    plt.imshow(np.squeeze(pred) > .5)

    plt.title('Prediction')

    plt.show()