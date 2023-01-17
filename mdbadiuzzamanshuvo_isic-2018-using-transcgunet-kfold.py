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
img_path = "../input/isic2018/ISIC2018_Task1-2_Training_Input/ISIC2018_Task1-2_Training_Input/"

mask_path = '../input/isic2018/ISIC2018_Task1_Training_GroundTruth/ISIC2018_Task1_Training_GroundTruth/'
width = 128

height = 128

channels = 3
train_img = glob(img_path + '*.jpg')

train_mask = [i.replace(img_path, mask_path).replace('.jpg', '_segmentation.png') for i in train_img]



        

print(train_img[:2],"\n" ,train_mask[:2])
# From: https://github.com/zhixuhao/unet/blob/master/data.py

from tensorflow.keras.applications.vgg16 import preprocess_input



# It contains 2594 training samples

img_files   = np.zeros([2594, height, width, channels])

mask_files   = np.zeros([2594, height, width])



print('Reading ISIC 2018')

for idx, (img_path, mask_path) in tqdm(enumerate(zip(train_img, train_mask))):

    img = cv2.imread(img_path)

    img = np.double(cv2.resize(img,(width,height)))

    img = preprocess_input(img)

    img_files[idx, :,:,:] = img



    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    mask = cv2.resize(mask,(width,height))

    mask = mask / 255

    mask[mask > 0.5] = 1

    mask[mask <= 0.5] = 0

    mask_files[idx, :,:] = mask    

         

print('Reading ISIC 2018 finished')
masks_sum = np.sum(mask_files, axis=(1,2)).reshape((-1, 1))

class_lab = (masks_sum != 0) + 0.
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



def res_block(inputs,filter_size):

    """

    res_block -- Residual block for building res path

    

    Arguments:

    inputs {<class 'tensorflow.python.framework.ops.Tensor'>} -- input for residual block

    filter_size {int} -- convolutional filter size 

    

    Returns:

    add {<class 'tensorflow.python.framework.ops.Tensor'>} -- addition of two convolutional filter output  

    """

    # First Conv2D layer

    cb1 = Conv2D(filter_size,(3,3),padding = 'same',activation="relu")(inputs)

    # Second Conv2D layer parallel to the first one

    cb2 = Conv2D(filter_size,(1,1),padding = 'same',activation="relu")(inputs)

    # Addition of cb1 and cb2

    add = Add()([cb1,cb2])

    

    return add



def res_path(inputs,filter_size,path_number):

    """

    res_path -- residual path / modified skip connection

    

    Arguments:

    inputs {<class 'tensorflow.python.framework.ops.Tensor'>} -- input for res path

    filter_size {int} -- convolutional filter size 

    path_number {int} -- path identifier 

    

    Returns:

    skip_connection {<class 'tensorflow.python.framework.ops.Tensor'>} -- final res path

    """

    # Minimum one residual block for every res path

    skip_connection = res_block(inputs, filter_size)



    # Two serial residual blocks for res path 2

    if path_number == 2:

        skip_connection = res_block(skip_connection,filter_size)

    

    # Three serial residual blocks for res path 1

    elif path_number == 1:

        skip_connection = res_block(skip_connection,filter_size)

        skip_connection = res_block(skip_connection,filter_size)

    

    return skip_connection



def decoder_block(inputs, res, out_channels, depth):

    

    """

    decoder_block -- decoder block formation

    

    Arguments:

    inputs {<class 'tensorflow.python.framework.ops.Tensor'>} -- input for decoder block

    mid_channels {int} -- no. of mid channels 

    out_channels {int} -- no. of out channels

    

    Returns:

    db {<class 'tensorflow.python.framework.ops.Tensor'>} -- returning the decoder block

    """

    conv_kwargs = dict(

        activation='relu',

        padding='same',

        kernel_initializer='he_normal',

        data_format='channels_last'  

    )

    

    # UpConvolutional layer

    db = UpSampling2D((2, 2), interpolation='bilinear')(inputs)

    db = concatenate([db, res], axis=3)

    # First conv2D layer 

    db = Conv2D(out_channels, 3, **conv_kwargs)(db)

    # Second conv2D layer

    db = Conv2D(out_channels, 3, **conv_kwargs)(db)



    if depth > 2:

        # Third conv2D layer

        db = Conv2D(out_channels, 3, **conv_kwargs)(db)



    return db



def TransCGUNet(input_size=(512, 512, 1)):

    """

    TransResUNet -- main architecture of TransResUNet

    

    Arguments:

    input_size {tuple} -- size of input image

    

    Returns:

    model {<class 'tensorflow.python.keras.engine.training.Model'>} -- final model

    """

    

    # Input 

    inputs = Input(input_size)



    # VGG16 with imagenet weights

    encoder = VGG16(include_top=False, weights='imagenet', input_shape=input_size)

       

    # First encoder block

    enc1 = encoder.get_layer(name='block1_conv1')(inputs)

    enc1 = encoder.get_layer(name='block1_conv2')(enc1)

    enc2 = MaxPooling2D(pool_size=(2, 2))(enc1)

    

    # Second encoder block

    enc2 = encoder.get_layer(name='block2_conv1')(enc2)

    enc2 = encoder.get_layer(name='block2_conv2')(enc2)

    enc3 = MaxPooling2D(pool_size=(2, 2))(enc2)

    

    # Third encoder block

    enc3 = encoder.get_layer(name='block3_conv1')(enc3)

    enc3 = encoder.get_layer(name='block3_conv2')(enc3)

    enc3 = encoder.get_layer(name='block3_conv3')(enc3)

    center = MaxPooling2D(pool_size=(2, 2))(enc3)



    # Center block

    center = Conv2D(512, (3, 3), activation='relu', padding='same')(center)

    center = Conv2D(512, (3, 3), activation='relu', padding='same')(center)

    

    # classification branch

    cls = Conv2D(32, (3,3), activation='relu', padding='same')(center)

    cls = Conv2D(1, (1,1))(cls)

    cls = GlobalAveragePooling2D()(cls)

    cls = Activation('sigmoid', name='class')(cls)

    clsr = Reshape((1, 1, 1), name='reshape')(cls)

    

    # Decoder block corresponding to third encoder

    res_path3 = res_path(enc3,128,3)

    dec3 = decoder_block(center, enc3, 256, 3)

    

    # Decoder block corresponding to second encoder

    res_path2 = res_path(enc2,64,2)

    dec2 = decoder_block(dec3, enc2, 128, 2)

    

    # Final Block concatenation with first encoded feature 

    res_path1 = res_path(enc1,32,1)

    dec1 = decoder_block(dec2, enc1, 64, 1)



    # Output

    out = Conv2D(1, 1)(dec1)

    out = Activation('sigmoid')(out)

    out = multiply(inputs=[out,clsr], name='seg')

    

    # Final model

    model = Model(inputs=[inputs], outputs=[out, cls])

    

    return model
kf = KFold(n_splits = 5, shuffle=False)



histories = []

losses = []

accuracies = []

dicecoefs = []

ious = []



EPOCHS = 120

BATCH_SIZE = 16



mask_files = mask_files[:, :, :, np.newaxis]



for k, (train_index, test_index) in enumerate(kf.split(img_files, mask_files)):

    X_train = img_files[train_index]

    y_train = mask_files[train_index]

    label_train = class_lab[train_index]

    X_test = img_files[test_index]

    y_test = mask_files[test_index]

    label_test = class_lab[test_index]

    

    model = TransCGUNet(input_size=(height,width, channels))

    model.compile(optimizer=Adam(lr=5e-6), loss={'seg':dice_coef_loss, 'class':'binary_crossentropy'}, \

                      loss_weights={'seg':1, 'class':1}, metrics={'seg':[iou, dice_coef, 'binary_accuracy'], 'class':['accuracy']}) 



    model_checkpoint = ModelCheckpoint(str(k+1) + '_unet_skin_seg.hdf5', 

                                       verbose=1, 

                                       save_best_only=True)



    history = model.fit(X_train,

                        [y_train, label_train],

                        epochs=EPOCHS,

                        callbacks=[model_checkpoint],

                        validation_data = (X_test, [y_test, label_test]))

    

    model = load_model(str(k+1) + '_unet_skin_seg.hdf5', custom_objects={'dice_coef_loss': dice_coef_loss, 'iou': iou, 'dice_coef': dice_coef})

    

    results = model.evaluate(X_test, [y_test, label_test])

    results = dict(zip(model.metrics_names,results))

    

    histories.append(history)

    accuracies.append(results['seg_binary_accuracy'])

    losses.append(results['seg_loss'])

    dicecoefs.append(results['seg_dice_coef'])

    ious.append(results['seg_iou'])
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

    index=np.random.randint(0,len(train_img))

    print(i+1, index)

    img = cv2.imread(train_img[index])

    img = cv2.resize(img, (height, width))

    img = img[np.newaxis, :, :, :]

    img = preprocess_input(img)

    pred = model.predict(img)



    plt.figure(figsize=(12,12))

    plt.subplot(1,3,1)

    plt.imshow(cv2.resize(cv2.imread(train_img[index]), (height, width)))

    plt.title('Original Image')

    plt.subplot(1,3,2)

    plt.imshow(np.squeeze(cv2.resize(cv2.imread(train_mask[index]), (height, width))))

    plt.title('Original Mask')

    plt.subplot(1,3,3)

    plt.imshow(np.squeeze(pred[0]) > .5)

    plt.title('Prediction')

    plt.show()