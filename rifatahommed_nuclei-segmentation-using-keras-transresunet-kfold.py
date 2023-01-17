from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
import os

import sys

import random

import warnings



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt



from tqdm import tqdm

from itertools import chain

from skimage.io import imread, imshow, imread_collection, concatenate_images

from skimage.transform import resize

from skimage.morphology import label



import tensorflow as tf

from skimage.color import rgb2gray

from tensorflow.keras import Input

from tensorflow.keras.models import Model, load_model, save_model

from tensorflow.keras.layers import Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Add, UpSampling2D, Reshape

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



from tensorflow.keras import backend as K

from tensorflow.keras.preprocessing.image import ImageDataGenerator



# Set some parameters

IMG_WIDTH = 128

IMG_HEIGHT = 128

IMG_CHANNELS = 3

TRAIN_PATH = '../input/mydata-ns2/stage1_train/'

TEST_PATH = '../input/mydata-ns2/stage1_test/'



warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

seed = 42

random.seed = seed

np.random.seed = seed
# Get train and test IDs

train_ids = next(os.walk(TRAIN_PATH))[1]

test_ids = next(os.walk(TEST_PATH))[1]
# Get and resize train images and masks

X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

print('Getting and resizing train images and masks ... ')

sys.stdout.flush()

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):

    path = TRAIN_PATH + id_

    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]

    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

    X_train[n] = img

    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

    for mask_file in next(os.walk(path + '/masks/'))[2]:

        mask_ = imread(path + '/masks/' + mask_file)

        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', 

                                      preserve_range=True), axis=-1)

        mask = np.maximum(mask, mask_)

    Y_train[n] = mask



# Get and resize test images

X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

sizes_test = []

print('Getting and resizing test images ... ')

sys.stdout.flush()

for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):

    path = TEST_PATH + id_

    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]

    sizes_test.append([img.shape[0], img.shape[1]])

    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

    X_test[n] = img



print('Done!')
X_train = X_train / 255

Y_train = np.array(Y_train, dtype='float32')
# Check if training data looks all right

ix = random.randint(0, len(train_ids))

imshow(X_train[ix])

plt.show()

imshow(np.squeeze(Y_train[ix]))

plt.show()

# From: https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py

def dice_coef(y_true, y_pred):

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)



def dice_coef_loss(y_true, y_pred):

    return -dice_coef(y_true, y_pred)



def iou(y_true, y_pred, smooth = 100):

    intersection = K.sum(K.abs(y_true * y_pred))

    sum_ = K.sum(K.square(y_true)) + K.sum(K.square(y_pred))

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return jac
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



def decoder_block(inputs, mid_channels, out_channels):

    

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

    

    # Upsampling (nearest neighbor interpolation) layer

    db = UpSampling2D(size=(2, 2))(inputs)

    # First conv2D layer 

    db = Conv2D(mid_channels, 3, **conv_kwargs)(db)

    # Second conv2D layer

    db = Conv2D(out_channels, 3, **conv_kwargs)(db)



    return db



def TransResUNet(input_size=(512, 512, 1)):

    """

    TransResUNet -- main architecture of TransResUNet

    

    Arguments:

    input_size {tuple} -- size of input image

    

    Returns:

    model {<class 'tensorflow.python.keras.engine.training.Model'>} -- final model

    """

    

    # Input 

    inputs = Input(input_size)

    inp = inputs

    input_shape = input_size

    

    # Handling input channels 

    # input with 1 channel will be converted to 3 channels to be compatible with VGG16 pretrained encoder 

    if input_size[-1] < 3:

        inp = Conv2D(3, 1)(inputs)                         

        input_shape = (input_size[0], input_size[0], 3)  

    else:

        inp = inputs

        input_shape = input_size



    # VGG16 with imagenet weights

    encoder = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)

       

    # First encoder block

    enc1 = encoder.get_layer(name='block1_conv1')(inp)

    enc1 = encoder.get_layer(name='block1_conv2')(enc1)

    # Second encoder block

    enc2 = MaxPooling2D(pool_size=(2, 2))(enc1)

    enc2 = encoder.get_layer(name='block2_conv1')(enc2)

    enc2 = encoder.get_layer(name='block2_conv2')(enc2)

    # Third encoder block

    enc3 = MaxPooling2D(pool_size=(2, 2))(enc2)

    enc3 = encoder.get_layer(name='block3_conv1')(enc3)

    enc3 = encoder.get_layer(name='block3_conv2')(enc3)

    enc3 = encoder.get_layer(name='block3_conv3')(enc3)



    # Center block

    center = MaxPooling2D(pool_size=(2, 2))(enc3)

    center = decoder_block(center, 512, 256)



    # Decoder block corresponding to third encoder

    res_path3 = res_path(enc3,128,3)

    dec3 = concatenate([res_path3, center], axis=3)

    dec3 = decoder_block(dec3, 256, 64)

    # Decoder block corresponding to second encoder

    res_path2 = res_path(enc2,64,2)

    dec2 = concatenate([res_path2, dec3], axis=3)

    dec2 = decoder_block(dec2, 128, 64)

    # Final Block concatenation with first encoded feature 

    res_path1 = res_path(enc1,32,1)

    dec1 = concatenate([res_path1, dec2], axis=3)

    dec1 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(dec1)

    dec1 = ReLU()(dec1)

   



    # Output

    out = Conv2D(1, 1)(dec1)

    out = Activation('sigmoid')(out)  

    

    # Final model

    model = Model(inputs=[inputs], outputs=[out])

    

    return model
from sklearn.model_selection import KFold

kf = KFold(n_splits = 5, shuffle=False)



BATCH_SIZE = 16

EPOCHS = 100
histories = []

losses = []

accuracies = []

dicecoefs = []

ious = []



for k, (train_index, test_index) in enumerate(kf.split(X_train, Y_train)):

    print('\nFold : ', k+1)

    x_train = X_train[train_index]

    y_train = Y_train[train_index]

    x_test = X_train[test_index]

    y_test = Y_train[test_index]



    model = TransResUNet(input_size=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    

    model.compile(optimizer=Adam(lr = 1e-5), loss=dice_coef_loss, metrics=[iou, dice_coef, 'binary_accuracy'])

    model_checkpoint = ModelCheckpoint(str(k+1) + '_unet_nuclei_seg.hdf5',

                                       verbose=1, 

                                       save_best_only=True)



    history = model.fit(x_train, y_train,

                        epochs=EPOCHS, 

                        callbacks=[model_checkpoint],

                        validation_data = (x_test, y_test),

                        batch_size=BATCH_SIZE)

    

    model = load_model(str(k+1) + '_unet_nuclei_seg.hdf5', custom_objects={'dice_coef_loss': dice_coef_loss, 'iou': iou, 'dice_coef': dice_coef})

    

    results = model.evaluate(x_test, y_test)

    results = dict(zip(model.metrics_names,results))

    

    histories.append(history)

    accuracies.append(results['binary_accuracy'])

    losses.append(results['loss'])

    dicecoefs.append(results['dice_coef'])

    ious.append(results['iou'])
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

        

    with open(str(h+1) + '_mri_trainHistoryDict', 'wb') as file_pi:

        pickle.dump(history.history, file_pi)
selector = np.argmin(abs(np.array(ious) - np.mean(ious)))

model = load_model(str(selector+1) +'_unet_nuclei_seg.hdf5', custom_objects={'dice_coef_loss': dice_coef_loss, 'iou': iou, 'dice_coef': dice_coef})
for i in range(20):

    index=np.random.randint(1,len(X_train))

    pred=model.predict(X_train_process[index][np.newaxis, :, :, :])



    plt.figure(figsize=(12,12))

    plt.subplot(1,3,1)

    plt.imshow(X_train[index])

    plt.title('Original Image')

    plt.subplot(1,3,2)

    plt.imshow(np.squeeze(Y_train[index]))

    plt.title('Original Mask')

    plt.subplot(1,3,3)

    plt.imshow(np.squeeze(pred) > .5)

    plt.title('Predicted mask')

    plt.show()