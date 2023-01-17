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

from tensorflow.keras.layers import Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, add, Reshape

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
def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None):

    '''

    2D Convolutional layers

    

    Arguments:

        x {keras layer} -- input layer 

        filters {int} -- number of filters

        num_row {int} -- number of rows in filters

        num_col {int} -- number of columns in filters

    

    Keyword Arguments:

        padding {str} -- mode of padding (default: {'same'})

        strides {tuple} -- stride of convolution operation (default: {(1, 1)})

        activation {str} -- activation function (default: {'relu'})

        name {str} -- name of the layer (default: {None})

    

    Returns:

        [keras layer] -- [output layer]

    '''



    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(x)

    x = BatchNormalization(axis=3, scale=False)(x)



    if(activation == None):

        return x



    x = Activation(activation, name=name)(x)



    return x





def trans_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(2, 2), name=None):

    '''

    2D Transposed Convolutional layers

    

    Arguments:

        x {keras layer} -- input layer 

        filters {int} -- number of filters

        num_row {int} -- number of rows in filters

        num_col {int} -- number of columns in filters

    

    Keyword Arguments:

        padding {str} -- mode of padding (default: {'same'})

        strides {tuple} -- stride of convolution operation (default: {(2, 2)})

        name {str} -- name of the layer (default: {None})

    

    Returns:

        [keras layer] -- [output layer]

    '''



    x = Conv2DTranspose(filters, (num_row, num_col), strides=strides, padding=padding)(x)

    x = BatchNormalization(axis=3, scale=False)(x)

    

    return x





def MultiResBlock(U, inp, alpha = 1.67):

    '''

    MultiRes Block

    

    Arguments:

        U {int} -- Number of filters in a corrsponding UNet stage

        inp {keras layer} -- input layer 

    

    Returns:

        [keras layer] -- [output layer]

    '''



    W = alpha * U



    shortcut = inp



    shortcut = conv2d_bn(shortcut, int(W*0.167) + int(W*0.333) +

                         int(W*0.5), 1, 1, activation=None, padding='same')



    conv3x3 = conv2d_bn(inp, int(W*0.167), 3, 3,

                        activation='relu', padding='same')



    conv5x5 = conv2d_bn(conv3x3, int(W*0.333), 3, 3,

                        activation='relu', padding='same')



    conv7x7 = conv2d_bn(conv5x5, int(W*0.5), 3, 3,

                        activation='relu', padding='same')



    out = concatenate([conv3x3, conv5x5, conv7x7], axis=3)

    out = BatchNormalization(axis=3)(out)



    out = add([shortcut, out])

    out = Activation('relu')(out)

    out = BatchNormalization(axis=3)(out)



    return out





def ResPath(filters, length, inp):

    '''

    ResPath

    

    Arguments:

        filters {int} -- [description]

        length {int} -- length of ResPath

        inp {keras layer} -- input layer 

    

    Returns:

        [keras layer] -- [output layer]

    '''





    shortcut = inp

    shortcut = conv2d_bn(shortcut, filters, 1, 1,

                         activation=None, padding='same')



    out = conv2d_bn(inp, filters, 3, 3, activation='relu', padding='same')



    out = add([shortcut, out])

    out = Activation('relu')(out)

    out = BatchNormalization(axis=3)(out)



    for i in range(length-1):



        shortcut = out

        shortcut = conv2d_bn(shortcut, filters, 1, 1,

                             activation=None, padding='same')



        out = conv2d_bn(out, filters, 3, 3, activation='relu', padding='same')



        out = add([shortcut, out])

        out = Activation('relu')(out)

        out = BatchNormalization(axis=3)(out)



    return out





def MultiResUnet(input_size=(256,256,1)):

    '''

    MultiResUNet

    

    Arguments:

        height {int} -- height of image 

        width {int} -- width of image 

        n_channels {int} -- number of channels in image

    

    Returns:

        [keras model] -- MultiResUNet model

    '''





    inputs = Input(input_size)



    mresblock1 = MultiResBlock(32, inputs)

    pool1 = MaxPooling2D(pool_size=(2, 2))(mresblock1)

    mresblock1 = ResPath(32, 4, mresblock1)



    mresblock2 = MultiResBlock(32*2, pool1)

    pool2 = MaxPooling2D(pool_size=(2, 2))(mresblock2)

    mresblock2 = ResPath(32*2, 3, mresblock2)



    mresblock3 = MultiResBlock(32*4, pool2)

    pool3 = MaxPooling2D(pool_size=(2, 2))(mresblock3)

    mresblock3 = ResPath(32*4, 2, mresblock3)



    mresblock4 = MultiResBlock(32*8, pool3)

    pool4 = MaxPooling2D(pool_size=(2, 2))(mresblock4)

    mresblock4 = ResPath(32*8, 1, mresblock4)



    mresblock5 = MultiResBlock(32*16, pool4)



    up6 = concatenate([Conv2DTranspose(

        32*8, (2, 2), strides=(2, 2), padding='same')(mresblock5), mresblock4], axis=3)

    mresblock6 = MultiResBlock(32*8, up6)



    up7 = concatenate([Conv2DTranspose(

        32*4, (2, 2), strides=(2, 2), padding='same')(mresblock6), mresblock3], axis=3)

    mresblock7 = MultiResBlock(32*4, up7)



    up8 = concatenate([Conv2DTranspose(

        32*2, (2, 2), strides=(2, 2), padding='same')(mresblock7), mresblock2], axis=3)

    mresblock8 = MultiResBlock(32*2, up8)



    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(

        2, 2), padding='same')(mresblock8), mresblock1], axis=3)

    mresblock9 = MultiResBlock(32, up9)



    conv10 = conv2d_bn(mresblock9, 1, 1, 1, activation='sigmoid')

    

    model = Model(inputs=[inputs], outputs=[conv10])



    return model
from sklearn.model_selection import KFold

kf = KFold(n_splits = 5, shuffle=False)



BATCH_SIZE = 16

EPOCHS = 50
histories = []

losses = []

accuracies = []

dicecoefs = []

ious = []



for k, (train_index, test_index) in enumerate(kf.split(X_train, Y_train)):

    x_train = X_train[train_index]

    y_train = Y_train[train_index]

    x_test = X_train[test_index]

    y_test = Y_train[test_index]



    model = MultiResUnet(input_size=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    

    model.compile(optimizer='adam', loss=dice_coef_loss, metrics=[iou, dice_coef, 'binary_accuracy'])

    model_checkpoint = ModelCheckpoint(str(k+1) + '_unet_nuclei_seg.hdf5', 

                                       monitor='loss', 

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
model = load_model('1_unet_nuclei_seg.hdf5', custom_objects={'dice_coef_loss': dice_coef_loss, 'iou': iou, 'dice_coef': dice_coef})
for i in range(20):

    index=np.random.randint(1,len(X_train))

    pred=model.predict(X_train[index][np.newaxis, :, :, :])



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