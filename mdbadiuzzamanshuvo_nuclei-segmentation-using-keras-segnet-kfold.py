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

from tensorflow.keras.layers import Input, Activation, BatchNormalization, Dropout, Lambda, Dense, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Add, UpSampling2D, Reshape, GlobalAveragePooling2D

from tensorflow.keras.optimizers import Adam, SGD

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



from tensorflow.keras import backend as K

from tensorflow.keras.preprocessing.image import ImageDataGenerator



# Set some parameters

IMG_WIDTH = 256

IMG_HEIGHT = 256

IMG_CHANNELS = 3

TRAIN_PATH = '../input/mydata-ns2/stage1_train/'

TEST_PATH = '../input/mydata-ns2/stage1_test/'



warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

seed = 42

random.seed = seed

np.random.seed = seed
tf.__version__
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
# X_train = X_train / 255

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

    x = Conv2D(1024, (3, 3), padding='same', name='deconv1')(x)

    x = Activation('relu')(x)

    x = BatchNormalization()(x)

    x = Conv2D(1024, (3, 3), padding='same', name='deconv2')(x)

    x = Activation('relu')(x)

    x = BatchNormalization()(x)

    x = Conv2D(1024, (3, 3), padding='same', name='deconv3')(x)

    x = Activation('relu')(x)

    x = BatchNormalization()(x)

    

    x = UpSampling2D()(x)

    x = Conv2D(512, (3, 3), padding='same', name='deconv4')(x)

    x = Activation('relu')(x)

    x = BatchNormalization()(x)

    x = Conv2D(512, (3, 3), padding='same', name='deconv5')(x)

    x = Activation('relu')(x)

    x = BatchNormalization()(x)

    x = Conv2D(512, (3, 3), padding='same', name='deconv6')(x)

    x = Activation('relu')(x)

    x = BatchNormalization()(x)



    x = UpSampling2D()(x)

    x = Conv2D(256, (3, 3), padding='same', name='deconv7')(x)

    x = Activation('relu')(x)

    x = BatchNormalization()(x)

    x = Conv2D(256, (3, 3), padding='same', name='deconv8')(x)

    x = Activation('relu')(x)

    x = BatchNormalization()(x)

    x = Conv2D(256, (3, 3), padding='same', name='deconv9')(x)

    x = Activation('relu')(x)

    x = BatchNormalization()(x)



    x = UpSampling2D()(x)

    x = Conv2D(128, (3, 3), padding='same', name='deconv10')(x)

    x = Activation('relu')(x)

    x = BatchNormalization()(x)

    x = Conv2D(128, (3, 3), padding='same', name='deconv11')(x)

    x = Activation('relu')(x)

    x = BatchNormalization()(x)

    

    x = UpSampling2D()(x)

    x = Conv2D(64, (3, 3), padding='same', name='deconv12')(x)

    x = Activation('relu')(x)

    x = BatchNormalization()(x)

    x = Conv2D(64, (3, 3), padding='same', name='deconv13')(x)

    x = Activation('relu')(x)

    x = BatchNormalization()(x)



    x = Conv2D(1, (3, 3), padding='same', name='deconv14')(x)

    x = Activation('sigmoid')(x)

    pred = Reshape((IMG_HEIGHT,IMG_WIDTH))(x)

    

    return Model(inputs=inp, outputs=pred)
from sklearn.model_selection import KFold

kf = KFold(n_splits = 5, shuffle=False)



BATCH_SIZE = 16

EPOCHS = 100
from tensorflow.keras.applications.vgg16 import preprocess_input

X_train_process = preprocess_input(X_train)
histories = []

losses = []

accuracies = []

dicecoefs = []

ious = []



for k, (train_index, test_index) in enumerate(kf.split(X_train, Y_train)):

    print('\nFold : ', k+1)

    x_train = X_train_process[train_index]

    y_train = Y_train[train_index]

    x_test = X_train_process[test_index]

    y_test = Y_train[test_index]



    model = segnet(input_size=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[iou, dice_coef, 'binary_accuracy'])

    model.summary()

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