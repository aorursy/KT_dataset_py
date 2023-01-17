# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import sys

import random

import warnings

print(os.listdir("../input"))



import numpy as np

import pandas as pd

import keras



import matplotlib.pyplot as plt



from tqdm import tqdm

from itertools import chain

from skimage.io import imread, imshow, imread_collection, concatenate_images

from skimage.transform import resize

from skimage.morphology import label

from sklearn.model_selection import train_test_split

from PIL import Image



from keras.models import Model, load_model

from keras.preprocessing.image import save_img 

from keras.layers import Input

from keras.layers.core import Dropout, Lambda

from keras.layers.convolutional import Conv2D, Conv2DTranspose

from keras.layers.pooling import MaxPooling2D

from keras.layers.merge import concatenate

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras import backend as K



import tensorflow as tf



# Set some parameters

IMG_WIDTH = 128

IMG_HEIGHT = 128

IMG_CHANNELS = 3

TRAIN_PATH = '../input/train_images/train_images/'

TEST_PATH = '../input/test_images/test_images/'



train_masks = pd.read_csv('../input/train_masks.csv')

print(train_masks.head())



img_names = train_masks['ImageId'].unique()

print('img_names[:10] = ', img_names[:10])



IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH)

OUTPUT_DIR = 'Y_train'

print('OUTPUT_DIR = ', OUTPUT_DIR)



def rle_decode(mask_rle, shape=(128, 128)):

    print('rle_decode(mask_rle = ', mask_rle)

    '''

    mask_rle: run-length as string formated (start length)

    shape: (height,width) of array to return

    Returns numpy array, 1 - mask, 0 - background



    '''

    s = mask_rle.split()

    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    starts -= 1

    ends = starts + lengths

    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):

        img[lo:hi] = 1

    return img.reshape(shape).T  # Needed to align to RLE direction



count = 0

for ImageId in img_names:

    print('ImageId', ImageId)

    

    count+= 1

    if count > 10:  # TODO: remove this to run all

        break

    else:

        print('TODO: remove this to run all cases !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    

    fname = ImageId.replace('.jpg', '.png')

    print('fname', fname)

    

    out_path = os.path.join(OUTPUT_DIR, fname)

    print('out_path = ', out_path)

    

    Y_train = np.zeros(IMG_SHAPE)

    # NOTE: multiple masks for the same image

    img_masks = train_masks.loc[train_masks['ImageId'] == ImageId, 'EncodedPixels'].tolist()

    for mask_rle in img_masks:

        print('mask_rle = ', mask_rle)

        if not pd.isnull(mask_rle):

            Y_train += rle_decode(mask_rle, shape=IMG_SHAPE)

    

    print('np.min(Y_train), np.max(Y_train) = ', np.min(Y_train), np.max(Y_train))

    

    # from keras.preprocessing.image import save_img 

    # Is there a way to save to kaggle?

    



warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

seed = 42

random.seed = seed

np.random.seed = seed



# Get train and test IDs

train_ids = next(os.walk(TRAIN_PATH))[1]

test_ids = next(os.walk(TEST_PATH))[1]



# Get and resize train images

X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

print('Getting and resizing train images and masks ... ')

sys.stdout.flush()

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):

    img = imread(TRAIN_PATH + id_ + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]

    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

    X_train[n] = img

    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

    for mask_file in next(os.walk(TRAIN_PATH + id_ + '/images/'))[2]:

        mask_ = imread(TRAIN_PATH + id_ + '/images/' + mask_file)

        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', 

                                      preserve_range=True), axis=-1)

    



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



# Check if training data looks all right

ix = random.randint(0, len(train_ids))

imshow(X_train[ix])

plt.show()

imshow(np.squeeze(Y_train[ix]))

plt.show()



# Define IoU metric

def mean_iou(y_true, y_pred):

    prec = []

    for t in np.arange(0.5, 1.0, 0.05):

        y_pred_ = tf.to_int32(y_pred > t)

        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)

        K.get_session().run(tf.local_variables_initializer())

        with tf.control_dependencies([up_opt]):

            score = tf.identity(score)

        prec.append(score)

    return K.mean(K.stack(prec), axis=0)



# Build U-Net model

inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

s = Lambda(lambda x: x / 255) (inputs)



c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)

c1 = Dropout(0.1) (c1)

c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)

p1 = MaxPooling2D((2, 2)) (c1)



c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)

c2 = Dropout(0.1) (c2)

c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)

p2 = MaxPooling2D((2, 2)) (c2)



c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)

c3 = Dropout(0.2) (c3)

c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)

p3 = MaxPooling2D((2, 2)) (c3)



c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)

c4 = Dropout(0.2) (c4)

c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)

p4 = MaxPooling2D(pool_size=(2, 2)) (c4)



c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)

c5 = Dropout(0.3) (c5)

c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)



u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)

u6 = concatenate([u6, c4])

c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)

c6 = Dropout(0.2) (c6)

c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)



u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)

u7 = concatenate([u7, c3])

c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)

c7 = Dropout(0.2) (c7)

c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)



u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)

u8 = concatenate([u8, c2])

c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)

c8 = Dropout(0.1) (c8)

c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)



u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)

u9 = concatenate([u9, c1], axis=3)

c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)

c9 = Dropout(0.1) (c9)

c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)



outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)



model = Model(inputs=[inputs], outputs=[outputs])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])

model.summary()



# Fit model

earlystopper = EarlyStopping(patience=5, verbose=1)

checkpointer = ModelCheckpoint('model-dsbowl2018-1.h5', verbose=1, save_best_only=True)

results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=50, 

                    callbacks=[earlystopper, checkpointer])



# Predict on train, val and test

model = load_model('model-dsbowl2018-1.h5', custom_objects={'mean_iou': mean_iou})

preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)

preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)

preds_test = model.predict(X_test, verbose=1)



# Threshold predictions

preds_train_t = (preds_train > 0.5).astype(np.uint8)

preds_val_t = (preds_val > 0.5).astype(np.uint8)

preds_test_t = (preds_test > 0.5).astype(np.uint8)



# Create list of upsampled test masks

preds_test_upsampled = []

for i in range(len(preds_test)):

    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]), 

                                       (sizes_test[i][0], sizes_test[i][1]), 

                                       mode='constant', preserve_range=True))

    

# Perform a sanity check on some random training samples

ix = random.randint(0, len(preds_train_t))

imshow(X_train[ix])

plt.show()

imshow(np.squeeze(Y_train[ix]))

plt.show()

imshow(np.squeeze(preds_train_t[ix]))

plt.show()



# Perform a sanity check on some random validation samples

ix = random.randint(0, len(preds_val_t))

imshow(X_train[int(X_train.shape[0]*0.9):][ix])

plt.show()

imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))

plt.show()

imshow(np.squeeze(preds_val_t[ix]))

plt.show()



# Run-length encoding 

def rle_encoding(x):

    dots = np.where(x.T.flatten() == 1)[0]

    run_lengths = []

    prev = -2

    for b in dots:

        if (b>prev+1): run_lengths.extend((b + 1, 0))

        run_lengths[-1] += 1

        prev = b

    return run_lengths



def prob_to_rles(x, cutoff=0.5):

    lab_img = label(x > cutoff)

    for i in range(1, lab_img.max() + 1):

        yield rle_encoding(lab_img == i)

        

new_test_ids = []

rles = []

for n, id_ in enumerate(test_ids):

    rle = list(prob_to_rles(preds_test_upsampled[n]))

    rles.extend(rle)

    new_test_ids.extend([id_] * len(rle))

    

# Create submission DataFrame

sub = pd.DataFrame()

sub['ImageId'] = new_test_ids

sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))

sub.to_csv('sub-dsbowl2018-1.csv', index=False)