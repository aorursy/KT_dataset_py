# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from skimage.io import imsave

from skimage.transform import resize

import csv

from glob import glob

from keras.preprocessing.image import ImageDataGenerator

import os



# import multiprocessing as mp

# mp.set_start_method('forkserver')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input/segmentation/'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



dt_y = {'se': None, 'he': None}



dt_x = glob("/".join(["", "kaggle", "input", "segmentation", 

                     "original", "*jpg"]))

dt_y['se'] = glob("/".join(["", "kaggle", "input", "segmentation",

                     "gt", "*", "*SE.tif"]))

dt_y['he'] = glob("/".join(["", "kaggle", "input", "segmentation",

                     "gt", "*", "*HE.tif"]))

dt_y['od'] = glob("/".join(["", "kaggle", "input", "segmentation",

                     "gt", "*", "*OD.tif"]))

# Any results you write to the current directory are saved as output.
from skimage.io import imread

import matplotlib.pyplot as plt



f, axes = plt.subplots(1, 2, sharey=True)

idx = 24

im = [imread(dt_x[idx]), imread(dt_y['od'][idx])]

for i, a in enumerate(axes):

    a.imshow(im[i])

    

print(im[1].ndim)
from keras.models import Model

from keras.layers import Input, Conv2D, MaxPooling2D

from keras.layers.convolutional import UpSampling2D

from keras.layers.merge import concatenate

from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, ProgbarLogger, CSVLogger # ReduceLROnPlateau

from keras import backend as K

from keras import initializers

import math



# if TENSORFLOW -> use channels_last

# if THEANO -> use channels_first

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

AXIS = -1 # -1 for 'channels_last' and 0 for 'channels_first'



CLASSES_NO = 2 # including bckgnd

IMAGE_ROWS = 512 #1024 #1020

IMAGE_COLS = 512 #1024 #1020

RESULT_ROWS = 512 #1024 #836

RESULT_COLS = 512 #1024 #836

EPOCHS_NO = 10

FEAT_MAP_NO = np.array([8, 16, 32, 64, 128])

W_SEED = list(range(40)) # None



BATCH_SIZE = 4

TRAIN_SAMPLES = (CLASSES_NO-1)*75

VAL_SAMPLES = (CLASSES_NO-1)*6

TRAIN_STEPS = math.ceil(TRAIN_SAMPLES / BATCH_SIZE)



BCKGND_W = 1./(CLASSES_NO-1)
from keras import backend as K

from math import log



BCKGND_WEIGHT = 1./CLASSES_NO

SMOOTH = 1.

BINARY_SMOOTH = 0





##=============================================================================

##WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW

##=============================================================================



def dc(y_true, y_pred): # dice coefficient

    weights = K.flatten(y_true[...,CLASSES_NO:2*CLASSES_NO])

    y_true_f = K.flatten(y_true[...,0:CLASSES_NO])

    y_pred_f = K.flatten(y_pred) * weights

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + SMOOTH) / (K.sum(y_true_f * weights) + K.sum(y_pred_f) + SMOOTH)



def dc_b_binary(y_true, y_pred): # dice coefficient background

    weights = K.flatten(y_true[...,2*CLASSES_NO])  

    y_true_f = K.flatten(y_true[...,0])

    y_pred_f = K.flatten(y_pred[...,0]) * weights

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + BINARY_SMOOTH) / (K.sum(y_true_f * weights) + K.sum(y_pred_f) + BINARY_SMOOTH)



def dc_b(y_true, y_pred): # dice coefficient background

    weights = K.flatten(y_true[...,CLASSES_NO])

    y_true_f = K.flatten(y_true[...,0])

    y_pred_f = K.flatten(y_pred[...,0]) * weights

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + SMOOTH) / (K.sum(y_true_f * weights) + K.sum(y_pred_f) + SMOOTH)



def dc_o_binary(y_true, y_pred): # dice coefficient object

    weights = K.flatten(y_true[...,2*CLASSES_NO+1:])

    y_true_f = K.flatten(y_true[...,1:CLASSES_NO])

    y_pred_f = K.flatten(y_pred[...,1:]) * weights

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + BINARY_SMOOTH) / (K.sum(y_true_f * weights) + K.sum(y_pred_f) + BINARY_SMOOTH)



def dc_o(y_true, y_pred): # dice coefficient object

    weights = K.flatten(y_true[...,CLASSES_NO+1:2*CLASSES_NO])

    y_true_f = K.flatten(y_true[...,1:CLASSES_NO])

    y_pred_f = K.flatten(y_pred[...,1:]) * weights

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + SMOOTH) / (K.sum(y_true_f * weights) + K.sum(y_pred_f) + SMOOTH)



def dc_loss(y_true, y_pred):

    return -dc(y_true, y_pred)



def dc_log_loss_w(y_true, y_pred): # dice coefficient weighted loss

    return -(K.log(dc_b(y_true, y_pred)) + K.log(dc_o(y_true, y_pred)))



def dc_loss_w(y_true, y_pred): # dice coefficient weighted loss

    return -(dc_b(y_true, y_pred) * BCKGND_WEIGHT + dc_o(y_true, y_pred) * (1-BCKGND_WEIGHT))



def dc_loss_w_binary(y_true, y_pred): # dice coefficient weighted loss

    return -(dc_b_binary(y_true, y_pred) * BCKGND_WEIGHT + dc_o_binary(y_true, y_pred) * (1-BCKGND_WEIGHT))



##=============================================================================

##WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW

##=============================================================================



def jaccard(y_true, y_pred):    # intersection over union

    weights = K.flatten(y_true[...,2*CLASSES_NO:])

    y_true_f = K.flatten(y_true[...,0:CLASSES_NO])

    y_pred_f = K.flatten(y_pred) * weights

    intersection = K.sum(y_true_f * y_pred_f)

    return (intersection + SMOOTH) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + SMOOTH)



def j_b_binary(y_true, y_pred):    # intersection over union background

    weights = K.flatten(y_true[...,2*CLASSES_NO])  

    y_true_f = K.flatten(y_true[...,0])

    y_pred_f = K.flatten(y_pred[...,0]) * weights

    intersection = K.sum(y_true_f * y_pred_f)

    return (intersection + BINARY_SMOOTH) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + BINARY_SMOOTH)



def j_b(y_true, y_pred):    # intersection over union background

    weights = K.flatten(y_true[...,CLASSES_NO]) 

    y_true_f = K.flatten(y_true[...,0])

    y_pred_f = K.flatten(y_pred[...,0]) * weights

    intersection = K.sum(y_true_f * y_pred_f)

    return (intersection + SMOOTH) / (K.sum(y_true_f * weights) + K.sum(y_pred_f) - intersection + SMOOTH)



def j_o_binary(y_true, y_pred):    # intersection over union object

    weights = K.flatten(y_true[...,2*CLASSES_NO+1:])

    y_true_f = K.flatten(y_true[...,1:CLASSES_NO])

    y_pred_f = K.flatten(y_pred[...,1:CLASSES_NO]) * weights

    intersection = K.sum(y_true_f * y_pred_f)

    return (intersection + BINARY_SMOOTH) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + BINARY_SMOOTH)



def j_o(y_true, y_pred):    # intersection over union object

    weights = K.flatten(y_true[...,CLASSES_NO+1:2*CLASSES_NO])

    y_true_f = K.flatten(y_true[...,1:CLASSES_NO])

    y_pred_f = K.flatten(y_pred[...,1:CLASSES_NO]) * weights

    intersection = K.sum(y_true_f * y_pred_f)

    return (intersection + SMOOTH) / (K.sum(y_true_f * weights) + K.sum(y_pred_f) - intersection + SMOOTH)



def jaccard_loss(y_true, y_pred):

    return -jaccard(y_true, y_pred)



def jaccard_loss_w(y_true, y_pred):

    return -(j_b(y_true, y_pred) * BCKGND_WEIGHT + j_o(y_true, y_pred) * (1-BCKGND_WEIGHT))



def jaccard_loss_w_binary(y_true, y_pred):

    return -(j_b_binary(y_true, y_pred) * BCKGND_WEIGHT + j_o_binary(y_true, y_pred) * (1-BCKGND_WEIGHT))



##=============================================================================

##WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW

##=============================================================================



def f1_score(y_true, y_pred):    

    y_true_f = K.flatten(y_true[...,0:CLASSES_NO])

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return 2.0 / (K.sum(y_true_f) / intersection + K.sum(y_pred_f) / intersection)



def f1_b(y_true, y_pred):    

    y_true_f = K.flatten(y_true[...,0])

    y_pred_f = K.flatten(y_pred[...,0])

    intersection = K.sum(y_true_f * y_pred_f)

    return 2.0 / (K.sum(y_true_f) / intersection + K.sum(y_pred_f) / intersection)



def f1_o(y_true, y_pred):    

    y_true_f = K.flatten(y_true[...,1:CLASSES_NO])

    y_pred_f = K.flatten(y_pred[...,1:])

    intersection = K.sum(y_true_f * y_pred_f)

    return 2.0 / (K.sum(y_true_f) / intersection + K.sum(y_pred_f) / intersection)



def f1_score_loss(y_true, y_pred):

    return -f1_score_loss(y_true, y_pred)



def f1_score_loss_w(y_true, y_pred):

    return -(f1_b(y_true, y_pred) * BCKGND_WEIGHT + f1_o(y_true, y_pred) * (1-BCKGND_WEIGHT))



##=============================================================================

##WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW

##=============================================================================



def accuracy(y_true, y_pred):    

    y_true_f = K.flatten(y_true[...,0:CLASSES_NO])

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    intersection_complements =  K.sum((1-y_true_f) * (1-y_pred_f))

    return (intersection + intersection_complements) / len(y_true_f)



def accuracy_bckgnd(y_true, y_pred):    

    y_true_f = K.flatten(y_true[...,0])

    y_pred_f = K.flatten(y_pred[...,0])

    intersection = K.sum(y_true_f * y_pred_f)

    intersection_complements =  K.sum((1-y_true_f) * (1-y_pred_f))

    return (intersection + intersection_complements) / len(y_true_f)



def accuracy_obj(y_true, y_pred):    

    y_true_f = K.flatten(y_true[...,1:CLASSES_NO])

    y_pred_f = K.flatten(y_pred[...,1:])

    intersection = K.sum(y_true_f * y_pred_f)

    intersection_complements =  K.sum((1-y_true_f) * (1-y_pred_f))

    return (intersection + intersection_complements) / len(y_true_f)



def accuracy_loss(y_true, y_pred):

    return -accuracy(y_true, y_pred)



def accuracy_loss_w(y_true, y_pred):

    return -(accuracy_bckgnd(y_true, y_pred) * BCKGND_WEIGHT + accuracy_obj(y_true, y_pred) * (1-BCKGND_WEIGHT))

from skimage.color import rgb2gray



# Preprocess seadanya hehe



dt_x_img = map(lambda x: imread(x), dt_x)

dt_y_img = {'od': None}

dt_y_img['od'] = map(lambda x: imread(x), dt_y['od'])



dt_x_img_r = map(lambda x: resize(x, (IMAGE_ROWS, IMAGE_COLS)), dt_x_img)

dt_y_img_r = {'od': None}

dt_y_img_r_od = map(lambda x: resize(x, (IMAGE_ROWS, IMAGE_COLS)), dt_y_img['od'])



dt_x_img_rg = map(lambda x: rgb2gray(x), dt_x_img_r)

dt_y_img_rg = {'od': None}

dt_y_img_rg_od = map(lambda x: rgb2gray(x), dt_y_img_r_od)
np.array(imread(dt_x[0])).shape
def normalizeImage(img):

    img = np.array([img])

    img = img_as_float(img)

    img = img.astype('float32')

    mean = np.mean(img)  # mean for data centering

    std = np.std(img)  # std for data normalization

    img -= mean

    img /= std

    return img



# -----------------------------------------------------------------------------

def normalizeMask(mask):

    mask = np.array([mask])

    mask = img_as_float(mask)

    mask = mask.astype('float32')

    return mask



# -----------------------------------------------------------------------------

def calculateWeights(obj_mask, bckgnd_msk):

    sum_all = np.sum(obj_mask + bckgnd_msk, dtype=np.float32) + 1  

    sum_obj = np.sum(obj_mask, dtype=np.float32)

    sum_bck = np.sum(bckgnd_msk, dtype=np.float32)

    # make sure there is at least some contribution and not 0s

    if sum_obj < 100:   

        sum_obj = 100

    if sum_bck < 100:

        sum_bck = 100

    return np.float32(obj_mask)*np.float32(sum_bck)/np.float32(sum_all) + np.float32(bckgnd_msk)*np.float32(sum_obj)/np.float32(sum_all)
from skimage.util import invert, img_as_float

from sklearn.utils import shuffle



Xtrain = []

YWtrain = []



def training_data_generator():

    X = np.zeros([BATCH_SIZE, IMAGE_ROWS, IMAGE_COLS, 1], dtype=np.float32)

    YW = np.zeros([BATCH_SIZE, RESULT_ROWS, RESULT_COLS, 3*CLASSES_NO], dtype=np.float32)   

    i_batch = 0

    

    print('Initializing training data generator\n')

    

    fx = lambda x: imread(x, as_gray=True)

    fbg = lambda x: invert(imread(x, as_gray=True))

    img = map(fx, dt_x[:50])

    img_mask = map(fx, dt_y['od'][:50])

    img_mask_bg = map(fbg, dt_y['od'][:50])

    

    fx = lambda x: resize(x, (512, 512))

    img = map(fx, img)

    img_mask = map(fx, img_mask)

    img_mask_bg = map(fx, img_mask_bg)

    

    fx = lambda x: normalizeImage(x)

    fy = lambda y: normalizeMask(y)

    fw = lambda w: calculateWeights(w)

    img = list(map(fx, img))

    img_mask = list(map(fy, img_mask))

    img_mask_bg = list(map(fy, img_mask_bg))

    img_weights = calculateWeights(img_mask, img_mask_bg)

    img_weights_binary = img_mask + img_mask_bg

    

#     datagen = ImageDataGenerator(featurewise_center=True,

# #     featurewise_std_normalization=True,

#     rotation_range=90,

#     width_shift_range=0.2,

#     height_shift_range=0.2,

#     horizontal_flip=True,

#     vertical_flip=True)

    

#     for i in range(50):

#         Y = np.zeros((RESULT_ROWS, RESULT_COLS, CLASSES_NO), dtype=np.float32)

#         Y[..., 0] = img_mask_bg[i]

#         Y[..., 1] = img_mask[i]



#         WF = np.ones((RESULT_ROWS, RESULT_COLS, CLASSES_NO), dtype=np.float32)

#         WB = np.ones((RESULT_ROWS, RESULT_COLS, CLASSES_NO), dtype=np.float32)



#         WF[..., 0] = img_weights[i]

#         WF[..., 1] = img_weights[i]



#         WB[..., 0] = img_weights_binary[i]

#         WB[..., 1] = img_weights_binary[i]



#         X[i_batch, ..., 0] = img[i]

#         YW[i_batch, ...] = np.concatenate((Y, WB, WB), -1)

#         i_batch = i_batch + 1

#         if i_batch >= BATCH_SIZE:

#             i_batch = 0 

    

#     global Xtrain

#     global YWtrain

#     Xtrain = X

#     YWtrain = YW

    

#     datagen.fit(X)

#     print('DONE')

#     return datagen, Xtrain, YWtrain

    

    while True:

        for i in range(50):

            Y = np.zeros((RESULT_ROWS, RESULT_COLS, CLASSES_NO), dtype=np.float32)

#             Y[..., 0] = img_mask_bg[i]

            Y[..., 0] = img_mask[i]



            WF = np.ones((RESULT_ROWS, RESULT_COLS, CLASSES_NO), dtype=np.float32)

            WB = np.ones((RESULT_ROWS, RESULT_COLS, CLASSES_NO), dtype=np.float32)



            WF[..., 0] = img_weights[i]

#             WF[..., 1] = img_weights[i]



            WB[..., 0] = img_weights_binary[i]

#             WB[..., 1] = img_weights_binary[i]



            X[i_batch, ..., 0] = img[i]

            YW[i_batch, ...] = np.concatenate((Y, WB, WB), -1)

            i_batch = i_batch + 1

#             print("Data: ", i)



            if i_batch >= BATCH_SIZE:

                i_batch = 0 

                yield (X,YW)

                
Xval = []

YWval = []



def validation_data_generator():

    X = np.zeros([BATCH_SIZE, IMAGE_ROWS, IMAGE_COLS, 1], dtype=np.float32)

    YW = np.zeros([BATCH_SIZE, RESULT_ROWS, RESULT_COLS, 3*CLASSES_NO], dtype=np.float32)   

    i_batch = 0

    

    print('Initializing validation data generator\n')

    

    fx = lambda x: imread(x, as_gray=True)

    fbg = lambda x: invert(imread(x, as_gray=True))

    img = map(fx, dt_x[50:])

    img_mask = map(fx, dt_y['od'][50:])

    img_mask_bg = map(fbg, dt_y['od'][50:])

    

    fx = lambda x: resize(x, (512, 512))

    img = map(fx, img)

    img_mask = map(fx, img_mask)

    img_mask_bg = map(fx, img_mask_bg)

    

    fx = lambda x: normalizeImage(x)

    fy = lambda y: normalizeMask(y)

    fw = lambda w: calculateWeights(w)

    img = list(map(fx, img))

    img_mask = list(map(fy, img_mask))

    img_mask_bg = list(map(fy, img_mask_bg))

    img_weights = list(map(lambda x, y: calculateWeights(x, y), img_mask, img_mask_bg))

    img_weights_binary = img_mask + img_mask_bg

        

#     datagen = ImageDataGenerator(featurewise_center=True,

# #     featurewise_std_normalization=True,

#     rotation_range=90,

#     width_shift_range=0.2,

#     height_shift_range=0.2,

#     horizontal_flip=True,

#     vertical_flip=True)

    

#     for i in range(0, 81-50):

#         Y = np.zeros((RESULT_ROWS, RESULT_COLS, CLASSES_NO), dtype=np.float32)

#         Y[..., 0] = img_mask_bg[i]

#         Y[..., 1] = img_mask[i]



#         WF = np.ones((RESULT_ROWS, RESULT_COLS, CLASSES_NO), dtype=np.float32)

#         WB = np.ones((RESULT_ROWS, RESULT_COLS, CLASSES_NO), dtype=np.float32)



#         WF[..., 0] = img_weights[i]

#         WF[..., 1] = img_weights[i]



#         WB[..., 0] = img_weights_binary[i]

#         WB[..., 1] = img_weights_binary[i]



#         X[i_batch, ..., 0] = img[i]

#         YW[i_batch, ...] = np.concatenate((Y, WB, WB), -1)

    

#     global Xval

#     global YWval

#     Xval = X

#     YWval = YW

    

#     datagen.fit(X)

#     print('DONE')

#     return datagen, Xval, YWval

    

    while True:

        for i in range(0, 81-50):

            Y = np.zeros((RESULT_ROWS, RESULT_COLS, CLASSES_NO), dtype=np.float32)

#             Y[..., 0] = img_mask_bg[i]

            Y[..., 0] = img_mask[i]



            WF = np.ones((RESULT_ROWS, RESULT_COLS, CLASSES_NO), dtype=np.float32)

            WB = np.ones((RESULT_ROWS, RESULT_COLS, CLASSES_NO), dtype=np.float32)



            WF[..., 0] = img_weights[i]

#             WF[..., 1] = img_weights[i]



            WB[..., 0] = img_weights_binary[i]

#             WB[..., 1] = img_weights_binary[i]



            X[i_batch, ..., 0] = img[i]

            YW[i_batch, ...] = np.concatenate((Y, WF, WB), -1)



#             print("Data: ", i)



            yield(X, YW)
def test_data_generator():

    X = np.zeros([BATCH_SIZE, IMAGE_ROWS, IMAGE_COLS, 1], dtype=np.float32)

    YW = np.zeros([BATCH_SIZE, RESULT_ROWS, RESULT_COLS, 3*CLASSES_NO], dtype=np.float32)   

    i_batch = 0

    

    print('Initializing testing data generator\n')

    

    fx = lambda x: imread(x, as_gray=True)

    fbg = lambda x: invert(imread(x, as_gray=True))

    img = map(fx, dt_x[50:])

    img_mask = map(fx, dt_y['od'][50:])

    img_mask_bg = map(fbg, dt_y['od'][50:])

    

    fx = lambda x: resize(x, (512, 512))

    img = map(fx, img)

    img_mask = map(fx, img_mask)

    img_mask_bg = map(fx, img_mask_bg)

    

    fx = lambda x: normalizeImage(x)

    fy = lambda y: normalizeMask(y)

    fw = lambda w: calculateWeights(w)

    img = list(map(fx, img))

    img_mask = list(map(fy, img_mask))

    img_mask_bg = list(map(fy, img_mask_bg))

    img_weights = list(map(lambda x, y: calculateWeights(x, y), img_mask, img_mask_bg))

    img_weights_binary = img_mask + img_mask_bg

    

    while True:

        for i in range(0, 81-50):

            Y = np.zeros((RESULT_ROWS, RESULT_COLS, CLASSES_NO), dtype=np.float32)

#             Y[..., 0] = img_mask_bg[i]

            Y[..., 0] = img_mask[i]



            WF = np.ones((RESULT_ROWS, RESULT_COLS, CLASSES_NO), dtype=np.float32)

            WB = np.ones((RESULT_ROWS, RESULT_COLS, CLASSES_NO), dtype=np.float32)



            WF[..., 0] = img_weights[i]

#             WF[..., 1] = img_weights[i]



            WB[..., 0] = img_weights_binary[i]

#             WB[..., 1] = img_weights_binary[i]



            X[i_batch, ..., 0] = img[i]

            YW[i_batch, ...] = np.concatenate((Y, WF, WB), -1)

            

            yield(X, YW)
datagen = training_data_generator()
valgen = validation_data_generator()
# qq = valgen[0].flow(valgen[1], valgen[2])



# next(qq)
def unet_6(feature_maps, last_layer):

    inputs = Input((IMAGE_ROWS, IMAGE_COLS, 1), name='input')

    conv1 = Conv2D(feature_maps[0], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[0]), name='conv1_1')(inputs)

    conv1 = Conv2D(feature_maps[0], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[1]), name='conv1_2')(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)



    conv2 = Conv2D(feature_maps[1], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[2]), name='conv2_1')(pool1)

    conv2 = Conv2D(feature_maps[1], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[3]), name='conv2_2')(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)



    conv3 = Conv2D(feature_maps[2], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[4]), name='conv3_1')(pool2)

    conv3 = Conv2D(feature_maps[2], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[5]), name='conv3_2')(conv3)

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)



    conv4 = Conv2D(feature_maps[3], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[6]), name='conv4_1')(pool3)

    conv4 = Conv2D(feature_maps[3], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[7]), name='conv4_2')(conv4)

    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    

    conv5 = Conv2D(feature_maps[4], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[8]), name='conv5_1')(pool4)

    conv5 = Conv2D(feature_maps[4], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[9]), name='conv5_2')(conv5)

    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)



    conv6 = Conv2D(feature_maps[5], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[10]), name='conv6_1')(pool5)

    conv6 = Conv2D(feature_maps[5], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[11]), name='conv6_2')(conv6)

    pool6 = MaxPooling2D(pool_size=(2, 2))(conv6)





    conv7 = Conv2D(feature_maps[6], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[12]), name='convDeep_1')(pool6)

    conv7 = Conv2D(feature_maps[6], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[13]), name='convDeep_2')(conv7)





    up_6 = UpSampling2D(size=(2, 2), name='upconv6_0')(conv7)

    up_6 = Conv2D(feature_maps[5], (2, 2), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[14]), name='upconv_6_1')(up_6)

    conv_6 = Conv2D(feature_maps[5], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[15]), name='conv_6_1')(concatenate([conv6, up_6], axis=AXIS))

    conv_6 = Conv2D(feature_maps[5], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[16]), name='conv_6_2')(conv_6)



    up_5 = UpSampling2D(size=(2, 2), name='upconv5_0')(conv_6)

    up_5 = Conv2D(feature_maps[4], (2, 2), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[17]), name='upconv_5_1')(up_5)

    conv_5 = Conv2D(feature_maps[4], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[18]), name='conv_5_1')(concatenate([conv5, up_5], axis=AXIS))

    conv_5 = Conv2D(feature_maps[4], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[19]), name='conv_5_2')(conv_5)



    up_4 = UpSampling2D(size=(2, 2), name='upconv4_0')(conv_5)

    up_4 = Conv2D(feature_maps[3], (2, 2), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[20]), name='upconv_4_1')(up_4)

    conv_4 = Conv2D(feature_maps[3], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[21]), name='conv_4_1')(concatenate([conv4, up_4], axis=AXIS))

    conv_4 = Conv2D(feature_maps[3], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[22]), name='conv_4_2')(conv_4)



    up_3 = UpSampling2D(size=(2, 2), name='upconv3_0')(conv_4)

    up_3 = Conv2D(feature_maps[2], (2, 2), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[23]), name='upconv_3_1')(up_3)

    conv_3 = Conv2D(feature_maps[2], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[24]), name='conv_3_1')(concatenate([conv3, up_3], axis=AXIS))

    conv_3 = Conv2D(feature_maps[2], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[25]), name='conv_3_2')(conv_3)



    up_2 = UpSampling2D(size=(2, 2), name='upconv2_0')(conv_3)

    up_2 = Conv2D(feature_maps[1], (2, 2), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[26]), name='upconv_2_1')(up_2)

    conv_2 = Conv2D(feature_maps[1], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[27]), name='conv_2_1')(concatenate([conv2, up_2], axis=AXIS))

    conv_2 = Conv2D(feature_maps[1], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[28]), name='conv_2_2')(conv_2)



    up_1 = UpSampling2D(size=(2, 2), name='upconv1_0')(conv_2)

    up_1 = Conv2D(last_layer, (2, 2), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[29]), name='upconv_1_1')(up_1)

    conv_1 = Conv2D(last_layer, (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[30]), name='conv_1_1')(concatenate([conv1, up_1], axis=AXIS))

    conv_1 = Conv2D(last_layer, (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[31]), name='conv_1_2')(conv_1)



    convOUT = Conv2D(CLASSES_NO, (1, 1), activation='softmax', kernel_initializer=initializers.he_normal(W_SEED[32]), name='convOUT')(conv_1)



    model = Model(inputs=[inputs], outputs=[convOUT])

    model.compile(optimizer=Adam(lr=1e-5), loss=dc_log_loss_w, metrics=[dc_b, dc_o, j_b, j_o])

    return model



#------------------------------------------------------------------------------

def unet_5(feature_maps, last_layer):

    inputs = Input((IMAGE_ROWS, IMAGE_COLS, 1), name='input')

    conv1 = Conv2D(feature_maps[0], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[0]), name='conv1_1')(inputs)

    conv1 = Conv2D(feature_maps[0], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[1]), name='conv1_2')(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)



    conv2 = Conv2D(feature_maps[1], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[2]), name='conv2_1')(pool1)

    conv2 = Conv2D(feature_maps[1], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[3]), name='conv2_2')(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)



    conv3 = Conv2D(feature_maps[2], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[4]), name='conv3_1')(pool2)

    conv3 = Conv2D(feature_maps[2], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[5]), name='conv3_2')(conv3)

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)



    conv4 = Conv2D(feature_maps[3], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[6]), name='conv4_1')(pool3)

    conv4 = Conv2D(feature_maps[3], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[7]), name='conv4_2')(conv4)

    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    

    conv5 = Conv2D(feature_maps[4], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[7]), name='conv5_1')(pool4)

    conv5 = Conv2D(feature_maps[4], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[8]), name='conv5_2')(conv5)

    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)





    conv6 = Conv2D(feature_maps[5], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[9]), name='convDeep_1')(pool5)

    conv6 = Conv2D(feature_maps[5], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[10]), name='convDeep_2')(conv6)





    up_5 = UpSampling2D(size=(2, 2), name='upconv5_0')(conv6)

    up_5 = Conv2D(feature_maps[4], (2, 2), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[11]), name='upconv_5_1')(up_5)

    conv_5 = Conv2D(feature_maps[4], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[12]), name='conv_5_1')(concatenate([conv5, up_5], axis=AXIS))

    conv_5 = Conv2D(feature_maps[4], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[13]), name='conv_5_2')(conv_5)



    up_4 = UpSampling2D(size=(2, 2), name='upconv4_0')(conv_5)

    up_4 = Conv2D(feature_maps[3], (2, 2), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[14]), name='upconv_4_1')(up_4)

    conv_4 = Conv2D(feature_maps[3], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[15]), name='conv_4_1')(concatenate([conv4, up_4], axis=AXIS))

    conv_4 = Conv2D(feature_maps[3], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[16]), name='conv_4_2')(conv_4)



    up_3 = UpSampling2D(size=(2, 2), name='upconv3_0')(conv_4)

    up_3 = Conv2D(feature_maps[2], (2, 2), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[17]), name='upconv_3_1')(up_3)

    conv_3 = Conv2D(feature_maps[2], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[18]), name='conv_3_1')(concatenate([conv3, up_3], axis=AXIS))

    conv_3 = Conv2D(feature_maps[2], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[19]), name='conv_3_2')(conv_3)



    up_2 = UpSampling2D(size=(2, 2), name='upconv2_0')(conv_3)

    up_2 = Conv2D(feature_maps[1], (2, 2), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[20]), name='upconv_2_1')(up_2)

    conv_2 = Conv2D(feature_maps[1], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[21]), name='conv_2_1')(concatenate([conv2, up_2], axis=AXIS))

    conv_2 = Conv2D(feature_maps[1], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[22]), name='conv_2_2')(conv_2)



    up_1 = UpSampling2D(size=(2, 2), name='upconv1_0')(conv_2)

    up_1 = Conv2D(last_layer, (2, 2), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[23]), name='upconv_1_1')(up_1)

    conv_1 = Conv2D(last_layer, (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[24]), name='conv_1_1')(concatenate([conv1, up_1], axis=AXIS))

    conv_1 = Conv2D(last_layer, (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[25]), name='conv_1_2')(conv_1)



    convOUT = Conv2D(CLASSES_NO, (1, 1), activation='softmax', kernel_initializer=initializers.he_normal(W_SEED[26]), name='convOUT')(conv_1)



    model = Model(inputs=[inputs], outputs=[convOUT])

    model.compile(optimizer=Adam(lr=1e-5), loss=dc_log_loss_w, metrics=[dc_b, dc_o, j_b, j_o])

    return model



#------------------------------------------------------------------------------

def unet_4(feature_maps, last_layer):

    inputs = Input((IMAGE_ROWS, IMAGE_COLS, 1), name='input')

    conv1 = Conv2D(feature_maps[0], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[0]), name='conv1_1')(inputs)

    conv1 = Conv2D(feature_maps[0], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[1]), name='conv1_2')(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)



    conv2 = Conv2D(feature_maps[1], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[2]), name='conv2_1')(pool1)

    conv2 = Conv2D(feature_maps[1], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[3]), name='conv2_2')(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)



    conv3 = Conv2D(feature_maps[2], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[4]), name='conv3_1')(pool2)

    conv3 = Conv2D(feature_maps[2], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[5]), name='conv3_2')(conv3)

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)



    conv4 = Conv2D(feature_maps[3], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[6]), name='conv4_1')(pool3)

    conv4 = Conv2D(feature_maps[3], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[7]), name='conv4_2')(conv4)

    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    



    conv5 = Conv2D(feature_maps[4], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[8]), name='convDeep_1')(pool4)

    conv5 = Conv2D(feature_maps[4], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[9]), name='convDeep_2')(conv5)





    up_4 = UpSampling2D(size=(2, 2), name='upconv4_0')(conv5)

    up_4 = Conv2D(feature_maps[3], (2, 2), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[16]), name='upconv_4_1')(up_4)

    conv_4 = Conv2D(feature_maps[3], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[17]), name='conv_4_1')(concatenate([conv4, up_4], axis=AXIS))

    conv_4 = Conv2D(feature_maps[3], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[18]), name='conv_4_2')(conv_4)



    up_3 = UpSampling2D(size=(2, 2), name='upconv3_0')(conv_4)

    up_3 = Conv2D(feature_maps[2], (2, 2), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[19]), name='upconv_3_1')(up_3)

    conv_3 = Conv2D(feature_maps[2], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[20]), name='conv_3_1')(concatenate([conv3, up_3], axis=AXIS))

    conv_3 = Conv2D(feature_maps[2], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[21]), name='conv_3_2')(conv_3)



    up_2 = UpSampling2D(size=(2, 2), name='upconv2_0')(conv_3)

    up_2 = Conv2D(feature_maps[1], (2, 2), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[22]), name='upconv_2_1')(up_2)

    conv_2 = Conv2D(feature_maps[1], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[23]), name='conv_2_1')(concatenate([conv2, up_2], axis=AXIS))

    conv_2 = Conv2D(feature_maps[1], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[24]), name='conv_2_2')(conv_2)



    up_1 = UpSampling2D(size=(2, 2), name='upconv1_0')(conv_2)

    up_1 = Conv2D(last_layer, (2, 2), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[25]), name='upconv_1_1')(up_1)

    conv_1 = Conv2D(last_layer, (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[26]), name='conv_1_1')(concatenate([conv1, up_1], axis=AXIS))

    conv_1 = Conv2D(last_layer, (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[27]), name='conv_1_2')(conv_1)



    convOUT = Conv2D(CLASSES_NO, (1, 1), activation='softmax', kernel_initializer=initializers.he_normal(W_SEED[28]), name='convOUT')(conv_1)



    model = Model(inputs=[inputs], outputs=[convOUT])

    model.compile(optimizer=Adam(lr=1e-5), loss=dc_log_loss_w, metrics=[dc_b, dc_o, j_b, j_o])    

    return model



#------------------------------------------------------------------------------

def unet_3(feature_maps, last_layer):

    inputs = Input((IMAGE_ROWS, IMAGE_COLS, 1), name='input')

    conv1 = Conv2D(feature_maps[0], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[0]), name='conv1_1')(inputs)

    conv1 = Conv2D(feature_maps[0], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[1]), name='conv1_2')(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)



    conv2 = Conv2D(feature_maps[1], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[2]), name='conv2_1')(pool1)

    conv2 = Conv2D(feature_maps[1], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[3]), name='conv2_2')(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)



    conv3 = Conv2D(feature_maps[2], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[4]), name='conv3_1')(pool2)

    conv3 = Conv2D(feature_maps[2], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[5]), name='conv3_2')(conv3)

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

  



    conv4 = Conv2D(feature_maps[3], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[8]), name='convDeep_1')(pool3)

    conv4 = Conv2D(feature_maps[3], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[9]), name='convDeep_2')(conv4)





    up_3 = UpSampling2D(size=(2, 2), name='upconv3_0')(conv4)

    up_3 = Conv2D(feature_maps[2], (2, 2), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[19]), name='upconv_3_1')(up_3)

    conv_3 = Conv2D(feature_maps[2], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[20]), name='conv_3_1')(concatenate([conv3, up_3], axis=AXIS))

    conv_3 = Conv2D(feature_maps[2], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[21]), name='conv_3_2')(conv_3)



    up_2 = UpSampling2D(size=(2, 2), name='upconv2_0')(conv_3)

    up_2 = Conv2D(feature_maps[1], (2, 2), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[22]), name='upconv_2_1')(up_2)

    conv_2 = Conv2D(feature_maps[1], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[23]), name='conv_2_1')(concatenate([conv2, up_2], axis=AXIS))

    conv_2 = Conv2D(feature_maps[1], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[24]), name='conv_2_2')(conv_2)



    up_1 = UpSampling2D(size=(2, 2), name='upconv1_0')(conv_2)

    up_1 = Conv2D(last_layer, (2, 2), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[27]), name='upconv_1_1')(up_1)

    conv_1 = Conv2D(last_layer, (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[25]), name='conv_1_1')(concatenate([conv1, up_1], axis=AXIS))

    conv_1 = Conv2D(last_layer, (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[26]), name='conv_1_2')(conv_1)



    convOUT = Conv2D(CLASSES_NO, (1, 1), activation='softmax', kernel_initializer=initializers.he_normal(W_SEED[27]), name='convOUT')(conv_1)



    model = Model(inputs=[inputs], outputs=[convOUT])

    model.compile(optimizer=Adam(lr=1e-5), loss=dc_log_loss_w, metrics=[dc_b, dc_o, j_b, j_o])

    return model



#------------------------------------------------------------------------------

def unet_2(feature_maps, last_layer):

    inputs = Input((IMAGE_ROWS, IMAGE_COLS, 1), name='input')

    conv1 = Conv2D(feature_maps[0], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[0]), name='conv1_1')(inputs)

    conv1 = Conv2D(feature_maps[0], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[1]), name='conv1_2')(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)



    conv2 = Conv2D(feature_maps[1], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[2]), name='conv2_1')(pool1)

    conv2 = Conv2D(feature_maps[1], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[3]), name='conv2_2')(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

  



    conv3 = Conv2D(feature_maps[2], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[8]), name='convDeep_1')(pool2)

    conv3 = Conv2D(feature_maps[2], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[9]), name='convDeep_2')(conv3)





    up_2 = UpSampling2D(size=(2, 2), name='upconv2_0')(conv3)

    up_2 = Conv2D(feature_maps[1], (2, 2), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[18]), name='upconv_2_1')(up_2)

    conv_2 = Conv2D(feature_maps[1], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[17]), name='conv_2_1')(concatenate([conv2, up_2], axis=AXIS))

    conv_2 = Conv2D(feature_maps[1], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[16]), name='conv_2_2')(conv_2)



    up_1 = UpSampling2D(size=(2, 2), name='upconv1_0')(conv_2)

    up_1 = Conv2D(last_layer, (2, 2), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[19]), name='upconv_1_1')(up_1)

    conv_1 = Conv2D(last_layer, (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[20]), name='conv_1_1')(concatenate([conv1, up_1], axis=AXIS))

    conv_1 = Conv2D(last_layer, (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[21]), name='conv_1_2')(conv_1)



    convOUT = Conv2D(CLASSES_NO, (1, 1), activation='softmax', kernel_initializer=initializers.he_normal(W_SEED[27]), name='convOUT')(conv_1)



    model = Model(inputs=[inputs], outputs=[convOUT])

    model.compile(optimizer=Adam(lr=1e-5), loss=dc_log_loss_w, metrics=[dc_b, dc_o, j_b, j_o])

    return model



def unet_1(feature_maps, last_layer):

    inputs = Input((IMAGE_ROWS, IMAGE_COLS, 1), name='input')

    conv1 = Conv2D(feature_maps[0], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[0]), name='conv1_1')(inputs)

    conv1 = Conv2D(feature_maps[0], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[1]), name='conv1_2')(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

  



    conv2 = Conv2D(feature_maps[1], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[8]), name='convDeep_1')(pool1)

    conv2 = Conv2D(feature_maps[1], (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[9]), name='convDeep_2')(conv2)





    up_1 = UpSampling2D(size=(2, 2), name='upconv1_0')(conv2)

    up_1 = Conv2D(last_layer, (2, 2), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[19]), name='upconv_1_1')(up_1)

    conv_1 = Conv2D(last_layer, (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[20]), name='conv_1_1')(concatenate([conv1, up_1], axis=AXIS))

    conv_1 = Conv2D(last_layer, (3, 3), activation='selu', padding='same', kernel_initializer=initializers.he_normal(W_SEED[21]), name='conv_1_2')(conv_1)



    convOUT = Conv2D(CLASSES_NO, (1, 1), activation='softmax', kernel_initializer=initializers.he_normal(W_SEED[22]), name='convOUT')(conv_1)



    model = Model(inputs=[inputs], outputs=[convOUT])

    model.compile(optimizer=Adam(lr=1e-5), loss=dc_log_loss_w, metrics=[dc_b, dc_o, j_b, j_o])

    return model



def train_cnn(feature_maps, last_layer, depth, model_name):

    print(f"Training model {model_name}")

    

    model = None

    if depth == 1:

        model = unet_1(feature_maps, last_layer)

    elif depth == 2:

        model = unet_2(feature_maps, last_layer)

    elif depth == 3:

        model = unet_3(feature_maps, last_layer)

    elif depth == 4:

        model = unet_4(feature_maps, last_layer)

    elif depth == 5:

        model = unet_5(feature_maps, last_layer)

    elif depth == 6:

        model = unet_6(feature_maps, last_layer)

        

    model.summary()

    model_dir = 'model_' + model_name

    if not os.path.exists(model_dir):

        os.mkdir(model_dir)

        

    model_checkpoint = ModelCheckpoint(model_dir+'/weights.h5', monitor='val_loss', save_best_only=True)

    progbar_logger = ProgbarLogger(count_mode='steps')

    csv_logger = CSVLogger(model_dir+'/log.csv', append=True)

    

    print('-'*30)

    print('Fitting model ' + model_name + '...')

    print('-'*30)

    

#     model.fit_generator(generator=datagen[0].flow(datagen[1], datagen[2], batch_size=BATCH_SIZE), steps_per_epoch=50, 

#                         validation_data=valgen[0].flow(valgen[1], valgen[2], batch_size=BATCH_SIZE), validation_steps=31,

#                         epochs=EPOCHS_NO, initial_epoch=0, 

#                         max_queue_size = 50, 

#                         callbacks=[model_checkpoint, progbar_logger, csv_logger], use_multiprocessing=True, workers=0)

    model.fit_generator(generator=datagen, steps_per_epoch=50, 

                        validation_data=valgen, validation_steps=31,

                        epochs=EPOCHS_NO, initial_epoch=0, 

                        max_queue_size = 50, 

                        callbacks=[model_checkpoint, progbar_logger, csv_logger], use_multiprocessing=True, workers=0)
if __name__ == '__main__':

    base_depth_multiplier_last = [[16,5,2,40]] #[[64,4,2,64]] #[[8,5,2,24],[8,6,2,20],[16,5,2,40],[64,4,2,64]]

    repetition = ['a', 'b', 'c']



    for bdml in base_depth_multiplier_last:

        b = bdml[0]

        d = bdml[1]

        m = bdml[2]

        l = bdml[3]

        b_str = 'b' + str(b) + '_'

        d_str = 'd' + str(d) + '_'

        m_str = 'm' + str(m).replace('.','') + '_'

        l_str = 'l' + str(l) + '_'



        feature_maps = np.zeros(d+1)

        feature_maps[0] = b

        for i in range(1,d+1):

            feature_maps[i] = feature_maps[i-1] * m

        print(feature_maps)

 

        for r in repetition:

            train_cnn(feature_maps.astype(int), l, d, b_str + d_str + m_str + l_str + r)
base_depth_multiplier_last = [16,5,2,40]  #[8,5,2,24], [8,6,2,20],[16,5,2,40],[64,4,2,64]

sequence = ['a', 'b', 'c']

m_evals = []

preds = []



def test_cnn():

    for seq in sequence:

        b = base_depth_multiplier_last[0]

        d = base_depth_multiplier_last[1]

        m = base_depth_multiplier_last[2]

        l = base_depth_multiplier_last[3]

        b_str = 'b' + str(b) + '_'

        d_str = 'd' + str(d) + '_'

        m_str = 'm' + str(m).replace('.','') + '_'

        l_str = 'l' + str(l) + '_'



        MODEL_PATH = f'/kaggle/working/model_{b_str}{d_str}{m_str}{l_str}{seq}/'

        print(MODEL_PATH)



        feature_maps = np.zeros(d+1)

        feature_maps[0] = b

        for i in range(1,d+1):

            feature_maps[i] = feature_maps[i-1] * m

        fm = feature_maps.astype(int)



        print('-'*30)

        print('Loading saved model...')

        model = None

        if d == 6:

            model = unet_6(fm, l)

        elif d == 5:

            model = unet_5(fm, l)

        elif d == 4:

            model = unet_4(fm, l)

        elif d == 3:

            model = unet_3(fm, l)

        elif d == 2:

            model = unet_2(fm, l)

        elif d == 1:

            model = unet_1(fm, l)

        model.load_weights(MODEL_PATH + 'weights.h5')

        metric_name = model.metrics_names



        print('DONE')



        generator = test_data_generator()

        m_evals.append(model.evaluate_generator(generator, steps=BATCH_SIZE, workers=0, verbose=1))

#     return m_evals, metric_name

    df = pd.DataFrame(data=m_evals, columns=metric_name, index=sequence)

    return df



def predict_cnn():

    for seq in sequence:

        b = base_depth_multiplier_last[0]

        d = base_depth_multiplier_last[1]

        m = base_depth_multiplier_last[2]

        l = base_depth_multiplier_last[3]

        b_str = 'b' + str(b) + '_'

        d_str = 'd' + str(d) + '_'

        m_str = 'm' + str(m).replace('.','') + '_'

        l_str = 'l' + str(l) + '_'



        MODEL_PATH = f'/kaggle/working/model_{b_str}{d_str}{m_str}{l_str}{seq}/'

        print(MODEL_PATH)



        feature_maps = np.zeros(d+1)

        feature_maps[0] = b

        for i in range(1,d+1):

            feature_maps[i] = feature_maps[i-1] * m

        fm = feature_maps.astype(int)



        print('-'*30)

        print('Loading saved model...')

        model = None

        if d == 6:

            model = unet_6(fm, l)

        elif d == 5:

            model = unet_5(fm, l)

        elif d == 4:

            model = unet_4(fm, l)

        elif d == 3:

            model = unet_3(fm, l)

        elif d == 2:

            model = unet_2(fm, l)

        elif d == 1:

            model = unet_1(fm, l)

        model.load_weights(MODEL_PATH + 'weights.h5')

        metric_name = model.metrics_names



        print('DONE')



        generator = test_data_generator()

        preds.append(model.predict_generator(generator, steps=BATCH_SIZE, workers=0, verbose=1))

    return preds

#     df = pd.DataFrame(data=m_evals, columns=metric_name, index=sequence)

#     return df
evals = test_cnn()
evals
preds = predict_cnn()
preds
imread(dt_x[0]).shape

imread(dt_y['od'][0]).shape
plt.imshow(imread(dt_y['od'][0]))
preds[0][0, :, :, 0].shape
plt.imshow(preds[1][6, :, :, 0])
plt.imshow(preds[1][6, :, :, 1])