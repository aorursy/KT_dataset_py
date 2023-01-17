# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from skimage.io import imsave

from skimage.transform import resize

import csv

from glob import glob



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

dt_y['ex'] = glob("/".join(["", "kaggle", "input", "segmentation",

                     "gt", "*", "*EX.tif"]))

dt_y['od'] = glob("/".join(["", "kaggle", "input", "segmentation",

                     "gt", "*", "*OD.tif"]))

# Any results you write to the current directory are saved as output.
dt_y['od'].sort()

dt_x.sort()
from skimage.io import imread

import matplotlib.pyplot as plt



f, axes = plt.subplots(1, 2, sharey=True)

idx = 26

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

EPOCHS_NO = 100

FEAT_MAP_NO = np.array([8, 16, 32, 64, 128])

W_SEED = list(range(40)) # None



BATCH_SIZE = 5

TRAIN_SAMPLES = (CLASSES_NO-1)*75

VAL_SAMPLES = (CLASSES_NO-1)*6

TRAIN_STEPS = math.ceil(TRAIN_SAMPLES / BATCH_SIZE)



BCKGND_W = 1./(CLASSES_NO-1)
def dice_coef(y_true, y_pred, smooth=1):

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)



def dice_coef_loss(y_true, y_pred):

    return 1-dice_coef(y_true, y_pred)



def jaccard_distance(y_true, y_pred, smooth=100):

    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)

    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return (1 - jac) * smooth
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



img = None

img_mask = None

img_mask_bg = None

img_weights = None

img_weights_binary = None



def data_fetch(): 

    global img

    global img_mask

        

    fx = lambda x: imread(x, as_gray=True)

    img = map(fx, dt_x)

    img_mask = map(fx, dt_y['od'])

    

    fx = lambda x: np.array(x[:, 250:3750], dtype=np.float32)

    img = map(fx, img)

    img_mask = map(fx, img_mask)

    

    fx = lambda x: resize(x, (512, 512))

    img = map(fx, img)

    img_mask = map(fx, img_mask)

    

    fx = lambda x: normalizeImage(x)

    fy = lambda y: normalizeMask(y)

    img = list(map(fx, img))

    img_mask = list(map(fy, img_mask))
data_fetch()
tr_in = lambda img: np.array(img)[:, 0, :, :, np.newaxis]



img_tr = tr_in(img)

img_mask_tr = tr_in(img_mask)
img_tr.shape
tr_i = 0



def train_datagen():

    global tr_i

    while True:

        tr_i+1

        tr_i = (tr_i+1) % 81

        yield (img_tr[tr_i:tr_i+1], img_mask_tr[tr_i:tr_i+1])
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



    convOUT = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer=initializers.he_normal(W_SEED[32]), name='convOUT')(conv_1)



    model = Model(inputs=[inputs], outputs=[convOUT])

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef, jaccard_distance])

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



    convOUT = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer=initializers.he_normal(W_SEED[26]), name='convOUT')(conv_1)



    model = Model(inputs=[inputs], outputs=[convOUT])

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef, jaccard_distance])

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



    convOUT = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer=initializers.he_normal(W_SEED[28]), name='convOUT')(conv_1)



    model = Model(inputs=[inputs], outputs=[convOUT])

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef, jaccard_distance])    

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



    convOUT = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer=initializers.he_normal(W_SEED[27]), name='convOUT')(conv_1)



    model = Model(inputs=[inputs], outputs=[convOUT])

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef, jaccard_distance])

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



    convOUT = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer=initializers.he_normal(W_SEED[27]), name='convOUT')(conv_1)



    model = Model(inputs=[inputs], outputs=[convOUT])

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef, jaccard_distance])

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



    convOUT = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer=initializers.he_normal(W_SEED[22]), name='convOUT')(conv_1)



    model = Model(inputs=[inputs], outputs=[convOUT])

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef, jaccard_distance])

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

        

#     model.summary()

    model_dir = 'model_' + model_name

    if not os.path.exists(model_dir):

        os.mkdir(model_dir)

        

    model_checkpoint = ModelCheckpoint(model_dir+'/weights.h5')

    csv_logger = CSVLogger(model_dir+'/log.csv', append=True)

    

    print('-'*30)

    print('Fitting model ' + model_name + '...')

    print('-'*30)

    

    model.fit_generator(generator=train_datagen(), steps_per_epoch=50,

                        epochs=EPOCHS_NO, initial_epoch=0, 

                        max_queue_size = 50, 

                        callbacks=[model_checkpoint, csv_logger], use_multiprocessing=True, workers=0)
print('Initial data fetching')

print('-'*30)

# data_fetch()



if __name__ == '__main__':

    base_depth_multiplier_last = [[8,5,2,24], [16,5,2,40], [64,4,2,64]] # [[64,4,2,64]] #[[8,5,2,24],[8,6,2,20],[16,5,2,40],[64,4,2,64]]

    repetition = ['a']



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
def test_cnn(b, d, m, l, c: chr):

    MODEL_PATH = f'/kaggle/working/model_b{b}_d{d}_m{m}_l{l}_{c}/'

    print(MODEL_PATH)



    feature_maps = np.zeros(d+1)

    feature_maps[0] = b

    for i in range(1,d+1):

        feature_maps[i] = feature_maps[i-1] * m

    fm = feature_maps.astype(int)



    print('-'*30)

    print(f'Loading saved model (model_b{b}_d{d}_m{m}_l{l}_{c})...')

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

    

    pred = model.predict(img_tr[60:,:,:,:])

    loss, dc, j = model.evaluate(img_tr[60:,:,:,:], img_mask_tr[60:,:,:,:])

    print(f'Loss: {loss}, Dice Coef: {dc}, Jaccard Distance: {j}')

    

    for i, k in enumerate(pred):

        imsave(f'{MODEL_PATH}{60+i}.png', k)



    return pred
model_1m_pred = test_cnn(8, 5, 2, 24, 'a')

# model_1m_pred_b = test_cnn(8, 5, 2, 24, 'b')

# model_1m_pred_c = test_cnn(8, 5, 2, 24, 'c')

model_7m_pred = test_cnn(16, 5, 2, 40, 'a')

# model_7m_pred_b = test_cnn(16, 5, 2, 40, 'b')

# model_7m_pred_c = test_cnn(16, 5, 2, 40, 'c')

model_31m_pred = test_cnn(64, 4, 2, 64, 'a')

# model_31m_pred_b = test_cnn(64, 4, 2, 64, 'b')

# model_31m_pred_c = test_cnn(64, 4, 2, 64, 'c')
def visualize(idx: int):

    f, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(1, 5, figsize=(25, 5))

    ax0.imshow(img_tr[60+idx, :, :, 0])

    ax1.imshow(img_mask_tr[60+idx, :, :, 0])

    ax2.imshow(model_1m_pred[idx, :, :, 0])

    ax3.imshow(model_7m_pred[idx, :, :, 0])

    ax4.imshow(model_31m_pred[idx, :, :, 0])



    ax0.set_title(f'Image {60+idx}')

    ax1.set_title(f'OD Segmentation Ground Truth')

    ax2.set_title(f'Model w/ 1M trainable weight prediction')

    ax3.set_title(f'Model w/ 7M trainable weight prediction')    

    ax4.set_title(f'Model w/ 31M trainable weight prediction')
visualize(10)
visualize(11)
visualize(12)