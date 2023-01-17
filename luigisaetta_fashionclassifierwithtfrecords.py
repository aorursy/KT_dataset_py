# we need to install the package for the NN used (EfficientNet)

!pip install efficientnet
# the idea of this Notebook is to show how to develop a "Fashion Classifier"

# where the input is an image and the output is the MainCategory the item belongs to

# to be as fast as possible image and metadata have been packed inb file in TFRecords format

# see description of the dataset



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import tensorflow as tf

from tensorflow.keras.datasets import fashion_mnist

from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Activation, Input, Flatten, Dense

from tensorflow.keras import Model

import tensorflow.keras.backend as K

import efficientnet.tfkeras as efn



import matplotlib.pyplot as plt

import random as python_random

import re,  math

import time



from sklearn.model_selection import KFold



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# to remove some warnings



# TF2 way to reduce logging

# this remove also INFO, verify if needed

import logging

logger = tf.get_logger()

logger.setLevel(logging.ERROR)
# taken from keras.io FAQs



# The below is necessary for starting Numpy generated random numbers

# in a well-defined initial state.

np.random.seed(123)



# The below is necessary for starting core Python generated random numbers

# in a well-defined state.

python_random.seed(123)



# The below set_seed() will make random number generation

# in the TensorFlow backend have a well-defined initial state.

tf.random.set_seed(1234)
# global constants

DEVICE = 'GPU'



# to make run shorter only 3 folds and 10 epochs. 

# Go for 5 folds and more epoch to increase accuracy

EPOCHS = 10

VERBOSE = 1

BATCH_SIZES = 64

FOLDS = 3



# 256x256

IMG_SIZES = 256



# WHICH EFFICIENTNET TO USE (B?, B0 from B7)

EFF_NETS = 0



# WEIGHTS FOR FOLD MODELS WHEN PREDICTING TEST

WGTS = 1/FOLDS



N_CLASSES = 7
# to define the correct distribution strategy for GPU



if DEVICE == "GPU":

    n_gpu = len(tf.config.experimental.list_physical_devices('GPU'))

    print("Num GPUs Available: ", n_gpu)

    

    if n_gpu > 1:

        print("Using strategy for multiple GPU")

        strategy = tf.distribute.MirroredStrategy()

    else:

        print('Standard strategy for GPU...')

        strategy = tf.distribute.get_strategy()



AUTO     = tf.data.experimental.AUTOTUNE

REPLICAS = strategy.num_replicas_in_sync



print(f'REPLICAS: {REPLICAS}')
# TFRecords file for training (not Kaggle location !)

BASE_DIR = '/kaggle/input/fashion-dataset-tfrecords-256x256'
# parameters for Image Augmentation

ROT_ = 180.0

SHR_ = 2.0

HZOOM_ = 8.0

WZOOM_ = 8.0

HSHIFT_ = 8.0

WSHIFT_ = 8.0
def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):

    # returns 3x3 transformmatrix which transforms indicies

        

    # CONVERT DEGREES TO RADIANS

    rotation = math.pi * rotation / 180.

    shear    = math.pi * shear    / 180.



    def get_3x3_mat(lst):

        return tf.reshape(tf.concat([lst],axis=0), [3,3])

    

    # ROTATION MATRIX

    c1   = tf.math.cos(rotation)

    s1   = tf.math.sin(rotation)

    one  = tf.constant([1],dtype='float32')

    zero = tf.constant([0],dtype='float32')

    

    rotation_matrix = get_3x3_mat([c1,   s1,   zero, 

                                   -s1,  c1,   zero, 

                                   zero, zero, one])    

    # SHEAR MATRIX

    c2 = tf.math.cos(shear)

    s2 = tf.math.sin(shear)    

    

    shear_matrix = get_3x3_mat([one,  s2,   zero, 

                                zero, c2,   zero, 

                                zero, zero, one])        

    # ZOOM MATRIX

    zoom_matrix = get_3x3_mat([one/height_zoom, zero,           zero, 

                               zero,            one/width_zoom, zero, 

                               zero,            zero,           one])    

    # SHIFT MATRIX

    shift_matrix = get_3x3_mat([one,  zero, height_shift, 

                                zero, one,  width_shift, 

                                zero, zero, one])

    

    return K.dot(K.dot(rotation_matrix, shear_matrix), 

                 K.dot(zoom_matrix,     shift_matrix))



def transform(image, DIM=256):    

    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]

    # output - image randomly rotated, sheared, zoomed, and shifted

    XDIM = DIM%2 #fix for size 331

    

    rot = ROT_ * tf.random.normal([1], dtype='float32')

    shr = SHR_ * tf.random.normal([1], dtype='float32') 

    h_zoom = 1.0 + tf.random.normal([1], dtype='float32') / HZOOM_

    w_zoom = 1.0 + tf.random.normal([1], dtype='float32') / WZOOM_

    h_shift = HSHIFT_ * tf.random.normal([1], dtype='float32') 

    w_shift = WSHIFT_ * tf.random.normal([1], dtype='float32') 



    # GET TRANSFORMATION MATRIX

    m = get_mat(rot,shr,h_zoom,w_zoom,h_shift,w_shift) 



    # LIST DESTINATION PIXEL INDICES

    x   = tf.repeat(tf.range(DIM//2, -DIM//2,-1), DIM)

    y   = tf.tile(tf.range(-DIM//2, DIM//2), [DIM])

    z   = tf.ones([DIM*DIM], dtype='int32')

    idx = tf.stack( [x,y,z] )

    

    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS

    idx2 = K.dot(m, tf.cast(idx, dtype='float32'))

    idx2 = K.cast(idx2, dtype='int32')

    idx2 = K.clip(idx2, -DIM//2+XDIM+1, DIM//2)

    

    # FIND ORIGIN PIXEL VALUES           

    idx3 = tf.stack([DIM//2-idx2[0,], DIM//2-1+idx2[1,]])

    d    = tf.gather_nd(image, tf.transpose(idx3))

        

    return tf.reshape(d,[DIM, DIM,3])
# not using metadata (only image, for now)

def read_labeled_tfrecord(example):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "image_name": tf.io.FixedLenFeature([], tf.string),

        'base_colour' : tf.io.FixedLenFeature([], tf.int64), # shape [] means single element

        'target' : tf.io.FixedLenFeature([], tf.int64)

    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = example['image']

    image_name = example['image_name']

    colour = example['base_colour']

    label = example['target']

    return image, label





def read_unlabeled_tfrecord(example, return_image_name):

    tfrec_format = {

        'image'                        : tf.io.FixedLenFeature([], tf.string),

        'image_name'                   : tf.io.FixedLenFeature([], tf.string),

    }

    example = tf.io.parse_single_example(example, tfrec_format)

    return example['image'], example['image_name'] if return_image_name else 0



# here we request also image augmentation



def prepare_image(img, augment=True, dim=256):    

    img = tf.image.decode_jpeg(img, channels=3)

    

    # normalizzazione

    img = tf.cast(img, tf.float32) / 255.0

    

    if augment:

        # random rotation...

        img = transform(img, DIM=dim)

        img = tf.image.random_flip_left_right(img)

        

        img = tf.image.random_saturation(img, 0.7, 1.3)

        img = tf.image.random_contrast(img, 0.8, 1.2)

        img = tf.image.random_brightness(img, 0.1)

                      

    img = tf.reshape(img, [dim,dim, 3])

            

    return img



# count # of images in files.. (embedded in file name)

def count_data_items(filenames):

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) 

         for filename in filenames]

    return np.sum(n)
def get_dataset(files, augment = False, shuffle = False, repeat = False, 

                labeled=True, return_image_names=True, batch_size=16, dim=256):

    

    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO)

    ds = ds.cache()

    

    if repeat:

        ds = ds.repeat()

    

    if shuffle: 

        ds = ds.shuffle(1024*8)

        opt = tf.data.Options()

        opt.experimental_deterministic = False

        ds = ds.with_options(opt)

        

    if labeled: 

        ds = ds.map(read_labeled_tfrecord, num_parallel_calls=AUTO)

    else:

        ds = ds.map(lambda example: read_unlabeled_tfrecord(example, return_image_names), 

                    num_parallel_calls=AUTO)      

    

    ds = ds.map(lambda img, imgname_or_label: (prepare_image(img, augment=augment, dim=dim), 

                                               imgname_or_label), 

                num_parallel_calls=AUTO)

    

    ds = ds.batch(batch_size * REPLICAS)

    ds = ds.prefetch(AUTO)

    return ds
# here we define the DNN Model



EFNS = [efn.EfficientNetB0, efn.EfficientNetB1, efn.EfficientNetB2, efn.EfficientNetB3, 

        efn.EfficientNetB4, efn.EfficientNetB5, efn.EfficientNetB6, efn.EfficientNetB7]



# as default it used B0



def build_model(dim = 128, ef = 0):

    inp = tf.keras.layers.Input(shape=(dim, dim, 3))

    

    base = EFNS[ef](input_shape=(dim, dim, 3), weights='noisy-student', include_top = False)

    

    x = base(inp)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    

    x = tf.keras.layers.Dense(N_CLASSES, activation='softmax')(x)

    

    model = tf.keras.Model(inputs = inp, outputs = x)

    

    opt = tf.keras.optimizers.Adam(learning_rate = 0.001)

    

    model.compile(optimizer = opt, loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

    

    return model
# temporal variation of learning rate



def get_lr_callback(batch_size=8):

    lr_start   = 0.000005

    lr_max     = 0.000020 * REPLICAS * batch_size/16

    lr_min     = 0.000001

    lr_ramp_ep = 5

    lr_sus_ep  = 0

    lr_decay   = 0.8

   

    def lrfn(epoch):

        if epoch < lr_ramp_ep:

            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start

            

        elif epoch < lr_ramp_ep + lr_sus_ep:

            lr = lr_max

            

        else:

            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min

            

        return lr



    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)

    

    return lr_callback
# constant to customize output

SHOW_FILES = True

PLOT = 1



files_test = np.sort(np.array(tf.io.gfile.glob(BASE_DIR + '/test*.tfrec')))



skf = KFold(n_splits = FOLDS, shuffle = True, random_state=42)



# for others investigations

# we store all the histories

histories = []



# count total provided train files

# these will be split in folds

num_total_train_files = len(tf.io.gfile.glob(BASE_DIR + '/train*.tfrec'))



# global info

print('#'*60)

print('#### Global info:')

print('#### Image Size %i, EfficientNet B%i, batch_size %i'%

          (IMG_SIZES, EFF_NETS, BATCH_SIZES*REPLICAS))

print('#### Epochs: %i' %(EPOCHS))

print('')



for fold,(idxT,idxV) in enumerate(skf.split(np.arange(num_total_train_files))):

    

    tStart = time.time()

    

    # display fold info

    print('#'*60) 

    print('#### FOLD', fold+1)    

    

    # CREATE TRAIN AND VALIDATION SUBSETS

    files_train = tf.io.gfile.glob([BASE_DIR + '/train%.2i*.tfrec'%x for x in idxT])

    

    np.random.shuffle(files_train) 

    print('#'*60)

    

    files_valid = tf.io.gfile.glob([BASE_DIR + '/train%.2i*.tfrec'%x for x in idxV])

    

    if SHOW_FILES:

        print('#### Number of training images', count_data_items(files_train))

        print('#### Number of validation images', count_data_items(files_valid))

        

    # BUILD MODEL

            

    K.clear_session()

    with strategy.scope():

        model = build_model(dim=IMG_SIZES, ef=EFF_NETS)

        

    # callback to save best model for each fold

    sv = tf.keras.callbacks.ModelCheckpoint(

        'fold-%i.h5'%fold, monitor='val_loss', verbose=1, save_best_only=True,

        save_weights_only=True, mode='min', save_freq='epoch')

   

    # TRAIN

    history = model.fit(

        get_dataset(files_train, augment=True, shuffle=True, repeat=True,

                dim=IMG_SIZES, batch_size = BATCH_SIZES), 

        epochs=EPOCHS, 

        callbacks = [sv, get_lr_callback(BATCH_SIZES)], 

        steps_per_epoch = count_data_items(files_train)/BATCH_SIZES//REPLICAS,

        validation_data = get_dataset(files_valid, augment=False,shuffle=False,

                repeat = False, dim=IMG_SIZES),

        verbose=VERBOSE

    )

    

    # save all histories

    histories.append(history)

    

    # evaluate accuracy on FOLD

    max_acc = np.max(history.history['accuracy'])

    print('')

    print('Max train accuracy: ', round(max_acc, 4))

    

    # load best model

    model.load_weights('fold-%i.h5'%fold)

    

    # evaluate max acc on validation set

    valid_loss, valid_acc = model.evaluate(get_dataset(files_valid, augment=False,shuffle=False,

                repeat = False, dim=IMG_SIZES), verbose = 0, batch_size = 4*BATCH_SIZES)



    print('Validation accuracy: ', round(valid_acc, 4))

    

    tElapsed = round(time.time() - tStart, 1)

    

    print(' ')

    print('Time (sec) elapsed for fold: ', tElapsed)

    print('...')

    print('...')

def plot_loss(hist, fold):

    plt.figure(figsize=(14,6))

    plt.title('Loss fold n. ' + str(fold +1))

    plt.plot(hist.history['loss'], label='Training loss')

    plt.plot(hist.history['val_loss'], label='Validation loss')

    plt.legend(loc='upper right')

    plt.ylabel('loss')

    plt.xlabel('epoch')

    plt.grid()

    plt.show();

    

def plot_acc(hist, fold):

    plt.figure(figsize=(14,6))

    plt.title('Acc. fold n. ' + str(fold +1))

    plt.plot(hist.history['accuracy'], label='Training accuracy')

    plt.plot(hist.history['val_accuracy'], label='Validation accuracy')

    plt.legend(loc='lower right')

    plt.ylabel('acc')

    plt.xlabel('epoch')

    plt.grid()

    plt.show();
# plot loss for each fold

for fold in range(FOLDS):

    hist1 = histories[fold]



    plot_loss(hist1, fold)
# plot accuracy for each fold



for fold in range(FOLDS):

    hist1 = histories[fold]



    plot_acc(hist1, fold)
# calculate on validation set accuracy

wi = [1/FOLDS]*FOLDS



avg_acc = 0



for fold in range(FOLDS):

    model.load_weights('fold-%i.h5'%fold)

    

    valid_loss, valid_acc = model.evaluate(get_dataset(files_valid, augment=False,shuffle=False, \

                repeat = False, dim=IMG_SIZES), verbose = 0, batch_size = 4*BATCH_SIZES)



    print('Validation accuracy fold n.', fold+1, ': ', round(valid_acc, 4))



    avg_acc += valid_acc * wi[fold]



print('Average accuracy: ', round(avg_acc,4))
# Test dataset (hold out)

    

wi = [1/FOLDS]*FOLDS



avg_acc = 0



for fold in range(FOLDS):

    model.load_weights('fold-%i.h5'%fold)

    

    test_loss, test_acc = model.evaluate(get_dataset(files_test, augment=False,shuffle=False, \

                repeat = False, dim=IMG_SIZES), verbose = 0, batch_size = 4*BATCH_SIZES)



    print('Test accuracy fold n.', fold+1, ': ', round(test_acc, 4))



    avg_acc += test_acc * wi[fold]



print('Average accuracy: ', round(avg_acc,4))