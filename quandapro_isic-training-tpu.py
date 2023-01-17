import warnings

warnings.filterwarnings("ignore")
%%capture

!pip install efficientnet
import cv2

import numpy as np

import pandas as pd

import random

import math



import gc

import os



from efficientnet.tfkeras import *



from kaggle_datasets import KaggleDatasets



import tensorflow as tf

import tensorflow.keras as keras

from tensorflow.keras import backend as K

from tensorflow.keras import Input

from tensorflow.keras.models import Model

from tensorflow.keras.utils import *

from tensorflow.keras.layers import *

from tensorflow.keras.losses import binary_crossentropy

from tensorflow.keras.metrics import AUC



import tensorflow_addons as tfa

from tensorflow_addons.optimizers import SWA



from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.utils import shuffle



import matplotlib.pyplot as plt
DEVICE = "TPU"

if DEVICE == "TPU":

    print("connecting to TPU...")

    try:

        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

        print('Running on TPU ', tpu.master())

    except ValueError:

        print("Could not connect to TPU")

        tpu = None



    if tpu:

        try:

            print("initializing  TPU ...")

            tf.config.experimental_connect_to_cluster(tpu)

            tf.tpu.experimental.initialize_tpu_system(tpu)

            strategy = tf.distribute.experimental.TPUStrategy(tpu)

            print("TPU initialized")

        except _:

            print("failed to initialize TPU")

    else:

        DEVICE = "GPU"



if DEVICE != "TPU":

    print("Using default strategy for CPU and single GPU")

    strategy = tf.distribute.get_strategy()



if DEVICE == "GPU":

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    



AUTO     = tf.data.experimental.AUTOTUNE

REPLICAS = strategy.num_replicas_in_sync

print(f'REPLICAS: {REPLICAS}')
def set_random_seed():

    random.seed(2021)

    tf.random.set_seed(2020)

    np.random.seed(2019)

set_random_seed()
FOLDS = [0,1,2,3,4]

CONFIG = {

    'img_size': 256,

    'batch_size': 16,

    'splits': 5,

    'epochs': 30,

    'phi': 2,

     

    # Transform params

    'rot'               :   180.0,

    'hzoom'             :   8.0,

    'wzoom'             :   8.0,

    'hshift'            :   8.0,

    'wshift'            :   8.0,

    

    'mode' : 'COMBINED'

}

AUTO = tf.data.experimental.AUTOTUNE
GCS_PATH    = KaggleDatasets().get_gcs_path(f'isic-{CONFIG["img_size"]}-tfrecord')

files_train_original = np.sort(np.array(tf.io.gfile.glob(GCS_PATH  + f'/tfrecord_{CONFIG["img_size"]}_original/*.tfrecord')))

files_train_external = np.sort(np.array(tf.io.gfile.glob(GCS_PATH  + f'/tfrecord_{CONFIG["img_size"]}_external/*.tfrecord')))

print(len(files_train_original))

print(len(files_train_external))
'''

    TF DATASET

'''

def ShiftScaleRotate(image, cfg, p=0.5):    

    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]

    # output - image randomly rotated, zoomed, and shifted

    P = tf.cast( tf.random.uniform([],0,1) < p, tf.int32)

    if (P==0): return image

    

    DIM = cfg["img_size"]

    XDIM = DIM % 2 #fix for size 331

    

    rot = cfg['rot'] * tf.random.normal([1], dtype='float32')

    h_zoom = 1.0 + tf.random.normal([1], dtype='float32') / cfg['hzoom']

    w_zoom = 1.0 + tf.random.normal([1], dtype='float32') / cfg['wzoom']

    h_shift = cfg['hshift'] * tf.random.normal([1], dtype='float32') 

    w_shift = cfg['wshift'] * tf.random.normal([1], dtype='float32') 

    

    def get_mat(rotation, height_zoom, width_zoom, height_shift, width_shift):

        # returns 3x3 transformmatrix which transforms indicies



        # CONVERT DEGREES TO RADIANS

        rotation = math.pi * rotation / 180.



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



        # ZOOM MATRIX

        zoom_matrix = get_3x3_mat([one/height_zoom, zero,           zero, 

                                   zero,            one/width_zoom, zero, 

                                   zero,            zero,           one])    

        # SHIFT MATRIX

        shift_matrix = get_3x3_mat([one,  zero, height_shift, 

                                    zero, one,  width_shift, 

                                    zero, zero, one])



        return K.dot(rotation_matrix, K.dot(zoom_matrix,     shift_matrix))



    # GET TRANSFORMATION MATRIX

    m = get_mat(rot,h_zoom,w_zoom,h_shift,w_shift) 



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



def Cutout(image, DIM=256, p = 0.5, CT = 1, SZ = 0.25):

    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]

    # output - image with CT squares of side size SZ*DIM removed

    

    # DO DROPOUT WITH PROBABILITY DEFINED ABOVE

    P = tf.cast( tf.random.uniform([],0,1)<p, tf.int32)

    if (P==0)|(CT==0)|(SZ==0): return image

    

    for k in range(CT):

        # CHOOSE RANDOM LOCATION

        x = tf.cast( tf.random.uniform([],0,DIM),tf.int32)

        y = tf.cast( tf.random.uniform([],0,DIM),tf.int32)

        # COMPUTE SQUARE 

        ratio = tf.cast( tf.random.uniform([],0,1),tf.float32)

        WIDTH = tf.cast( SZ*DIM*ratio,tf.int32 ) 

        ya = tf.math.maximum(0,y-WIDTH//2)

        yb = tf.math.minimum(DIM,y+WIDTH//2)

        xa = tf.math.maximum(0,x-WIDTH//2)

        xb = tf.math.minimum(DIM,x+WIDTH//2)

        # DROPOUT IMAGE

        one = image[ya:yb,0:xa,:]

        two = tf.zeros([yb-ya,xb-xa,3]) 

        three = image[ya:yb,xb:DIM,:]

        middle = tf.concat([one,two,three],axis=1)

        image = tf.concat([image[0:ya,:,:],middle,image[yb:DIM,:,:]],axis=0)

            

    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR 

    image = tf.reshape(image,[DIM,DIM,3])

    return image



def Flip(img):

    img = tf.image.random_flip_left_right(img)

    return tf.image.random_flip_up_down(img)



def RandomBrightnessContrast(img, p = 0.5):

    P = tf.cast( tf.random.uniform([],0,1) < p, tf.int32)

    if (P==0): 

        return img

    img = tf.image.random_brightness(img, 0.1)

    return tf.image.random_contrast(img, 0.8, 1.2)



def Augment(img, cfg=CONFIG):

    img = Flip(img)

    img = ShiftScaleRotate(img, cfg)

    img = RandomBrightnessContrast(img)

    img = Cutout(img, cfg['img_size'])

    return img



def prepare_image(img, cfg=CONFIG, augment=True):    

    img = tf.image.decode_jpeg(img, channels=3)

    img = tf.cast(img, tf.float32) 

    if augment:

        img = Augment(img, cfg)

    img = tf.reshape(img, [cfg['img_size'], cfg['img_size'], 3])

    return img / 255.0



def read_tfrecord(example):

    tfrec_format = {

        'targets': tf.io.FixedLenFeature([], tf.int64),

        'image': tf.io.FixedLenFeature([], tf.string)

    }

    example = tf.io.parse_single_example(example, tfrec_format)

    return example['image'], example['targets']



def get_dataset(files, cfg, augment = False, 

                shuffle = True, repeat = False):

    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO)

    ds = ds.cache()

    

    if repeat:

        ds = ds.repeat()

    

    if shuffle: 

        ds = ds.shuffle(1024*8)

        opt = tf.data.Options()

        opt.experimental_deterministic = False

        ds = ds.with_options(opt)

        

    ds = ds.map(read_tfrecord, num_parallel_calls=AUTO)

    ds = ds.map(lambda img, target: (prepare_image(img, cfg=cfg, augment=augment), target), 

            num_parallel_calls=AUTO)

    ds = ds.batch(cfg['batch_size']*REPLICAS)

    ds = ds.prefetch(AUTO)

    return ds
'''

    Model

'''

def build_model(fold=None):

    with strategy.scope():

        EFFNET_MODEL = [EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7]

        M = EFFNET_MODEL[CONFIG['phi']]

        base_model = M(weights='noisy-student', include_top=False, input_shape=(None, None, 3))

        x = base_model.output

        x = GlobalAveragePooling2D()(x)

        out = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(x)

        model = Model(inputs=[base_model.input], outputs=[out])

        opt = keras.optimizers.Adam(lr=0.001)

        loss = tfa.losses.SigmoidFocalCrossEntropy(gamma=2.0, alpha=0.9)

#         loss = keras.losses.BinaryCrossentropy(label_smoothing=0.05)

        model.compile(optimizer=opt, 

                  loss=loss,

                  metrics=[AUC()])

    return model
'''

    Callback

'''

def get_lr_callback(batch_size=CONFIG['batch_size']):

    lr_start   = 0.000005

    lr_max     = 0.00000125 * REPLICAS * batch_size

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



    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)

    return lr_callback



def get_callbacks():

    return [keras.callbacks.ModelCheckpoint(f'efficientnetb{CONFIG["phi"]}_{CONFIG["mode"]}_fold_{fold}.h5',

                                                 verbose=1,

                                                 monitor='val_auc',

                                                 mode='max',

                                                 save_best_only=True,

                                                 save_weights_only=True),

                get_lr_callback(batch_size=CONFIG['batch_size'])]
'''

    Training

'''

def image_counts(files):

    result = 0

    for file in files:

        result += int(file.split('_')[3])

    return result



kf = KFold(n_splits=CONFIG['splits'], shuffle=False, random_state=2020)

fold = 0

val_auc = []

hists = []

files_train = files_train_original

if CONFIG["mode"] == "EXTERNAL":

    files_train = files_train_external

    

for train_idx, test_idx in kf.split(files_train):

    print(f"***********Fold {fold}*************")

    if fold not in FOLDS:

        fold += 1

        continue

    K.clear_session()

    gc.collect()

    

    if CONFIG["mode"] == "COMBINED":

        train_tfrecords = np.concatenate([files_train[train_idx], files_train_external])

        test_tfrecords = files_train[test_idx]

    else:

        train_tfrecords = files_train[train_idx]

        test_tfrecords = files_train[test_idx]

    

        

    train_ds = get_dataset(train_tfrecords, CONFIG, augment=True, shuffle=True, repeat=True)

    test_ds = get_dataset(test_tfrecords, CONFIG, augment=False, shuffle=False, repeat=False)

    

    train_len = image_counts(train_tfrecords)

    test_len = image_counts(files_train_original[test_idx])

    

    train_len = train_len // (CONFIG['batch_size']*REPLICAS)

    test_len = test_len // (CONFIG['batch_size']*REPLICAS)

    

    print(f"Training Steps: {train_len}")

    print(f"Validation Steps: {test_len}")

    

    callbacks = get_callbacks()

    

    model = build_model(fold)

    hist = model.fit(train_ds,

            steps_per_epoch=train_len,

            epochs=CONFIG['epochs'],

            callbacks=callbacks,

            validation_data=test_ds,

            validation_steps=test_len,

            verbose=1)

    fold += 1

    hists.append(hist)
val_auc = [max(hist.history['val_auc']) for hist in hists]

print(val_auc)

print(np.mean(val_auc))