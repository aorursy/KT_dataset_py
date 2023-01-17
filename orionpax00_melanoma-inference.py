!pip install efficientnet > /dev/null
import os

import re

import random

import pandas as pd

import numpy as np

import math



import tensorflow as tf

import tensorflow.keras.backend as K

from kaggle_datasets import KaggleDatasets

import efficientnet.tfkeras as efn
GCS_DS_PATH = KaggleDatasets().get_gcs_path('siim-isic-melanoma-classification')

TEST_FILES = tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords/test*')



BATCH_SIZE = 256
try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy() 



def count_data_items(filenames):

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]

    return np.sum(n)
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



## Helper Functions

def data_augment(image, label):

    image = transform(image, DIM=128)

    image = tf.image.random_flip_left_right(image)

    image = tf.image.rot90(image)

    

    return image, label  



def process_test_data(data_file):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string),

        "image_name": tf.io.FixedLenFeature([], tf.string),

    }

    data = tf.io.parse_single_example(data_file, LABELED_TFREC_FORMAT)

    img = tf.image.decode_jpeg(data['image'], channels=3)

    img = tf.image.resize(img, (128,128))

    img = tf.cast(img, tf.float32) / 255.0

    img = tf.reshape(img, [128,128, 3])



    idnum = data['image_name']



    return img, idnum
def efficientnetbx():

    return tf.keras.Sequential([

                    efn.EfficientNetB0(

                        input_shape=(128,128, 3),

                        include_top=False

                    ),

                    tf.keras.layers.GlobalAveragePooling2D(),

                    tf.keras.layers.Dense(1, activation='sigmoid')

                ])

TTA = 11

if strategy != None:

    with strategy.scope():

        model = efficientnetbx()

else:

    model = efficientnetbx()



ignore_order = tf.data.Options()

test_dataset = (

    tf.data.TFRecordDataset(

        TEST_FILES,  

        num_parallel_reads=tf.data.experimental.AUTOTUNE

    ).with_options(

        ignore_order

    ).map(

        process_test_data,

        num_parallel_calls=tf.data.experimental.AUTOTUNE

    ).map(

        data_augment, 

        num_parallel_calls=tf.data.experimental.AUTOTUNE

    ).repeat(

    ).batch(

        BATCH_SIZE * 4 

    ).prefetch(

        tf.data.experimental.AUTOTUNE

    )

)







model.load_weights("../input/efficientnetb0-128x128/fold-3.h5")



test_imgs = test_dataset.map(lambda images, ids: images)

img_ids_ds = test_dataset.map(lambda images, ids: ids).unbatch()



ct_test = count_data_items(TEST_FILES)

STEPS = TTA * ct_test/(BATCH_SIZE * 4)

pred = model.predict(test_imgs,steps=STEPS,verbose=1)[:TTA*ct_test,] 

predictions = np.mean(pred.reshape((ct_test,TTA),order='F'),axis=1) 



test_ids = next(iter(img_ids_ds.batch(ct_test))).numpy().astype('U') 



pd.DataFrame({

     'image_name'  : test_ids, 

     'target'      : predictions

    }).to_csv('submission.csv', index=False)