!pip install efficientnet

import numpy as np

import pandas as pd

import os

import re

import random, math, time

import tensorflow as tf

import tensorflow.keras as keras

import tensorflow.keras.backend as K

import efficientnet.tfkeras as effnet

import matplotlib.pyplot as plt

from kaggle_datasets import KaggleDatasets

from tqdm import tqdm
try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Running on TPU ', tpu.master())

except ValueError:

    print('No TPU')

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy()

    

AUTO = tf.data.experimental.AUTOTUNE

REPLICAS = strategy.num_replicas_in_sync

GCS_PATH = KaggleDatasets().get_gcs_path('melanoma-512x512')



print("REPLICAS: %d" % REPLICAS)



files_train = tf.io.gfile.glob(GCS_PATH + '/train*.tfrec')

files_test = tf.io.gfile.glob(GCS_PATH + '/test*.tfrec')
config = dict(

    img_read_size = 512,

    img_crop_size = 400,

    img_size      = 320,

    

    pos_ratio     = 0.2,

    

    batch_size    = 16 * REPLICAS,

    epochs        = 40,

    

    initial_learning_rate = 1e-6,

    min_learning_rate     = 1e-6,

    max_learning_rate     = 5e-5 * REPLICAS,

    rampup_epochs         = 5,

    sustain_epochs        = 0,

    exp_decay             = 0.8,

    

    test_batch_size       = 32 * REPLICAS,

    aug_reps_in_test      = 20,

    

    rot               = 180.0,

    shr               =   2.0,

    hzoom             =   8.0,

    wzoom             =   8.0,

    hshift            =   8.0,

    wshift            =   8.0

)
def decode_image(image):

    image = tf.image.decode_jpeg(image, channels=3) 

    image = tf.cast(image, tf.float32) / 255.0

    image = tf.reshape(image, [config["img_read_size"], config["img_read_size"], 3])

    return image





def _read_image_label(example):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string),

        "target": tf.io.FixedLenFeature([], tf.int64)

    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    label = tf.cast(example['target'], tf.int32)

    return image, label 





def _read_image_meta_label(example):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string),

        "sex": tf.io.FixedLenFeature([], tf.int64),

        "age_approx": tf.io.FixedLenFeature([], tf.int64),

        "target": tf.io.FixedLenFeature([], tf.int64)

    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    label = tf.cast(example['target'], tf.int32)

    meta = tf.stack([example["sex"], example["age_approx"]])

    return image, meta, label 





def _read_image_name(example):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string),

        "image_name": tf.io.FixedLenFeature([], tf.string)

    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    name = tf.cast(example['image_name'], tf.string)

    return image, name





def _read_image_meta_name(example):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string),

        "sex": tf.io.FixedLenFeature([], tf.int64),

        "age_approx": tf.io.FixedLenFeature([], tf.int64),

        "image_name": tf.io.FixedLenFeature([], tf.string)

    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    meta = tf.stack([example["sex"], example["age_approx"]])

    name = tf.cast(example['image_name'], tf.string)

    return image, meta, name 





def load_dataset_image_label(filenames):

    dataset = (

        tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) 

        .map(_read_image_label, num_parallel_calls=AUTO)

    )

    return dataset





def load_dataset_image_meta_label(filenames):

    dataset = (

        tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) 

        .map(_read_image_meta_label, num_parallel_calls=AUTO)

    )

    return dataset





def load_dataset_image_name(filenames):

    dataset = (

        tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) 

        .map(_read_image_name, num_parallel_calls=AUTO)

    )

    return dataset





def load_dataset_image_meta_name(filenames):

    dataset = (

        tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) 

        .map(_read_image_meta_name, num_parallel_calls=AUTO)

    )

    return dataset





def count_data_items(filenames):

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]

    return np.sum(n)
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





def transform(image):

    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]

    # output - image randomly rotated, sheared, zoomed, and shifted

    DIM = image.shape[1]

    XDIM = DIM % 2 # fix for size 331

    

    rot = config['rot'] * tf.random.normal([1], dtype='float32')

    shr = config['shr'] * tf.random.normal([1], dtype='float32') 

    h_zoom = 1.0 + tf.random.normal([1], dtype='float32') / config['hzoom']

    w_zoom = 1.0 + tf.random.normal([1], dtype='float32') / config['wzoom']

    h_shift = config['hshift'] * tf.random.normal([1], dtype='float32') 

    w_shift = config['wshift'] * tf.random.normal([1], dtype='float32') 



    # GET TRANSFORMATION MATRIX

    m = get_mat(rot,shr,h_zoom,w_zoom,h_shift,w_shift) 



    # LIST DESTINATION PIXEL INDICES

    x   = tf.repeat(tf.range(DIM // 2, -DIM // 2, -1), DIM)

    y   = tf.tile(tf.range(-DIM // 2, DIM // 2), [DIM])

    z   = tf.ones([DIM * DIM], dtype='int32')

    idx = tf.stack( [x,y,z] )

    

    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS

    idx2 = K.dot(m, tf.cast(idx, dtype='float32'))

    idx2 = K.cast(idx2, dtype='int32')

    idx2 = K.clip(idx2, -DIM // 2 + XDIM + 1, DIM // 2)

    

    # FIND ORIGIN PIXEL VALUES           

    idx3 = tf.stack([DIM // 2 - idx2[0,], DIM // 2 - 1 + idx2[1,]])

    d    = tf.gather_nd(image, tf.transpose(idx3))

        

    return tf.reshape(d, [DIM, DIM, 3])
def aug_img(img, *args):

    img = transform(img)

    img = tf.image.random_crop(img, [config["img_crop_size"], config["img_crop_size"], 3])

    img = tf.image.random_flip_left_right(img)

    img = tf.image.random_flip_up_down(img)

    img = tf.image.random_brightness(img, max_delta=0.5)

    img = tf.image.random_hue(img, max_delta=0.2)

    img = tf.image.random_saturation(img, 0.8, 1.2)

    img = tf.image.random_contrast(img, 0.8, 1.2)

    return (img,) + tuple(args)





def finalize(img, *args):

    img = tf.image.resize(img, [config["img_size"], config["img_size"]])

    return (img,) + tuple(args)
AUTO = tf.data.experimental.AUTOTUNE



d0 = load_dataset_image_label(files_train[:-1]).filter(lambda i, l: (l <= 0)).repeat()

d1 = load_dataset_image_label(files_train[:-1]).filter(lambda i, l: (l >= 1)).cache().repeat()



choice_dataset = tf.data.Dataset.range(2).repeat().map(lambda x: tf.cast(tf.random.uniform(shape=[]) < config['pos_ratio'], tf.int64))



data_train = (

    tf.data.experimental.choose_from_datasets([d0, d1], choice_dataset)

    .shuffle(1024)

    .map(aug_img, num_parallel_calls=AUTO)

    .map(finalize, num_parallel_calls=AUTO)

    .batch(config["batch_size"])

    .prefetch(AUTO)

)



data_val = (

    load_dataset_image_label(files_train[-1:])

    .map(finalize, num_parallel_calls=AUTO)

    .batch(config["batch_size"])

    .prefetch(AUTO)

)



data_test = (

    load_dataset_image_name(files_test)

    .map(aug_img, num_parallel_calls=AUTO)

    .map(finalize, num_parallel_calls=AUTO)

    .batch(config["test_batch_size"])

    .prefetch(AUTO)

)



print("Train", data_train)

print("Val  ", data_val)

print("Test ", data_test)
print("Estimating train dataset performance")

ones = 0

N = 200

for i, (imgs, labels) in tqdm(enumerate(data_train), total=N):

    ones += labels.numpy().sum()

    if i >= N:

        break

print("Average number of positives:\t%.4f" % (ones / (N * REPLICAS * config['batch_size'])))
K.clear_session()

model = keras.Sequential([

    effnet.EfficientNetB5(weights='imagenet', include_top=False),

    keras.layers.GlobalAveragePooling2D(),

    keras.layers.Dense(1024, activation='relu'),

    keras.layers.Dropout(0.2),

    keras.layers.Dense(512, activation='relu'),

    keras.layers.Dropout(0.2),

    keras.layers.Dense(256, activation='relu'),

    keras.layers.Dense(1, activation='sigmoid')

])

model.compile(

    optimizer=keras.optimizers.Adam(1e-4), 

    loss=keras.losses.BinaryCrossentropy(), 

    metrics=[keras.metrics.AUC()])
def lrfn(epoch):

    if epoch < config['rampup_epochs']:

        return config['initial_learning_rate'] + (config['max_learning_rate'] - config['initial_learning_rate']) * epoch / config['rampup_epochs']

    elif epoch < config['rampup_epochs'] + config['sustain_epochs']:

        return max_lr

    else:

        return config['min_learning_rate'] + (config['max_learning_rate'] - config['min_learning_rate']) * config['exp_decay'] ** (epoch - config['rampup_epochs'] - config['sustain_epochs'])

    pass



    

lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda epoch: lrfn(epoch), verbose=True)



lrs = []

for e in range(config['epochs']):

    lrs.append(lrfn(e))

plt.plot(lrs)

plt.ylim(0)
N_POS_APPROX = 584

STEPS_PER_EPOCH = int(np.ceil(N_POS_APPROX / config['pos_ratio'] / config['batch_size']))

history = model.fit(

    data_train,

    epochs          = config['epochs'],

    steps_per_epoch = STEPS_PER_EPOCH,

    validation_data = data_val,

    callbacks       = [lr_callback]

)
plt.plot(history.history['loss'], color='firebrick')

plt.plot(history.history['val_loss'], color='firebrick',linestyle='--')

plt.figure()

plt.plot(history.history['auc'], color='firebrick')

plt.plot(history.history['val_auc'], color='firebrick',linestyle='--')
names = list(data_test.map(lambda i, n: n).as_numpy_iterator())

names = np.vectorize(lambda x: x.decode('utf-8'))(np.concatenate(names, 0))

preds = model.predict(data_test.map(lambda i, n: i), verbose=1)
names = list(data_test.map(lambda i, n: n).as_numpy_iterator())

names = np.vectorize(lambda x: x.decode('utf-8'))(np.concatenate(names, 0))
preds = [model.predict(data_test.map(lambda i, n: i), verbose=1) for rep in range(config['aug_reps_in_test'])]
pd.DataFrame({'image_name': names, 'target': np.concatenate(preds, 1).mean(1)}).groupby('image_name').mean().reset_index().to_csv('submission.csv', index=False)