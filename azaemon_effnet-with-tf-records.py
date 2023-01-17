!pip install -q efficientnet
import pandas as pd
import numpy as np

#

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#

import seaborn as sns
import plotly.express as px

#

import os
import random
import re
import math
import time

from tqdm import tqdm
from tqdm.keras import TqdmCallback


from pandas_summary import DataFrameSummary

import warnings


warnings.filterwarnings('ignore') # Disabling warnings for clearer outputs



seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
import tensorflow as tf
import tensorflow.keras.backend as K
import efficientnet.tfkeras as efn
from kaggle_datasets import KaggleDatasets

tf.random.set_seed(seed_val)
GCS_PATH1 = KaggleDatasets().get_gcs_path('landmarktfv31')
GCS_PATH2 = KaggleDatasets().get_gcs_path('landmarktfv32')
GCS_PATH3 = KaggleDatasets().get_gcs_path('landmarktfv33')
GCS_PATH4 = KaggleDatasets().get_gcs_path('landmarktfrecordtest')


filenames_train1 = tf.io.gfile.glob(GCS_PATH1 + '/*.tfrec')
filenames_train2 = tf.io.gfile.glob(GCS_PATH2 + '/*.tfrec')
filenames_train3 = tf.io.gfile.glob(GCS_PATH3 + '/*.tfrec')

filenames_test = np.array(tf.io.gfile.glob(GCS_PATH4 + '/*.tfrec'))
#filenames_validation= np.array(tf.io.gfile.glob(GCS_PATH1 + '/*.tfrec'))
filenames_train=  np.array(filenames_train1 + filenames_train2 + filenames_train3)
raw_dataset = tf.data.TFRecordDataset(filenames_train)
for raw_record in raw_dataset.take(1):
  example = tf.train.Example()
  example.ParseFromString(raw_record.numpy())
  #li= raw_record.numpy()
  print(example)
cfg = dict(
           batch_size=16,
           img_size=256,
    
           lr_start=0.000005,
           lr_max=0.00000125,
           lr_min=0.000001,
           lr_rampup=5,
           lr_sustain=0,
           lr_decay=0.8,
           epochs=18,
    
           transform_prob=1.0,
           rot=180.0,
           shr=2.0,
           hzoom=8.0,
           wzoom=8.0,
           hshift=8.0,
           wshift=8.0,
    
           optimizer='adam',
           label_smooth_fac=0.05,
           tta_steps=20
            
        )
DEVICE = 'TPU'
if DEVICE == 'TPU':
    print('connecting to TPU...')
    try:        
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.master())
    except ValueError:
        print('Could not connect to TPU')
        tpu = None

    if tpu:
        try:
            print('Initializing  TPU...')
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.experimental.TPUStrategy(tpu)
            print('TPU initialized')
        except _:
            print('Failed to initialize TPU!')
    else:
        DEVICE = 'GPU'

if DEVICE != 'TPU':
    print('Using default strategy for CPU and single GPU')
    strategy = tf.distribute.get_strategy()

if DEVICE == 'GPU':
    print('Num GPUs Available: ',
          len(tf.config.experimental.list_physical_devices('GPU')))

print('REPLICAS: ', strategy.num_replicas_in_sync)
AUTO = tf.data.experimental.AUTOTUNE
IMAGE_SIZE = [256, 256] # At this size, a GPU will run out of memory. Use the TPU.
                        # For GPU training, please select 224 x 224 px image size.
EPOCHS = 12
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
# numpy and matplotlib defaults
np.set_printoptions(threshold=15, linewidth=80)

def batch_to_numpy_images_and_labels(data):
    images, labels = data
    numpy_images = images.numpy()
    numpy_labels = labels.numpy()
    if numpy_labels.dtype == object: # binary string in this case, these are image ID strings
        numpy_labels = [None for _ in enumerate(numpy_images)]
    # If no labels, only image IDs, return None for labels (this is the case for test data)
    return numpy_images, numpy_labels

def title_from_label_and_target(label, correct_label):
    if correct_label is None:
        return CLASSES[label], True
    correct = (label == correct_label)
    return "{} [{}{}{}]".format(CLASSES[label], 'OK' if correct else 'NO', u"\u2192" if not correct else '',
                                CLASSES[correct_label] if not correct else ''), correct

def display_one_flower(image, title, subplot, red=False, titlesize=16):
    plt.subplot(*subplot)
    plt.axis('off')
    plt.imshow(image)
    if len(title) > 0:
        plt.title(title, fontsize=int(titlesize) if not red else int(titlesize/1.2), color='red' if red else 'black', fontdict={'verticalalignment':'center'}, pad=int(titlesize/1.5))
    return (subplot[0], subplot[1], subplot[2]+1)
    
def display_batch_of_images(databatch, predictions=None):
    """This will work with:
    display_batch_of_images(images)
    display_batch_of_images(images, predictions)
    display_batch_of_images((images, labels))
    display_batch_of_images((images, labels), predictions)
    """
    # data
    images, labels = batch_to_numpy_images_and_labels(databatch)
    if labels is None:
        labels = [None for _ in enumerate(images)]
        
    # auto-squaring: this will drop data that does not fit into square or square-ish rectangle
    rows = int(math.sqrt(len(images)))
    cols = len(images)//rows
        
    # size and spacing
    FIGSIZE = 13.0
    SPACING = 0.1
    subplot=(rows,cols,1)
    if rows < cols:
        plt.figure(figsize=(FIGSIZE,FIGSIZE/cols*rows))
    else:
        plt.figure(figsize=(FIGSIZE/rows*cols,FIGSIZE))
    
    # display
    for i, (image, label) in enumerate(zip(images[:rows*cols], labels[:rows*cols])):
        #title = '' if label is None else CLASSES[label]
        title='Landmark'
        correct = True
        if predictions is not None:
            title, correct = title_from_label_and_target(predictions[i], label)
        dynamic_titlesize = FIGSIZE*SPACING/max(rows,cols)*40+3 # magic formula tested to work from 1x1 to 10x10 images
        subplot = display_one_flower(image, title, subplot, not correct, titlesize=dynamic_titlesize)
    
    #layout
    plt.tight_layout()
    if label is None and predictions is None:
        plt.subplots_adjust(wspace=0, hspace=0)
    else:
        plt.subplots_adjust(wspace=SPACING, hspace=SPACING)
    plt.show()

def display_confusion_matrix(cmat, score, precision, recall):
    plt.figure(figsize=(15,15))
    ax = plt.gca()
    ax.matshow(cmat, cmap='Reds')
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, fontdict={'fontsize': 7})
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
    ax.set_yticks(range(len(CLASSES)))
    ax.set_yticklabels(CLASSES, fontdict={'fontsize': 7})
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    titlestring = ""
    if score is not None:
        titlestring += 'f1 = {:.3f} '.format(score)
    if precision is not None:
        titlestring += '\nprecision = {:.3f} '.format(precision)
    if recall is not None:
        titlestring += '\nrecall = {:.3f} '.format(recall)
    if len(titlestring) > 0:
        ax.text(101, 1, titlestring, fontdict={'fontsize': 18, 'horizontalalignment':'right', 'verticalalignment':'top', 'color':'#804040'})
    plt.show()
    
def display_training_curves(training, validation, title, subplot):
    if subplot%10==1: # set up the subplots on the first call
        plt.subplots(figsize=(10,10), facecolor='#F0F0F0')
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.set_facecolor('#F8F8F8')
    ax.plot(training)
    ax.plot(validation)
    ax.set_title('model '+ title)
    ax.set_ylabel(title)
    #ax.set_ylim(0.28,1.05)
    ax.set_xlabel('epoch')
    ax.legend(['train', 'valid.'])
def get_mat(rotation, shear, height_zoom, width_zoom, height_shift,
            width_shift):
    
    ''' Settings for image preparations '''

    # CONVERT DEGREES TO RADIANS
    rotation = math.pi * rotation / 180.
    shear = math.pi * shear / 180.

    # ROTATION MATRIX
    c1 = tf.math.cos(rotation)
    s1 = tf.math.sin(rotation)
    one = tf.constant([1], dtype='float32')
    zero = tf.constant([0], dtype='float32')
    rotation_matrix = tf.reshape(
        tf.concat([c1, s1, zero, -s1, c1, zero, zero, zero, one], axis=0),
        [3, 3])

    # SHEAR MATRIX
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)
    shear_matrix = tf.reshape(
        tf.concat([one, s2, zero, zero, c2, zero, zero, zero, one], axis=0),
        [3, 3])

    # ZOOM MATRIX
    zoom_matrix = tf.reshape(
        tf.concat([
            one / height_zoom, zero, zero, zero, one / width_zoom, zero, zero,
            zero, one
        ],
                  axis=0), [3, 3])

    # SHIFT MATRIX
    shift_matrix = tf.reshape(
        tf.concat(
            [one, zero, height_shift, zero, one, width_shift, zero, zero, one],
            axis=0), [3, 3])

    return K.dot(K.dot(rotation_matrix, shear_matrix),
                 K.dot(zoom_matrix, shift_matrix))


def transform(image, cfg):
    
    ''' This function takes input images of [: , :, 3] sizes and returns them as randomly rotated, sheared, shifted and zoomed. '''

    DIM = cfg['img_size']
    XDIM = DIM % 2  # fix for size 331

    rot = cfg['rot'] * tf.random.normal([1], dtype='float32')
    shr = cfg['shr'] * tf.random.normal([1], dtype='float32')
    h_zoom = 1.0 + tf.random.normal([1], dtype='float32') / cfg['hzoom']
    w_zoom = 1.0 + tf.random.normal([1], dtype='float32') / cfg['wzoom']
    h_shift = cfg['hshift'] * tf.random.normal([1], dtype='float32')
    w_shift = cfg['wshift'] * tf.random.normal([1], dtype='float32')

    # GET TRANSFORMATION MATRIX
    m = get_mat(rot, shr, h_zoom, w_zoom, h_shift, w_shift)

    # LIST DESTINATION PIXEL INDICES
    x = tf.repeat(tf.range(DIM // 2, -DIM // 2, -1), DIM)
    y = tf.tile(tf.range(-DIM // 2, DIM // 2), [DIM])
    z = tf.ones([DIM * DIM], dtype='int32')
    idx = tf.stack([x, y, z])

    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = K.dot(m, tf.cast(idx, dtype='float32'))
    idx2 = K.cast(idx2, dtype='int32')
    idx2 = K.clip(idx2, -DIM // 2 + XDIM + 1, DIM // 2)

    # FIND ORIGIN PIXEL VALUES
    idx3 = tf.stack([DIM // 2 - idx2[0, ], DIM // 2 - 1 + idx2[1, ]])
    d = tf.gather_nd(image, tf.transpose(idx3))

    return tf.reshape(d, [DIM, DIM, 3])

def get_f_data():
    dataset = load_dataset(filenames_train, labeled=True)
    #print(dataset)

    dataset = dataset.shuffle(2048)
    
    return dataset

def get_t_data(dataset):
    DATASET_SIZE= count_data_items(filenames_train)
    print("Dataset Size: ",DATASET_SIZE)
    train_size = int(0.8 * DATASET_SIZE)
    train_dataset = dataset.take(train_size)
    train_dataset = train_dataset.map(data_augment, num_parallel_calls=AUTO)

    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(AUTO)
    
    return train_dataset

def get_v_data(dataset):
    DATASET_SIZE= count_data_items(filenames_train)
    train_size = int(0.8 * DATASET_SIZE)
    val_size = int(0.20 * DATASET_SIZE)
    valid_dataset = dataset.skip(train_size)
    valid_dataset = valid_dataset.take(val_size)
    
    
    #valid_dataset = valid_dataset.repeat()
    valid_dataset = valid_dataset.batch(BATCH_SIZE)
    valid_dataset = valid_dataset.prefetch(AUTO)
    
    return valid_dataset
    
    
def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU
    return image

def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "landmark_id": tf.io.FixedLenFeature([890], tf.float32 ),  # shape [] means single element
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    label = tf.cast(example['landmark_id'], tf.float32)
    #print(label)
    #l= K.argmax(label)
    #print(label,l)
    return image, label # returns a dataset of (image, label) pairs

def read_unlabeled_tfrecord(example):
    UNLABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "id": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element
        # class is missing, this competitions's challenge is to predict flower classes for the test dataset
    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    idnum = example['id']
    return image # returns a dataset of image(s)

def load_dataset(filenames, labeled=True, ordered=False):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # disregarding data order. Order does not matter since we will be shuffling the data anyway.

    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
    if labeled:   
        dataset = dataset.map(read_labeled_tfrecord, num_parallel_calls=AUTO)
    else:
        dataset = dataset.map(lambda example: read_unlabeled_tfrecord(example), num_parallel_calls=AUTO)
    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
    return dataset

def data_augment(img, label):
    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),
    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part
    # of the TPU while the TPU itself is computing gradients.
    #image = tf.image.random_flip_left_right(image)
    #img = tf.cast(img, tf.float32) / 255.0
    img = transform(img, cfg)

    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_saturation(img, 0.7, 1.3)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    img = tf.image.random_brightness(img, 0.1)
    
    return img, label   

def get_test_dataset(ordered=True):
    dataset = load_dataset(filenames_test, labeled=False, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def count_data_items(filenames):
    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

"""

def get_training_dataset():
    dataset = load_dataset(filenames_train, labeled=True)
    #print(dataset)
    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    #dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(2048)
    
    #i=1
    #for e in dataset:
    #    print(i)
    #    i=i+1
    
    DATASET_SIZE= count_data_items(filenames_train)
    train_size = int(0.8 * DATASET_SIZE)
    val_size = int(0.20 * DATASET_SIZE)
    train_dataset = dataset.take(train_size)
    train_dataset = train_dataset.repeat()
    valid_dataset = dataset.skip(train_size)
    valid_dataset = valid_dataset.take(val_size)
    
    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(AUTO)
    valid_dataset = valid_dataset.batch(BATCH_SIZE)
    valid_dataset = valid_dataset.cache()
    valid_dataset = valid_dataset.prefetch(AUTO)
    
    #dataset = dataset.batch(BATCH_SIZE)
    #dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return train_dataset,valid_dataset

def get_validation_dataset(ordered=False):
    dataset = load_dataset(filenames_validation, labeled=True, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset


NUM_TRAINING_IMAGES =  int(0.8 * count_data_items(filenames_train))
#NUM_VALIDATION_IMAGES = count_data_items(filenames_validation)
NUM_TEST_IMAGES = count_data_items(filenames_test)
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
print('Dataset: {} training images, {} unlabeled test images'.format(NUM_TRAINING_IMAGES, NUM_TEST_IMAGES))

"""
# data dump
dataset= get_f_data()
#NUM_TRAINING_IMAGES =  count_data_items(filenames_train)
#print("Dataset: ", NUM_TRAINING_IMAGES)

print("Training data shapes:")
training_dataset = get_t_data(dataset)
for image, label in training_dataset.take(3):
    print(image.numpy().shape, label.numpy().shape)
print("Training data label examples:", label.numpy())

validation_dataset= get_t_data(dataset)
print("Validation data shapes:")
for image, label in validation_dataset.take(3):
    print(image.numpy().shape, label.numpy().shape)
print("Validation data label examples:", label.numpy())

print("Test data shapes:")
for image in get_test_dataset().take(3):
    print(image.numpy().shape)
#print("Test data IDs:", idnum.numpy().astype('U')) # U=unicode string
# Peek at training data
#training_dataset = get_training_dataset()
training_dataset = training_dataset.unbatch().batch(20)
print(training_dataset)
train_batch = iter(training_dataset)
print(train_batch)
# run this cell again for next set of images
display_batch_of_images(next(train_batch))
# peer at test data
test_dataset = get_test_dataset()
test_dataset = test_dataset.unbatch().batch(20)
test_batch = iter(test_dataset)
# run this cell again for next set of images
display_batch_of_images(next(test_batch))
def getLearnRateCallback(cfg):
    
    ''' Using callbacks for learning rate adjustments. '''
    
    lr_start = cfg['lr_start']
    lr_max = cfg['lr_max'] * strategy.num_replicas_in_sync * cfg['batch_size']
    lr_min = cfg['lr_min']
    lr_rampup = cfg['lr_rampup']
    lr_sustain = cfg['lr_sustain']
    lr_decay = cfg['lr_decay']

    def lrfn(epoch):
        if epoch < lr_rampup:
            lr = (lr_max - lr_start) / lr_rampup * epoch + lr_start
        elif epoch < lr_rampup + lr_sustain:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_rampup -
                                                lr_sustain) + lr_min
        return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)
    return lr_callback

callbacks = [getLearnRateCallback(cfg)]
def get_model():
    
    ''' This function gets the layers inclunding efficientnet ones. '''
    with strategy.scope():
        model_input = tf.keras.Input(shape=(cfg['img_size'], cfg['img_size'], 3),
                                     name='img_input')

        dummy = tf.keras.layers.Lambda(lambda x: x)(model_input)

        outputs = []
        
        x = efn.EfficientNetB3(include_top=False,
                               weights='noisy-student',
                               input_shape=(cfg['img_size'], cfg['img_size'], 3),
                               pooling='avg')(dummy)
        x = tf.keras.layers.Dense(890, activation='softmax')(x)
        """
        outputs.append(x)

        x = efn.EfficientNetB4(include_top=False,
                               weights='noisy-student',
                               input_shape=(cfg['img_size'], cfg['img_size'], 3),
                               pooling='avg')(dummy)
        x = tf.keras.layers.Dense(890, activation='softmax')(x)
        outputs.append(x)

        x = efn.EfficientNetB5(include_top=False,
                               weights='noisy-student',
                               input_shape=(cfg['img_size'], cfg['img_size'], 3),
                               pooling='avg')(dummy)
        x = tf.keras.layers.Dense(890, activation='softmax')(x)
        outputs.append(x)
        """
        model = tf.keras.Model(model_input, x, name='aNetwork')
    
    with strategy.scope():
        model.compile(optimizer=cfg['optimizer'],
                      loss=[
                          tf.keras.losses.BinaryCrossentropy(
                              label_smoothing=cfg['label_smooth_fac'])
                      ],
                      metrics=[tf.keras.metrics.AUC(name='auc')])
    model.summary()
    return model
def pre_model():
    with strategy.scope():
        pretrained_model = tf.keras.applications.DenseNet201(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])
        #pretrained_model = tf.keras.applications.Xception(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])
        #pretrained_model = tf.keras.applications.DenseNet169(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])
        pretrained_model.trainable = False # False = transfer learning, True = fine-tuning

        model = tf.keras.Sequential([
            pretrained_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(890, activation='softmax')
        ])
    with strategy.scope():
        model.compile(
            optimizer= tf.keras.optimizers.SGD(lr=0.0001),
            loss = 'categorical_crossentropy',
            metrics=['accuracy']
        )
    model.summary()
    return model
dataset= get_f_data()
NUM_TRAINING_IMAGES =  int(0.8 * count_data_items(filenames_train))
NUM_VALIDATION_IMAGES = int(0.2 * count_data_items(filenames_train))
NUM_TEST_IMAGES = count_data_items(filenames_test)
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
print('Dataset: {} training images,{} validation images, {} unlabeled test images'.format(NUM_TRAINING_IMAGES, NUM_VALIDATION_IMAGES, NUM_TEST_IMAGES))

model= get_model()

history = model.fit(get_t_data(dataset), steps_per_epoch=STEPS_PER_EPOCH, 
                    epochs=5,validation_data=get_v_data(dataset),callbacks=callbacks
                   )
cnt_test= count_data_items(filenames_test)
steps      = cnt_test / (BATCH_SIZE) 
#ds_testAug = get_testdataset(files_test, CFG, augment=True, repeat=True, 
#                         labeled=False, return_image_names=False)

probs = model.predict(get_test_dataset(), verbose=1, steps=steps)

#probs = np.stack(probs)
#probs = probs[:,:cnt_test * CFG['tta_steps']]
#probs = np.stack(np.split(probs, CFG['tta_steps'], axis=1), axis=1)
#probs = np.mean(probs, axis=1)
#probs = np.mean(probs,axis=0)
len(probs)
classes=[*range(1, 891, 1)] 
print(classes)
labels=[]
for i in range(len(probs)):
    
    index = np.argsort(probs[i,:])
    label= classes[index[1]]
    p=round(probs[i,index[1]],6)
    #print(p)
    labels.append(f'{label} {p:.6f}')
    
len(labels)
labels
sub= pd.read_csv('../input/landmark-recognition-2020/sample_submission.csv')
sub['landmarks']=labels
sub.head()
sub.to_csv('submission.csv', index=False, float_format='%.6f')
