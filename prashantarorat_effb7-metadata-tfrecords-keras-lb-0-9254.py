
import tensorflow as tf
import requests
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn
import keras
from keras.layers import *
from keras.applications import *
from keras.models import *
from keras.activations import *
import random
import tensorflow as tf
import cv2
import math
import warnings
warnings.filterwarnings("ignore")

tf.__version__
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
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)
train_filenames = ['gs://kds-63eb66738ab6d0f9b23f4216c213f560824173818288c3d85afd8bb8/train00-2071.tfrec',
 'gs://kds-63eb66738ab6d0f9b23f4216c213f560824173818288c3d85afd8bb8/train01-2071.tfrec',
 'gs://kds-63eb66738ab6d0f9b23f4216c213f560824173818288c3d85afd8bb8/train02-2071.tfrec',
 'gs://kds-63eb66738ab6d0f9b23f4216c213f560824173818288c3d85afd8bb8/train03-2071.tfrec',
 'gs://kds-63eb66738ab6d0f9b23f4216c213f560824173818288c3d85afd8bb8/train04-2071.tfrec',
 'gs://kds-63eb66738ab6d0f9b23f4216c213f560824173818288c3d85afd8bb8/train05-2071.tfrec',
 'gs://kds-63eb66738ab6d0f9b23f4216c213f560824173818288c3d85afd8bb8/train06-2071.tfrec',
 'gs://kds-63eb66738ab6d0f9b23f4216c213f560824173818288c3d85afd8bb8/train07-2071.tfrec',
 'gs://kds-63eb66738ab6d0f9b23f4216c213f560824173818288c3d85afd8bb8/train08-2071.tfrec',
 'gs://kds-63eb66738ab6d0f9b23f4216c213f560824173818288c3d85afd8bb8/train09-2071.tfrec',
 'gs://kds-63eb66738ab6d0f9b23f4216c213f560824173818288c3d85afd8bb8/train10-2071.tfrec',
 'gs://kds-63eb66738ab6d0f9b23f4216c213f560824173818288c3d85afd8bb8/train11-2071.tfrec',
 'gs://kds-63eb66738ab6d0f9b23f4216c213f560824173818288c3d85afd8bb8/train12-2071.tfrec',
 'gs://kds-63eb66738ab6d0f9b23f4216c213f560824173818288c3d85afd8bb8/train13-2071.tfrec',
 'gs://kds-63eb66738ab6d0f9b23f4216c213f560824173818288c3d85afd8bb8/train14-2071.tfrec',
 'gs://kds-63eb66738ab6d0f9b23f4216c213f560824173818288c3d85afd8bb8/train15-2071.tfrec',
 'gs://kds-63eb66738ab6d0f9b23f4216c213f560824173818288c3d85afd8bb8/train16-2071.tfrec',
 'gs://kds-63eb66738ab6d0f9b23f4216c213f560824173818288c3d85afd8bb8/train17-2071.tfrec',
 'gs://kds-63eb66738ab6d0f9b23f4216c213f560824173818288c3d85afd8bb8/train18-2071.tfrec',
 'gs://kds-63eb66738ab6d0f9b23f4216c213f560824173818288c3d85afd8bb8/train19-2071.tfrec',
 'gs://kds-63eb66738ab6d0f9b23f4216c213f560824173818288c3d85afd8bb8/train20-2071.tfrec',
 'gs://kds-63eb66738ab6d0f9b23f4216c213f560824173818288c3d85afd8bb8/train21-2071.tfrec',
 'gs://kds-63eb66738ab6d0f9b23f4216c213f560824173818288c3d85afd8bb8/train22-2071.tfrec',
 'gs://kds-63eb66738ab6d0f9b23f4216c213f560824173818288c3d85afd8bb8/train23-2071.tfrec',
 'gs://kds-63eb66738ab6d0f9b23f4216c213f560824173818288c3d85afd8bb8/train24-2071.tfrec',
 'gs://kds-63eb66738ab6d0f9b23f4216c213f560824173818288c3d85afd8bb8/train25-2071.tfrec',
 'gs://kds-63eb66738ab6d0f9b23f4216c213f560824173818288c3d85afd8bb8/train26-2071.tfrec',
 'gs://kds-63eb66738ab6d0f9b23f4216c213f560824173818288c3d85afd8bb8/train27-2071.tfrec',
 'gs://kds-63eb66738ab6d0f9b23f4216c213f560824173818288c3d85afd8bb8/train28-2071.tfrec',
 'gs://kds-63eb66738ab6d0f9b23f4216c213f560824173818288c3d85afd8bb8/train29-428.tfrec']
from sklearn.model_selection import train_test_split
train_filenames , valid_filenames = train_test_split(train_filenames , test_size=0.15,shuffle=True)
BATCH_SIZE = 8 * strategy.num_replicas_in_sync
IMAGE_SIZE = [512,512]
AUTO = tf.data.experimental.AUTOTUNE
imSize = 512
import re

tab_cols = ['sex','age_approx','anatom_site_general_challenge']

def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU
    image = tf.image.resize(image, [imSize,imSize])
    return image

# https://www.kaggle.com/niteshx2/full-pipeline-dual-input-cnn-model-with-tpus?scriptVersionId=40368875

def read_labeled_tfrecord(example):
    tfrec_format = {
              
        'image': tf.io.FixedLenFeature([], tf.string),
        'image_name': tf.io.FixedLenFeature([], tf.string),
        'patient_id': tf.io.FixedLenFeature([], tf.int64),
        'sex': tf.io.FixedLenFeature([], tf.int64),
        'age_approx': tf.io.FixedLenFeature([], tf.int64),
        'anatom_site_general_challenge': tf.io.FixedLenFeature([], tf.int64),
        'source': tf.io.FixedLenFeature([], tf.int64),
        'target': tf.io.FixedLenFeature([], tf.int64)
        }

    example = tf.io.parse_single_example(example, tfrec_format)
    image = decode_image(example['image'])
    features = [example[k] for k in tab_cols]
    features = tf.stack(features)
    label = tf.cast(example['target'], tf.int32)
    return { 'inp1':image,"inp2":features }, label # returns a dataset of (image, label) pairs

def read_unlabeled_tfrecord(example):
    tfrec_format = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'image_name': tf.io.FixedLenFeature([], tf.string),
        'patient_id': tf.io.FixedLenFeature([], tf.int64),
        'sex': tf.io.FixedLenFeature([], tf.int64),
        'age_approx': tf.io.FixedLenFeature([], tf.int64),
        'anatom_site_general_challenge': tf.io.FixedLenFeature([], tf.int64),
        # 'source': tf.io.FixedLenFeature([], tf.int64),
        # 'target': tf.io.FixedLenFeature([], tf.int64)
      }
    example = tf.io.parse_single_example(example, tfrec_format)
    image = decode_image(example['image'])
    features = [tf.cast(example[k],dtype=tf.float32) for k in tab_cols]
    features = tf.stack(features)
    idnum = example['image_name']
    return {"inp1": image ,"inp2":features }, idnum # returns a dataset of image(s)

def load_dataset(filenames, labeled=True, ordered=False):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # disregarding data order. Order does not matter since we will be shuffling the data anyway.

    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)
    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
    return dataset

def data_augment(image, label):
    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),
    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part
    # of the TPU while the TPU itself is computing gradients.
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    # image = tf.image.adjust_gamma(image,gamma=0.5)
    # image = tf.image.central_crop(image,0.9)
    # image = tf.image.adjust_hue(image,0.3)
    image = tf.image.adjust_saturation(image,0.9)
    return image, label   

def test_augment(image,image_name):
    image = tf.image.adjust_saturation(image,0.9)
    return image, image_name  

def get_training_dataset():
    dataset = load_dataset(train_filenames, labeled=True)
    dataset = dataset.map(lambda dict_ ,label:({"inp1":data_augment(dict_['inp1'],label)[0],"inp2":dict_['inp2']},label), num_parallel_calls=AUTO)
    # dataset = dataset.map(lambda image,label: (tf.image.resize(image, [256, 256]),label), num_parallel_calls=AUTO)
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_val_dataset():
    dataset = load_dataset(valid_filenames, labeled=True)
    dataset = dataset.map(lambda dict_ ,label:({"inp1":test_augment(dict_['inp1'],label)[0],"inp2":dict_['inp2']},label), num_parallel_calls=AUTO)
    # dataset = dataset.map(test_augment , num_parallel_calls=AUTO)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

NUM_TRAINING_IMAGES = count_data_items(train_filenames)
NUM_TEST_IMAGES = count_data_items(valid_filenames)
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
print('Dataset: {} training images, {} labeled validation images'.format(NUM_TRAINING_IMAGES, NUM_TEST_IMAGES))
import math

for dic, label in get_training_dataset().take(3):
    print(dic['inp1'].numpy().shape,dic['inp2'].numpy().shape,label.numpy().shape)
    
print("Training data label examples:", label.numpy())
!pip install efficientnet
!pip install tensorflow_addons
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model , load_model
from tensorflow.keras.utils import plot_model
import efficientnet.tfkeras as efn
import math


EPOCHS = 60

def get_cosine_schedule_with_warmup(lr,num_warmup_steps, num_training_steps, num_cycles=0.75):
  
    def lrfn(epoch):
        if epoch < num_warmup_steps:
            return (float(epoch) / float(max(1, num_warmup_steps))) * lr

        progress = float(epoch - num_warmup_steps ) / float(max(1, num_training_steps - num_warmup_steps))

        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * lr

    return tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)

lr_schedule = get_cosine_schedule_with_warmup(lr=0.00004,num_warmup_steps=4,num_training_steps=45)

with strategy.scope():
    
    input1 = Input(shape=(imSize,imSize,3),name="inp1")
    input2 = Input(shape=(3,),name="inp2")
    output_tab = Dense(24,activation="relu")(input2)
    output_tab = BatchNormalization()(output_tab)

    base_model1 = efn.EfficientNetB7(weights="imagenet",include_top=False,input_shape=(imSize,imSize,3))
    base_model1.trainable = True
    
    base_model1 = base_model1(input1)
    
    output = GlobalMaxPooling2D()(base_model1)
    output = Dropout(0.7)(output)

    output = Dense(256,activation="relu")(output)
    output = Concatenate()([output,output_tab])
    output = Dropout(0.3)(output)

    output = Dense(16,activation="relu")(output)
    output = Dropout(0.2)(output)

    output = Dense(1,activation="sigmoid")(output)
    
    model = Model([input1,input2],output)
    
    model.compile(
        optimizer='adam',
        loss = 'binary_crossentropy',
        metrics=[tf.keras.metrics.AUC()]
    )
    model.summary()
    plot_model(model,to_file="model.png",show_shapes=True)

model.fit(get_training_dataset(),
          epochs=EPOCHS,
          verbose=True,
          steps_per_epoch=NUM_TRAINING_IMAGES // BATCH_SIZE,
          validation_data=get_val_dataset(),
          callbacks=[lr_schedule,
                     tf.keras.callbacks.EarlyStopping(patience=6,verbose=True)])
test_filenames = ['gs://kds-63eb66738ab6d0f9b23f4216c213f560824173818288c3d85afd8bb8/test00-687.tfrec',
 'gs://kds-63eb66738ab6d0f9b23f4216c213f560824173818288c3d85afd8bb8/test01-687.tfrec',
 'gs://kds-63eb66738ab6d0f9b23f4216c213f560824173818288c3d85afd8bb8/test02-687.tfrec',
 'gs://kds-63eb66738ab6d0f9b23f4216c213f560824173818288c3d85afd8bb8/test03-687.tfrec',
 'gs://kds-63eb66738ab6d0f9b23f4216c213f560824173818288c3d85afd8bb8/test04-687.tfrec',
 'gs://kds-63eb66738ab6d0f9b23f4216c213f560824173818288c3d85afd8bb8/test05-687.tfrec',
 'gs://kds-63eb66738ab6d0f9b23f4216c213f560824173818288c3d85afd8bb8/test06-687.tfrec',
 'gs://kds-63eb66738ab6d0f9b23f4216c213f560824173818288c3d85afd8bb8/test07-687.tfrec',
 'gs://kds-63eb66738ab6d0f9b23f4216c213f560824173818288c3d85afd8bb8/test08-687.tfrec',
 'gs://kds-63eb66738ab6d0f9b23f4216c213f560824173818288c3d85afd8bb8/test09-687.tfrec',
 'gs://kds-63eb66738ab6d0f9b23f4216c213f560824173818288c3d85afd8bb8/test10-687.tfrec',
 'gs://kds-63eb66738ab6d0f9b23f4216c213f560824173818288c3d85afd8bb8/test11-687.tfrec',
 'gs://kds-63eb66738ab6d0f9b23f4216c213f560824173818288c3d85afd8bb8/test12-687.tfrec',
 'gs://kds-63eb66738ab6d0f9b23f4216c213f560824173818288c3d85afd8bb8/test13-687.tfrec',
 'gs://kds-63eb66738ab6d0f9b23f4216c213f560824173818288c3d85afd8bb8/test14-687.tfrec',
 'gs://kds-63eb66738ab6d0f9b23f4216c213f560824173818288c3d85afd8bb8/test15-677.tfrec']
num_test_images = count_data_items(test_filenames)
num_test_images
def get_test_dataset(ordered=False):
    dataset = load_dataset(test_filenames, labeled=False,ordered=ordered)
    dataset = dataset.map(lambda dict_ ,label:({"inp1":test_augment(dict_['inp1'],label)[0],"inp2":dict_['inp2']},label), num_parallel_calls=AUTO)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

test_dataset = get_test_dataset(ordered=True)
print('Computing predictions...')
test_images_ds = test_dataset.map(lambda image, idnum: image)
probabilities = model.predict(test_images_ds).flatten()
print(probabilities)


print('Generating submission.csv file...')
test_ids_ds = test_dataset.map(lambda image, idnum: idnum).unbatch()
test_ids = next(iter(test_ids_ds.batch(num_test_images))).numpy().astype('U') # all in one batch
np.savetxt('Eff_7(512x512)_21_epochs_0.9562_val_auc_with_saturation_tabular_manually_stopped.csv', np.rec.fromarrays([test_ids, probabilities]), fmt=['%s', '%f'], delimiter=',', header='image_name,target', comments='')
import seaborn as sns
sns.distplot(probabilities)