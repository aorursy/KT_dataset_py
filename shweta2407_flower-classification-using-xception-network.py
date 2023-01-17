import pandas as pd

import numpy as np

import torch

from torch.utils.data import Dataset, DataLoader

import os, cv2, re

import tensorflow as tf

import matplotlib.pyplot as plt

import seaborn as sns

from kaggle_datasets import KaggleDatasets

from keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout, BatchNormalization

from keras.applications.xception import Xception

from keras.models import Model, load_model

from keras.callbacks import ModelCheckpoint
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

    

strategy.num_replicas_in_sync
Image_size=[512,512]

batch_size=16 * strategy.num_replicas_in_sync



AUTO = tf.data.experimental.AUTOTUNE
GCS_DS_PATH = KaggleDatasets().get_gcs_path('tpu-getting-started')



GCS_PATH_SELECT = { 

    192: GCS_DS_PATH + '/tfrecords-jpeg-192x192',

    224: GCS_DS_PATH + '/tfrecords-jpeg-224x224',

    331: GCS_DS_PATH + '/tfrecords-jpeg-331x331',

    512: GCS_DS_PATH + '/tfrecords-jpeg-512x512'

}



GCS_PATH = GCS_PATH_SELECT[Image_size[0]]



train_files = tf.io.gfile.glob(GCS_PATH + '/train/*.tfrec') 

val_files = tf.io.gfile.glob(GCS_PATH + '/val/*.tfrec')

test_files = tf.io.gfile.glob(GCS_PATH + '/test/*.tfrec')
def data_augment(image, label):

    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),

    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part

    # of the TPU while the TPU itself is computing gradients.

    image = tf.image.random_flip_left_right(image)

    return image, label 



def decode_image(image_data):

    image = tf.image.decode_jpeg(image_data, channels=3)

    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range

    image = tf.reshape(image, [*Image_size, 3]) # explicit size needed for TPU

    return image



def read_labeled_tfrecord(example):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element

    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    label = tf.cast(example['class'], tf.int32)

    return image, label



def read_unlabeled_tfrecord(example):

    UNLABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "id": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element

        # class is missing, this competitions's challenge is to predict flower classes for the test dataset

    }

    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    idnum = example['id']

    return image, idnum



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



def get_training_dataset(train_files):

    dataset = load_dataset(train_files, labeled=True)

    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)

    dataset = dataset.repeat() # the training dataset must repeat for several epochs

    dataset = dataset.shuffle(2048)

    dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def get_validation_dataset(val_files, ordered=False):

    dataset = load_dataset(val_files, labeled=True, ordered=False)

    dataset = dataset.batch(batch_size)

    dataset = dataset.cache()

    dataset = dataset.prefetch(AUTO) 

    return dataset

    

def get_test_dataset(test_files, ordered=False):

    dataset = load_dataset(test_files , labeled=False, ordered=ordered)

    dataset = dataset.batch(batch_size)

    dataset = dataset.cache() 

    dataset = dataset.prefetch(AUTO)

    return dataset



def count_data_items(filenames):

    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]

    return np.sum(n)
train_dataset = get_training_dataset(train_files)

validation_dataset = get_validation_dataset(val_files)

test_dataset = get_test_dataset(test_files)



print(validation_dataset)

print(test_dataset)
NUM_TRAINING_IMAGES = count_data_items(train_files)

NUM_VALIDATION_IMAGES = count_data_items(val_files)

NUM_TEST_IMAGES = count_data_items(test_files)



print('Training_size=',NUM_TRAINING_IMAGES  ,'Validation size=',NUM_VALIDATION_IMAGES , 'Test size=',NUM_TEST_IMAGES )
with strategy.scope():

    image_input = Input(shape=(*Image_size,3))

    base_model = Xception(include_top=False, input_tensor=image_input, weights='imagenet')

    x = base_model.layers[-1].output

    x = BatchNormalization()(x)

    x = GlobalAveragePooling2D()(x)

#     x = Dropout(0.25)(x)

    output = Dense(104, activation='softmax')(x)



    model = Model(inputs = image_input, outputs = output )

    model.compile(loss= 'sparse_categorical_crossentropy', optimizer='adam', metrics =['accuracy'])



# model.summary()
# define checkpoint callback

filepath = './model-ep{epoch:02d}-val_acc{val_accuracy:.3f}.h5'



callbacks = [ ModelCheckpoint(filepath= filepath, save_best_only=True, monitor='val_accuracy', mode='max') ]
# epochs = 30

# STEPS_FOR_EPOCH = NUM_TRAINING_IMAGES// batch_size



# history = model.fit(train_dataset, 

#                     steps_per_epoch=STEPS_FOR_EPOCH, 

#                     epochs = epochs, 

#                     batch_size = batch_size,

#                     callbacks = callbacks,

#                     validation_data=validation_dataset)
# plt.figure(figsize=(20,6))

# titles = ['LOSS', 'ACCURACY']

# ylabel = ['LOSSES','ACCURACY']

# par = ['loss', 'accuracy']

# color = ['Red','Blue','Orange','Green']

# j =0

# for i in range(2):

#     plt.subplot(1,2, i+1)

#     plt.plot(history.history[par[i]], c= color[j])

#     plt.plot(history.history['val_'+ par[i] ], c= color[j+1])

#     plt.title('MODEL '+ titles[i])

#     plt.xlabel('EPOCHS')

#     plt.ylabel(ylabel[i])

#     j = j+2

#     plt.legend(['training', 'validation'])

# plt.show()
model = load_model('../input/flower-classification-model/model-ep30-val_acc0.907.h5')
test_ds = get_test_dataset(test_files, ordered=True) 



test_images_ds = test_ds.map(lambda image, idnum: image)

probabilities = model.predict(test_images_ds)

predictions = np.argmax(probabilities, axis=-1)

print(predictions)
test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()

test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch

np.savetxt('submission.csv', np.rec.fromarrays([test_ids, np.array(predictions)]), fmt=['%s', '%d'], delimiter=',', header='id,label', comments='')
pd.read_csv('./submission.csv')