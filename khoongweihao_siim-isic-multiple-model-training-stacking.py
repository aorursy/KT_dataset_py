!pip install -q efficientnet 
import os

import re



import numpy as np

import pandas as pd

import math



from matplotlib import pyplot as plt



from sklearn import metrics

from sklearn.model_selection import train_test_split



import tensorflow as tf

import tensorflow.keras.layers as L



import efficientnet.tfkeras as efn



from kaggle_datasets import KaggleDatasets
TRAINING = False # set to True if you wanna train ze models!
# Detect hardware, return appropriate distribution strategy

try:

    # TPU detection. No parameters necessary if TPU_NAME environment variable is

    # set: this is always the case on Kaggle.

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
# For tf.dataset

AUTO = tf.data.experimental.AUTOTUNE



# Data access

GCS_PATH = KaggleDatasets().get_gcs_path('siim-isic-melanoma-classification')



# Configuration

EPOCHS = 5

BATCH_SIZE = 8 * strategy.num_replicas_in_sync

IMAGE_SIZE = [1024, 1024]
def append_path(pre):

    return np.vectorize(lambda file: os.path.join(GCS_DS_PATH, pre, file))
sub = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')

TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/tfrecords/train*.tfrec')

TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/tfrecords/test*.tfrec')
print(len(TRAINING_FILENAMES))
print(TRAINING_FILENAMES)
VALIDATION_FILENAMES = TRAINING_FILENAMES[int(0.8*len(TRAINING_FILENAMES)):]

TRAINING_FILENAMES = TRAINING_FILENAMES[:int(0.8*len(TRAINING_FILENAMES))]
len(TRAINING_FILENAMES)
len(VALIDATION_FILENAMES)
def decode_image(image_data):

    image = tf.image.decode_jpeg(image_data, channels=3)

    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range

    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU

    return image



def read_labeled_tfrecord(example):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        #"class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element

        "target": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element

    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    #label = tf.cast(example['class'], tf.int32)

    label = tf.cast(example['target'], tf.int32)

    return image, label # returns a dataset of (image, label) pairs



def read_unlabeled_tfrecord(example):

    UNLABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "image_name": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element

        # class is missing, this competitions's challenge is to predict flower classes for the test dataset

    }

    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    idnum = example['image_name']

    return image, idnum # returns a dataset of image(s)



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

    #image = tf.image.random_saturation(image, 0, 2)

    return image, label   



def get_training_dataset():

    dataset = load_dataset(TRAINING_FILENAMES, labeled=True)

    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)

    dataset = dataset.repeat() # the training dataset must repeat for several epochs

    dataset = dataset.shuffle(2048)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def get_validation_dataset(ordered=False):

    dataset = load_dataset(VALIDATION_FILENAMES, labeled=True, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.cache()

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def get_test_dataset(ordered=False):

    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def count_data_items(filenames):

    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]

    return np.sum(n)



NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)

NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

print('Dataset: {} training images, {} unlabeled test images'.format(NUM_TRAINING_IMAGES, NUM_TEST_IMAGES))
def build_lrfn(lr_start=0.00001, lr_max=0.000075, 

               lr_min=0.000001, lr_rampup_epochs=20, 

               lr_sustain_epochs=0, lr_exp_decay=.8):

    lr_max = lr_max * strategy.num_replicas_in_sync



    def lrfn(epoch):

        if epoch < lr_rampup_epochs:

            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start

        elif epoch < lr_rampup_epochs + lr_sustain_epochs:

            lr = lr_max

        else:

            lr = (lr_max - lr_min) * lr_exp_decay**(epoch - lr_rampup_epochs - lr_sustain_epochs) + lr_min

        return lr

    

    return lrfn
if TRAINING:

    with strategy.scope():

        model = tf.keras.Sequential([

            efn.EfficientNetB7(

                input_shape=(*IMAGE_SIZE, 3),

                weights='imagenet',

                include_top=False

            ),

            L.GlobalAveragePooling2D(),

            L.Dense(1, activation='sigmoid')

        ])

        model.compile(

            optimizer='adam',

            loss = 'binary_crossentropy',

            metrics=['accuracy']

        )

        model.summary()
if TRAINING:

    with strategy.scope():

        model2 = tf.keras.Sequential([

            efn.EfficientNetB0(

                input_shape=(*IMAGE_SIZE, 3),

                weights='imagenet',

                include_top=False

            ),

            L.GlobalAveragePooling2D(),

            L.Dense(1, activation='sigmoid')

        ])

        model2.compile(

            optimizer='adam',

            loss = 'binary_crossentropy',

            metrics=['accuracy']

        )

        model2.summary()
from tensorflow.keras.applications import DenseNet201



if TRAINING:

    with strategy.scope():

        dnet201 = DenseNet201(

            input_shape=(*IMAGE_SIZE, 3),

            weights='imagenet',

            include_top=False

        )

        dnet201.trainable = True



        model3 = tf.keras.Sequential([

            dnet201,

            L.GlobalAveragePooling2D(),

            L.Dense(1, activation='sigmoid')

        ])

        model3.compile(

            optimizer='adam',

            loss = 'binary_crossentropy',

            metrics=['accuracy']

        )



    model3.summary()
lrfn = build_lrfn()

lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)
train_dataset = get_training_dataset()

valid_dataset = get_validation_dataset()
if TRAINING:

    history = model.fit(

        train_dataset, 

        epochs=EPOCHS, 

        callbacks=[lr_schedule],

        steps_per_epoch=STEPS_PER_EPOCH,

        validation_data=valid_dataset

    )

    model.save("efficientnetb7.h5")
if TRAINING:

    history2 = model2.fit(

        train_dataset, 

        epochs=EPOCHS, 

        callbacks=[lr_schedule],

        steps_per_epoch=STEPS_PER_EPOCH,

        validation_data=valid_dataset

    )

    model2.save("efficientnetb0.h5")
#model3.save("densenet201.h5")
def display_training_curves(training, validation, title, subplot):

    """

    Source: https://www.kaggle.com/mgornergoogle/getting-started-with-100-flowers-on-tpu

    """

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
if TRAINING:

    display_training_curves(

        history.history['loss'], 

        history.history['val_loss'], 

        'model 1 loss', 211)

    display_training_curves(

        history.history['accuracy'], 

        history.history['val_accuracy'], 

        'model 1 accuracy', 212)
if TRAINING:

    display_training_curves(

        history2.history['loss'], 

        history2.history['val_loss'], 

        'model 2 loss', 211)

    display_training_curves(

        history2.history['accuracy'], 

        history2.history['val_accuracy'], 

        'model 2 accuracy', 212)
# create copies for each model if you want to

sub1 = sub.copy()

sub2 = sub.copy()
if TRAINING:

    test_ds = get_test_dataset(ordered=True)



    print('Computing predictions...')

    test_images_ds = test_ds.map(lambda image, idnum: image)

    probabilities = model.predict(test_images_ds)

    probabilities2 = model2.predict(test_images_ds)

    print('Generating submission.csv file...')

    test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()

    test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch
if TRAINING:

    pred_df = pd.DataFrame({'image_name': test_ids, 'target': np.concatenate(probabilities)})

    pred_df2 = pd.DataFrame({'image_name': test_ids, 'target': np.concatenate(probabilities2)})

    del sub1['target']

    sub1 = sub1.merge(pred_df, on='image_name')

    sub1.to_csv('submission_efficientnetb7.csv', index=False)

    del sub2['target']

    sub2 = sub2.merge(pred_df2, on='image_name')

    sub2.to_csv('submission_efficientnetb0.csv', index=False)
if TRAINING:

    sub_es = sub1[['image_name']]

    sub_es['target'] = 0.5*sub1['target'] + 0.5*sub2['target']

    sub_es.to_csv('submission_blend.csv', index=False)

    sub_es.head()
import seaborn as sns
sub_path = "../input/siimisic-submission-files"

all_files = os.listdir(sub_path)

all_files
outs = [pd.read_csv(os.path.join(sub_path, f), index_col=0) for f in all_files]

concat_sub = pd.concat(outs, axis=1)

cols = list(map(lambda x: "target" + str(x), range(len(concat_sub.columns))))

concat_sub.columns = cols

concat_sub.reset_index(inplace=True)

concat_sub.head()

ncol = concat_sub.shape[1]
# check correlation

concat_sub.iloc[:,1:ncol].corr()
corr = concat_sub.iloc[:,1:].corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
# get the data fields ready for stacking

concat_sub['target_max'] = concat_sub.iloc[:, 1:ncol].max(axis=1)

concat_sub['target_min'] = concat_sub.iloc[:, 1:ncol].min(axis=1)

concat_sub['target_mean'] = concat_sub.iloc[:, 1:ncol].mean(axis=1)

concat_sub['target_median'] = concat_sub.iloc[:, 1:ncol].median(axis=1)
concat_sub.describe()
cutoff_lo = 0.002

cutoff_hi = 0.11
concat_sub['target'] = concat_sub['target_mean']

concat_sub[['image_name', 'target']].to_csv('submission_mean.csv', 

                                        index=False, float_format='%.6f')
concat_sub['target'] = concat_sub['target_median']

concat_sub[['image_name', 'target']].to_csv('submission_median.csv', 

                                        index=False, float_format='%.6f')
concat_sub['target'] = np.where(np.all(concat_sub.iloc[:,1:ncol] > cutoff_lo, axis=1), 1, 

                                    np.where(np.all(concat_sub.iloc[:,1:ncol] < cutoff_hi, axis=1),

                                             0, concat_sub['target_median']))

concat_sub[['image_name', 'target']].to_csv('submission_pushout_median.csv', 

                                        index=False, float_format='%.6f')
concat_sub['target'] = np.where(np.all(concat_sub.iloc[:,1:ncol] > cutoff_lo, axis=1), 

                                    concat_sub['target_max'], 

                                    np.where(np.all(concat_sub.iloc[:,1:ncol] < cutoff_hi, axis=1),

                                             concat_sub['target_min'], 

                                             concat_sub['target_mean']))

concat_sub[['image_name', 'target']].to_csv('submission_minmax_mean.csv', 

                                        index=False, float_format='%.6f')
concat_sub['target'] = np.where(np.all(concat_sub.iloc[:,1:ncol] > cutoff_lo, axis=1), 

                                    concat_sub['target_max'], 

                                    np.where(np.all(concat_sub.iloc[:,1:ncol] < cutoff_hi, axis=1),

                                             concat_sub['target_min'], 

                                             concat_sub['target_median']))

concat_sub[['image_name', 'target']].to_csv('submission_minmax_median.csv', 

                                        index=False, float_format='%.6f')