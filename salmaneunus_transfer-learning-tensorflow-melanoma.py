import re

import os

import numpy as np

import pandas as pd

import tensorflow as tf

from functools import partial

from kaggle_datasets import KaggleDatasets

from sklearn.model_selection import train_test_split

import tempfile

import matplotlib.pyplot as plt



try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Device:', tpu.master())

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

except:

    strategy = tf.distribute.get_strategy()

print('Number of replicas:', strategy.num_replicas_in_sync)

    

print(tf.__version__)
AUTOTUNE = tf.data.experimental.AUTOTUNE

GCS_PATH = KaggleDatasets().get_gcs_path()

BATCH_SIZE = 16 * strategy.num_replicas_in_sync

IMAGE_SIZE = [1024, 1024]

IMAGE_RESIZE = [256, 256]
TRAINING_FILENAMES, VALID_FILENAMES = train_test_split(

    tf.io.gfile.glob(GCS_PATH + '/tfrecords/train*.tfrec'),

    test_size=0.1, random_state=5

)

TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/tfrecords/test*.tfrec')

print('Train TFRecord Files:', len(TRAINING_FILENAMES))

print('Validation TFRecord Files:', len(VALID_FILENAMES))

print('Test TFRecord Files:', len(TEST_FILENAMES))
def decode_image(image):

    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.cast(image, tf.float32) / 255.0

    image = tf.reshape(image, [*IMAGE_SIZE, 3])

    return image
def read_tfrecord(example, labeled):

    tfrecord_format = {

        "image": tf.io.FixedLenFeature([], tf.string),

        "target": tf.io.FixedLenFeature([], tf.int64)

    } if labeled else {

        "image": tf.io.FixedLenFeature([], tf.string),

        "image_name": tf.io.FixedLenFeature([], tf.string)

    }

    example = tf.io.parse_single_example(example, tfrecord_format)

    image = decode_image(example['image'])

    if labeled:

        label = tf.cast(example['target'], tf.int32)

        return image, label

    idnum = example['image_name']

    return image, idnum
def load_dataset(filenames, labeled=True, ordered=False):

    ignore_order = tf.data.Options()

    if not ordered:

        ignore_order.experimental_deterministic = False # disable order, increase speed

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE) # automatically interleaves reads from multiple files

    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order

    dataset = dataset.map(partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE)

    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False

    return dataset
def augmentation_pipeline(image, label):

    image = tf.image.random_flip_left_right(image)

    image = tf.image.resize(image, IMAGE_RESIZE)

    return image, label
def get_training_dataset():

    dataset = load_dataset(TRAINING_FILENAMES, labeled=True)

    dataset = dataset.map(augmentation_pipeline, num_parallel_calls=AUTOTUNE)

    dataset = dataset.repeat()

    dataset = dataset.shuffle(2048)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTOTUNE)

    return dataset
def get_validation_dataset(ordered=False):

    dataset = load_dataset(VALID_FILENAMES, labeled=True, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.cache()

    dataset = dataset.prefetch(AUTOTUNE)

    return dataset
def get_test_dataset(ordered=False):

    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTOTUNE)

    return dataset
def count_data_items(filenames):

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]

    return np.sum(n)
NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)

NUM_VALIDATION_IMAGES = count_data_items(VALID_FILENAMES)

NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

print(

    'Dataset: {} training images, {} validation images, {} unlabeled test images'.format(

        NUM_TRAINING_IMAGES, NUM_VALIDATION_IMAGES, NUM_TEST_IMAGES

    )

)
def exponential_decay(lr0, s):

    def exponential_decay_fn(epoch):

        return lr0 * 0.1 **(epoch / s)

    return exponential_decay_fn



exponential_decay_fn = exponential_decay(0.01, 20)



lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)
train_csv = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')

test_csv = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')
total_img = train_csv['target'].size



malignant = np.count_nonzero(train_csv['target'])

benign = total_img - malignant



print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(

    total_img, malignant, 100 * malignant / total_img))
train_dataset = get_training_dataset()

valid_dataset = get_validation_dataset()
image_batch, label_batch = next(iter(train_dataset))
def show_batch(image_batch, label_batch):

    plt.figure(figsize=(20,10))

    for n in range(25):

        ax = plt.subplot(5,5,n+1)

        plt.imshow(image_batch[n])

        if label_batch[n]:

            plt.title("MALIGNANT")

        else:

            plt.title("BENIGN")

        plt.axis("off")
show_batch(image_batch.numpy(), label_batch.numpy())
def make_model(output_bias = None, metrics = None):    

    if output_bias is not None:

        output_bias = tf.keras.initializers.Constant(output_bias)

        

    base_model = tf.keras.applications.InceptionV3(input_shape=(*IMAGE_RESIZE, 3),

                                                include_top=False,

                                                weights='imagenet')

    

    base_model.trainable = False

    

    model = tf.keras.Sequential([

    base_model,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(8, activation='relu'),

        tf.keras.layers.Dense(1, activation='sigmoid',

                              bias_initializer=output_bias)

    ])

    

    model.compile(optimizer='adam',

                  loss='binary_crossentropy',

                  metrics=metrics)

    

    return model
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

VALID_STEPS = NUM_VALIDATION_IMAGES // BATCH_SIZE
initial_bias = np.log([malignant/benign])

initial_bias
weight_for_0 = (1 / benign)*(total_img)/2.0 

weight_for_1 = (1 / malignant)*(total_img)/2.0



class_weight = {0: weight_for_0, 1: weight_for_1}



print('Weight for class 0: {:.2f}'.format(weight_for_0))

print('Weight for class 1: {:.2f}'.format(weight_for_1))
with strategy.scope():

    model = make_model(output_bias = initial_bias, metrics=tf.keras.metrics.AUC(name='auc'))
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("melanoma_model.h5",

                                                    save_best_only=True)



early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=20,

                                                     restore_best_weights=True)
history = model.fit(

    train_dataset, epochs=50,

    steps_per_epoch=STEPS_PER_EPOCH,

    validation_data=valid_dataset,

    validation_steps=VALID_STEPS,

    callbacks=[checkpoint_cb, lr_scheduler, early_stopping_cb],

    class_weight=class_weight

)
test_ds = get_test_dataset(ordered=True)



print('Computing predictions...')

test_images_ds = test_ds.map(lambda image, idnum: image)

probabilities = model.predict(test_images_ds)
sub = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')

sub.head()
print('Generating submission.csv file...')

test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()

test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U')
pred_df = pd.DataFrame({'image_name': test_ids, 'target': np.concatenate(probabilities)})

pred_df.head()
del sub['target']

sub = sub.merge(pred_df, on='image_name')

sub.to_csv('submission.csv', index=False)

sub.head()