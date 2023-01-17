import tensorflow as tf

from kaggle_datasets import KaggleDatasets

import math, re, os, random

import numpy as np

from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import StratifiedKFold

from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.optimizers import Adam



print("Tensorflow version " + tf.__version__)
# Detect hardware, return appropriate distribution strategy

try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.



print("REPLICAS: ", strategy.num_replicas_in_sync)
GCS_DS_PATH = KaggleDatasets().get_gcs_path() # you can list the bucket with "!gsutil ls $GCS_DS_PATH"
IMAGE_SIZE = [224, 224] # at this size, a GPU will run out of memory. Use the TPU

EPOCHS = 70

BATCH_SIZE = 16 * strategy.num_replicas_in_sync

SEED = 42

NUM_TRAINING_IMAGES = 12753

NUM_TEST_IMAGES = 7382

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
def random_blockout(img, sl = 0.1, sh = 0.2, rl = 0.4):

    p = random.random()

    if p >= 0.25:

        w, h, c = IMAGE_SIZE[0], IMAGE_SIZE[1], 3

        origin_area = tf.cast(h * w, tf.float32)



        e_size_l = tf.cast(tf.round(tf.sqrt(origin_area * sl * rl)), tf.int32)

        e_size_h = tf.cast(tf.round(tf.sqrt(origin_area * sh / rl)), tf.int32)



        e_height_h = tf.minimum(e_size_h, h)

        e_width_h = tf.minimum(e_size_h, w)



        erase_height = tf.random.uniform(shape = [], minval = e_size_l, maxval = e_height_h, dtype = tf.int32)

        erase_width = tf.random.uniform(shape = [], minval = e_size_l, maxval = e_width_h, dtype = tf.int32)



        erase_area = tf.zeros(shape = [erase_height, erase_width, c])

        erase_area = tf.cast(erase_area, tf.uint8)



        pad_h = h - erase_height

        pad_top = tf.random.uniform(shape = [], minval = 0, maxval = pad_h, dtype = tf.int32)

        pad_bottom = pad_h - pad_top



        pad_w = w - erase_width

        pad_left = tf.random.uniform(shape = [], minval = 0, maxval = pad_w, dtype = tf.int32)

        pad_right = pad_w - pad_left



        erase_mask = tf.pad([erase_area], [[0,0],[pad_top, pad_bottom], [pad_left, pad_right], [0,0]], constant_values=1)

        erase_mask = tf.squeeze(erase_mask, axis = 0)

        erased_img = tf.multiply(tf.cast(img,tf.float32), tf.cast(erase_mask, tf.float32))



        return tf.cast(erased_img, img.dtype)

    else:

        return tf.cast(img, img.dtype)
def decode_image(image_data):

    image = tf.image.decode_jpeg(image_data, channels=3)

    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range

    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU

    return image



def read_labeled_tfrecord(example):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element

    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    label = tf.cast(example['class'], tf.int32)

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

    return image, idnum # returns a dataset of image(s)



def load_dataset(filenames, labeled=True, ordered=False):

    # Read from TFRecords. For optimal performance, reading from multiple files at once and

    # disregarding data order. Order does not matter since we will be shuffling the data anyway.



    ignore_order = tf.data.Options()

    if not ordered:

        ignore_order.experimental_deterministic = False # disable order, increase speed



    dataset = tf.data.TFRecordDataset(filenames) # automatically interleaves reads from multiple files

    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order

    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord)

    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False

    return dataset



def data_augment(image, label):

    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),

    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part

    # of the TPU while the TPU itself is computing gradients.



    image = tf.image.random_flip_left_right(image, seed=SEED)

    image = random_blockout(image)

    return image, label



def get_training_dataset():

    dataset = load_dataset(tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords-jpeg-224x224/train/*.tfrec'), labeled=True)

    dataset = dataset.map(data_augment)

    dataset = dataset.repeat() # the training dataset must repeat for several epochs

    dataset = dataset.shuffle(2048)

    dataset = dataset.batch(BATCH_SIZE)

    return dataset



def get_validation_dataset():

    dataset = load_dataset(tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords-jpeg-224x224/val/*.tfrec'), labeled=True, ordered=False)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.cache()

    return dataset



def get_test_dataset(ordered=False):

    dataset = load_dataset(tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords-jpeg-224x224/test/*.tfrec'), labeled=False, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    return dataset

!pip install -q efficientnet

import efficientnet.tfkeras as efn
learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_sparse_categorical_accuracy', patience = 3, verbose = 1, 

                                           factor = 0.2, min_lr = 0.00001)



optimizer = Adam(lr = .0001, beta_1 = .9, beta_2 = .999, epsilon = None, decay = .0, amsgrad = False)
training_dataset = get_training_dataset()

validation_dataset = get_validation_dataset()



with strategy.scope():    

    pretrained_model =  efn.EfficientNetB0(weights = 'imagenet', include_top = False , input_shape = [*IMAGE_SIZE, 3])

    pretrained_model.trainable = False # tramsfer learning

    

    model = tf.keras.Sequential([

        pretrained_model,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(104, activation='softmax')

    ])

        

model.compile(

    optimizer = optimizer,

    loss = 'sparse_categorical_crossentropy',

    metrics=['sparse_categorical_accuracy']

)



historical = model.fit(training_dataset, 

          steps_per_epoch=STEPS_PER_EPOCH, 

          epochs = EPOCHS,

          callbacks = [learning_rate_reduction],

          validation_data=validation_dataset)
test_ds = get_test_dataset(ordered = True) # since we are splitting the dataset and iterating separately on images and ids, order matters.



print('Computing predictions...')

test_images_ds = test_ds.map(lambda image, idnum: image)

probabilities = model.predict(test_images_ds)

predictions = np.argmax(probabilities, axis=-1)

print(predictions)



print('Generating submission.csv file...')

test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()

test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch

np.savetxt('submission.csv', np.rec.fromarrays([test_ids, predictions]), fmt = ['%s', '%d'], delimiter = ',', header = 'id,label', comments = '')