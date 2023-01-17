import re

import tensorflow as tf

import numpy as np

from matplotlib import pyplot as plt



print("Tensorflow version " + tf.__version__)

AUTO = tf.data.experimental.AUTOTUNE

from kaggle_datasets import KaggleDatasets
try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print("Running on TPU", tpu.master())

except ValueError:

    tpu = None

    

if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy()



print("REPLICAS: ", strategy.num_replicas_in_sync)
GCS_PATH = KaggleDatasets().get_gcs_path()
EPOCHS = 12

IMAGE_SIZE = [331, 331]



FLOWERS_DATASETS = {

    192: GCS_PATH + '/tfrecords-jpeg-192x192/*.tfrec',

    224: GCS_PATH + '/tfrecords-jpeg-224x224/*.tfrec',

    331: GCS_PATH + '/tfrecords-jpeg-331x331/*.tfrec',

    512: GCS_PATH + '/tfrecords-jpeg-512x512/*.tfrec'

}

CLASSES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']



assert IMAGE_SIZE[0] == IMAGE_SIZE[1], "only square images are supported"

assert IMAGE_SIZE[0] in FLOWERS_DATASETS, "this image size is not supported"



BATCH_SIZE = 16 * strategy.num_replicas_in_sync

LR_START = 0.00001

LR_MAX   = 0.00005 * strategy.num_replicas_in_sync

LR_MIN   = 0.00001

LR_RAMPUP_EPOCHS  = 5

LR_SUSTAIN_EPOCHS = 0

LR_EXP_DECAY      = .8



def lrfn(epoch):

    if epoch < LR_RAMPUP_EPOCHS:

        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START

    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:

        lr = LR_MAX

    else:

        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN

    return lr



lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)



rng = [i for i in range(EPOCHS)]

y = [lrfn(x) for x in rng]

plt.plot(rng, y)

print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))
def dataset_to_numpy_util(dataset, N):

    dataset = dataset.unbatch().batch(N)

    for images, labels in dataset:

        numpy_images = images.numpy()

        numpy_labels = labels.numpy()

        break;

    return numpy_images, numpy_labels



def title_from_label_and_target(label, correct_label):

    label = np.argmax(label, axis=-1)

    correct_label = np.argmax(correcti_label, axis=-1)

    correct = (label == correct_label)

    return "{} [{} {} {}]".format(CLASSES[label], str(correct), ', should be' if not correct else '',

                                  CLASSES[correct_label] if not correct else ''), correct



def display_one_flower(image, title, subplot, red=False):

    plt.subplot(subplot)

    plt.axis('off')

    plt.imshow(image)

    plt.title(title, fontsize=16, color='red' if red else 'black')

    return subplot + 1



def display_9_images_from_dataset(dataset):

    subplot = 331

    plt.figure(figsize=(13,13))

    images, labels = dataset_to_numpy_util(dataset, 9)

    for i, image in enumerate(images):

        title = CLASSES[np.argmax(labels[i], axis=-1)]

        subplot = display_one_flower(image, title, subplot)

        if i >= 8:

            break;



    plt.tight_layout()

    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    plt.show()



def display_9_images_with_predictions(images, predictions, labels):

    subplot = 331

    plt.figure(figsize=(13,13))

    for i, image in enumerate(images):

        title, correct = title_from_label_and_target(predictions[i], labels[i])

        subplot = display_one_flower(image, title, subplot, not correct)

        if i >= 8:

            break;



    plt.tight_layout()

    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    plt.show()



def display_training_curves(training, validation, title, subplot):

    if subplot%10 == 1: # set up the subplots on the first call

        plt.subplots(figsize=(10,10))

        plt.tight_layout()

    ax = plt.subplot(subplot)

    ax.set_facecolor('#F8F8F8')

    ax.plot(training)

    ax.plot(validation)

    ax.set_title('model' + title)

    ax.set_ylabel(title)

    ax.set_xlabel('epoch')

    ax.legend(['train', 'valid.'])
def count_data_items(filenames):

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]

    return np.sum(n)



gcs_pattern = FLOWERS_DATASETS[IMAGE_SIZE[0]]

validation_split = 0.19

filenames = tf.io.gfile.glob(gcs_pattern)

split = len(filenames) - int(len(filenames) * validation_split)

TRAINING_FILENAMES = filenames[:split]

VALIDATION_FILENAMES = filenames[split:]

TRAIN_STEPS = count_data_items(TRAINING_FILENAMES) // BATCH_SIZE # ?



print("TRAINING IMAGES: ", count_data_items(TRAINING_FILENAMES), ", STEPS PRE EPOCH: ", TRAIN_STEPS)

print("VALIDATION_FILENAMES: ", count_data_items(VALIDATION_FILENAMES))



def read_tfrecord(example):

    features = {

        "image": tf.io.FixedLenFeature([], tf.string), 

        "class": tf.io.FixedLenFeature([], tf.int64),

        "one_hot_class": tf.io.VarLenFeature(tf.float32),

    }



    example = tf.io.parse_single_example(example, features)

    image = tf.image.decode_jpeg(example['image'], channels=3)

    image = tf.cast(image, tf.float32) / 255.0

    class_label = tf.cast(example['class'], tf.int32)

    one_hot_class = tf.sparse.to_dense(example['one_hot_class'])

    one_hot_class = tf.reshape(one_hot_class, [5])

    return image, one_hot_class



def force_image_sizes(dataset, image_size):

    reshape_images = lambda image, label: (tf.reshape(image, [*image_size, 3]), label)

    dataset = dataset.map(reshape_images, num_parallel_calls=AUTO) # easier to use parallel compared C++

    return dataset



def load_dataset(filenames):

    ignore_order = tf.data.Options()

    ignore_order.experimental_deterministic = False



    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)

    dataset = dataset.with_options(ignore_order)

    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTO)

    dataset = force_image_sizes(dataset, IMAGE_SIZE)

    return dataset



def data_augment(image, one_hot_class):

    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_saturation(image, 0, 2)

    return image, one_hot_class



def get_training_dataset():

    dataset = load_dataset(TRAINING_FILENAMES)

    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)

    dataset = dataset.repeat()

    dataset = dataset.shuffle(2048) # *

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO)

    return dataset



def get_validation_dataset():

    dataset = load_dataset(VALIDATION_FILENAMES)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO)

    return dataset
training_dataset = get_training_dataset()

validation_dataset = get_validation_dataset()
display_9_images_from_dataset(validation_dataset)