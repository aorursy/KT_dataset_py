!pip install -q efficientnet
# Flower Classification with TPUs 

# Copied from the Getting Started Notebook for this competition

# https://www.kaggle.com/mgornergoogle/getting-started-with-100-flowers-on-tpu/



import math, re, os

import tensorflow as tf

import numpy as np

from matplotlib import pyplot as plt

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix



# My imports

import json

import seaborn as sns

from datetime import datetime

start_time = datetime.now()

print('start time:', start_time)



print("Tensorflow version " + tf.__version__)

AUTO = tf.data.experimental.AUTOTUNE



def my_tf_initialize():

    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:

        try:

            for gpu in gpus:

                tf.config.experimental.set_memory_growth(gpu, True)

                logical_gpus = tf.config.experimental.list_logical_devices('GPU')

                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

        except RuntimeError as e:

            print(e)

    tf.keras.backend.clear_session()

    # make the notebook's output stable across runs

    np.random.seed(42)

    tf.random.set_seed(42)



# Detect hardware, return appropriate distribution strategy

try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None

    my_tf_initialize() # required for my machine



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.



print("REPLICAS: ", strategy.num_replicas_in_sync)



MIXED_PRECISION = True

if MIXED_PRECISION:

    if tpu: 

        # Disable due to errors on Kaggle's TPU Accelerator

        # policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')

        # policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')

        print('Mixed precision not enabled')

    else:

        policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')

        tf.config.optimizer.set_jit(True) # XLA compilation

        tf.keras.mixed_precision.experimental.set_policy(policy)

        print('Mixed precision enabled')

# TPUs read data directly from Google Cloud Storage (GCS). This Kaggle utility will copy the dataset to 

# a GCS bucket co-located with the TPU. If you have multiple datasets attached to the notebook, you can 

# pass the name of a specific dataset to the get_gcs_path function. The name of the dataset is the name 

# of the directory it is mounted in. Use !ls /kaggle/input/ to list attached datasets.



if tpu:

    from kaggle_datasets import KaggleDatasets

    GCS_DS_PATH = KaggleDatasets().get_gcs_path('flower-classification-with-tpus')

else:

    GCS_DS_PATH = os.getcwd() # local machine

    print(GCS_DS_PATH)
# Configuration

if tpu:

    IMAGE_SIZE = [512, 512] # At this size, a GPU will run out of memory. Use the TPU.

                            # For GPU training, please select 224 x 224 px image size.

else:

    #IMAGE_SIZE = [224, 224] # At this size, a GPU will run out of memory. Use the TPU.

    IMAGE_SIZE = [192, 192] # At this size, a GPU will run out of memory. Use the TPU.

                            # For GPU training, please select 224 x 224 px image size.

        

EPOCHS = 20

BATCH_SIZE = 16 * strategy.num_replicas_in_sync

if tpu:

    BATCH_SIZE = 16 * strategy.num_replicas_in_sync

else:

    BATCH_SIZE = 16 * strategy.num_replicas_in_sync

    BATCH_SIZE = 4 * strategy.num_replicas_in_sync



GCS_PATH_SELECT = { # available image sizes

    192: GCS_DS_PATH + '/tfrecords-jpeg-192x192',

    224: GCS_DS_PATH + '/tfrecords-jpeg-224x224',

    331: GCS_DS_PATH + '/tfrecords-jpeg-331x331',

    512: GCS_DS_PATH + '/tfrecords-jpeg-512x512'

}

GCS_PATH = GCS_PATH_SELECT[IMAGE_SIZE[0]]



TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/train/*.tfrec')

VALIDATION_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/val/*.tfrec')

TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/test/*.tfrec') # predictions on this dataset should be submitted for the competition



CLASSES = ['pink primrose',    'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea',     'wild geranium',     'tiger lily',           'moon orchid',              'bird of paradise', 'monkshood',        'globe thistle',         # 00 - 09

           'snapdragon',       "colt's foot",               'king protea',      'spear thistle', 'yellow iris',       'globe-flower',         'purple coneflower',        'peruvian lily',    'balloon flower',   'giant white arum lily', # 10 - 19

           'fire lily',        'pincushion flower',         'fritillary',       'red ginger',    'grape hyacinth',    'corn poppy',           'prince of wales feathers', 'stemless gentian', 'artichoke',        'sweet william',         # 20 - 29

           'carnation',        'garden phlox',              'love in the mist', 'cosmos',        'alpine sea holly',  'ruby-lipped cattleya', 'cape flower',              'great masterwort', 'siam tulip',       'lenten rose',           # 30 - 39

           'barberton daisy',  'daffodil',                  'sword lily',       'poinsettia',    'bolero deep blue',  'wallflower',           'marigold',                 'buttercup',        'daisy',            'common dandelion',      # 40 - 49

           'petunia',          'wild pansy',                'primula',          'sunflower',     'lilac hibiscus',    'bishop of llandaff',   'gaura',                    'geranium',         'orange dahlia',    'pink-yellow dahlia',    # 50 - 59

           'cautleya spicata', 'japanese anemone',          'black-eyed susan', 'silverbush',    'californian poppy', 'osteospermum',         'spring crocus',            'iris',             'windflower',       'tree poppy',            # 60 - 69

           'gazania',          'azalea',                    'water lily',       'rose',          'thorn apple',       'morning glory',        'passion flower',           'lotus',            'toad lily',        'anthurium',             # 70 - 79

           'frangipani',       'clematis',                  'hibiscus',         'columbine',     'desert-rose',       'tree mallow',          'magnolia',                 'cyclamen ',        'watercress',       'canna lily',            # 80 - 89

           'hippeastrum ',     'bee balm',                  'pink quill',       'foxglove',      'bougainvillea',     'camellia',             'mallow',                   'mexican petunia',  'bromelia',         'blanket flower',        # 90 - 99

           'trumpet creeper',  'blackberry lily',           'common tulip',     'wild rose']                                                                                                                                               # 100 - 102
# Visualization utilities

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

        #return CLASSES[label], True

        return CLASSES[np.argmax(label, axis=-1)], True

    correct = (label == correct_label)

    #return "{} [{}{}{}]".format(CLASSES[label], 'OK' if correct else 'NO', u"\u2192" if not correct else '',

    #                            CLASSES[correct_label] if not correct else ''), correct

    return "{} [{}{}{}]".format(CLASSES[np.argmax(label, axis=-1)], 'OK' if correct else 'NO', u"\u2192" if not correct else '',

                                CLASSES[np.argmax(correct_label, axis=-1)] if not correct else ''), correct



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

        title = '' if label is None else CLASSES[np.argmax(label, axis=-1)]

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
# Datasets

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



    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files

    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order

    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)

    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False

    return dataset



def data_to_label(image, label):

    return label



def data_to_categorical(image, label):

    #label = tf.keras.utils.to_categorical(label, len(CLASSES))

    label = tf.one_hot(label, depth=len(CLASSES))

    return image, label   



def data_augment(image, label):

    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),

    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part

    # of the TPU while the TPU itself is computing gradients.

    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_saturation(image, 0, 2)

    return image, label   



import random



def element_mixup(a, b):

    X_a, y_a = a

    X_b, y_b = b

    lambda_1 = tf.random.uniform(shape=[], minval=0., maxval=1.)

    lambda_2 = tf.math.subtract(tf.constant(1.), lambda_1)

    X = tf.math.add(tf.math.scalar_mul(lambda_1, X_a), tf.math.scalar_mul(lambda_2, X_b))

    y = tf.math.add(tf.math.scalar_mul(lambda_1, y_a), tf.math.scalar_mul(lambda_2, y_b))

    #if lambda_1 > 0.5:

    #    y = y_a

    #else:

    #    y = y_b

    return tf.data.Dataset.from_tensors((X, y))

    

#def dataset_mixup(dataset_a):

def dataset_mixup(dataset_a):

    dataset_b = dataset_a.shuffle(2048)

    dataset = tf.data.Dataset.zip((dataset_a, dataset_b))

    dataset = dataset.flat_map(lambda a, b: element_mixup(a, b))

    return dataset   



def get_training_dataset():

    dataset = load_dataset(TRAINING_FILENAMES, labeled=True)

    dataset = dataset.map(data_to_categorical, num_parallel_calls=AUTO)

    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)

    #dataset = dataset_mixup(dataset)

    dataset = dataset.repeat() # the training dataset must repeat for several epochs

    dataset = dataset.shuffle(2048)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def get_training_dataset_mixup():

    dataset = load_dataset(TRAINING_FILENAMES, labeled=True)

    dataset = dataset.map(data_to_categorical, num_parallel_calls=AUTO)

    #dataset = dataset.map(data_augment, num_parallel_calls=AUTO)

    dataset = dataset_mixup(dataset)

    dataset = dataset.repeat() # the training dataset must repeat for several epochs

    dataset = dataset.shuffle(2048)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def get_validation_dataset_labels(ordered=False):

    dataset = load_dataset(VALIDATION_FILENAMES, labeled=True, ordered=ordered)

    dataset = dataset.map(data_to_label, num_parallel_calls=AUTO)

    #dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.cache()

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def get_validation_dataset(ordered=False):

    dataset = load_dataset(VALIDATION_FILENAMES, labeled=True, ordered=ordered)

    dataset = dataset.map(data_to_categorical, num_parallel_calls=AUTO)

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

NUM_VALIDATION_IMAGES = count_data_items(VALIDATION_FILENAMES)

NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

print('Dataset: {} training images, {} validation images, {} unlabeled test images'.format(NUM_TRAINING_IMAGES, NUM_VALIDATION_IMAGES, NUM_TEST_IMAGES))



# Dataset visualizations



# data dump

print("Training data shapes:")

for image, label in get_training_dataset().take(3):

    print(image.numpy().shape, label.numpy().shape)

print("Training data label examples:", label.numpy())

print("Validation data shapes:")

for image, label in get_validation_dataset().take(3):

    print(image.numpy().shape, label.numpy().shape)

print("Validation data label examples:", label.numpy())

print("Test data shapes:")

for image, idnum in get_test_dataset().take(3):

    print(image.numpy().shape, idnum.numpy().shape)

print("Test data IDs:", idnum.numpy().astype('U')) # U=unicode string



# Peek at training data

training_dataset = get_training_dataset()

training_dataset = training_dataset.unbatch().batch(20)

train_batch = iter(training_dataset)



# peer at test data

test_dataset = get_test_dataset()

test_dataset = test_dataset.unbatch().batch(20)

test_batch = iter(test_dataset)

from sklearn.utils import class_weight

labels = get_validation_dataset_labels().take(NUM_VALIDATION_IMAGES)

labels = list(labels.as_numpy_iterator())

sns.countplot(labels)

class_weights = class_weight.compute_class_weight('balanced', np.unique(labels), labels)

#%xdel labels
# run this cell again for next set of images

display_batch_of_images(next(train_batch))
# run this cell again for next set of images

display_batch_of_images(next(test_batch))
early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=5, restore_best_weights=True)



if tpu:

    start_lr = 0.00001

    min_lr = 0.00001

    max_lr = 0.00005 * strategy.num_replicas_in_sync

    rampup_epochs = 5

    sustain_epochs = 0

    exp_decay = .8

else:

    start_lr = 0.00001

    min_lr = 0.00001

    max_lr = 0.0002

    rampup_epochs = 5

    sustain_epochs = 0

    exp_decay = .8



def lrfn(epoch):

    def lr(epoch, start_lr, min_lr, max_lr, rampup_epochs, sustain_epochs, exp_decay):

        if epoch < rampup_epochs:

            lr = (max_lr - start_lr)/rampup_epochs * epoch + start_lr

        elif epoch < rampup_epochs + sustain_epochs:

            lr = max_lr

        else:

            lr = (max_lr - min_lr) * exp_decay**(epoch-rampup_epochs-sustain_epochs) + min_lr

        return lr

    return lr(epoch, start_lr, min_lr, max_lr, rampup_epochs, sustain_epochs, exp_decay)

    

lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda epoch: lrfn(epoch), verbose=True)



rng = [i for i in range(EPOCHS)]

y = [lrfn(x) for x in rng]

plt.plot(rng, [lrfn(x) for x in rng])

print(y[0], y[-1])
# Models

import efficientnet.tfkeras as efntf

from functools import partial



with strategy.scope():

    DefaultConv2D = partial(tf.keras.layers.Conv2D,

                        kernel_size=3, activation='relu', padding="SAME")

    

    simple_cnn_model = tf.keras.models.Sequential([

        DefaultConv2D(filters=64, kernel_size=7, input_shape=[*IMAGE_SIZE, 3]),

        tf.keras.layers.MaxPooling2D(pool_size=2),

        DefaultConv2D(filters=128),

        DefaultConv2D(filters=128),

        tf.keras.layers.MaxPooling2D(pool_size=2),

        DefaultConv2D(filters=256),

        DefaultConv2D(filters=256),

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(len(CLASSES), activation="softmax")

        #tf.keras.layers.Activation(activation="softmax")

    ])

    

    simple_cnn_model.compile(

        optimizer='adam',

        loss = 'categorical_crossentropy',

        metrics=['categorical_accuracy']

    )

    #simple_cnn_model.summary()

        

    ##base_model = tf.keras.applications.densenet.DenseNet121(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])

    ##base_model = tf.keras.applications.densenet.DenseNet169(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])

    #base_model = tf.keras.applications.densenet.DenseNet201(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])

    #base_model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])

    #base_model = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])

    ##base_model = tf.keras.applications.mobilenet.MobileNet(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])

    #base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])

    ##base_model = tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])

    ##base_model = tf.keras.applications.resnet.ResNet101(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])

    ##base_model = tf.keras.applications.resnet.ResNet152(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])

    ##base_model = tf.keras.applications.resnet_v2.ResNet101V2(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])

    #base_model = tf.keras.applications.resnet_v2.ResNet152V2(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])

    #base_model = tf.keras.applications.resnet_v2.ResNet50V2(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])

    #base_model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])

    #base_model = tf.keras.applications.vgg19.VGG19(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])

    #base_model = tf.keras.applications.xception.Xception(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])

    

    enetb7_base_model = efntf.EfficientNetB7(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])

    enetb7_avg = tf.keras.layers.GlobalAveragePooling2D()(enetb7_base_model.output)

    enetb7_output = tf.keras.layers.Dense(len(CLASSES), activation="softmax")(enetb7_avg)

    enetb7_model = tf.keras.Model(inputs=enetb7_base_model.input, outputs=enetb7_output)

    enetb7_model.trainable = True

    

    enetb7_base_model2 = efntf.EfficientNetB7(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])

    enetb7_avg2 = tf.keras.layers.GlobalAveragePooling2D()(enetb7_base_model2.output)

    enetb7_output2 = tf.keras.layers.Dense(len(CLASSES), activation="softmax")(enetb7_avg2)

    enetb7_model2 = tf.keras.Model(inputs=enetb7_base_model2.input, outputs=enetb7_output2)

    enetb7_model2.trainable = True

    

    dense201_base_model = tf.keras.applications.densenet.DenseNet201(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])

    dense201_avg = tf.keras.layers.GlobalAveragePooling2D()(dense201_base_model.output)

    dense201_output = tf.keras.layers.Dense(len(CLASSES), activation="softmax")(dense201_avg)

    dense201_model = tf.keras.Model(inputs=dense201_base_model.input, outputs=dense201_output)

    dense201_model.trainable = True

    

    dense201_base_model2 = tf.keras.applications.densenet.DenseNet201(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])

    dense201_avg2 = tf.keras.layers.GlobalAveragePooling2D()(dense201_base_model2.output)

    dense201_output2 = tf.keras.layers.Dense(len(CLASSES), activation="softmax")(dense201_avg2)

    dense201_model2 = tf.keras.Model(inputs=dense201_base_model2.input, outputs=dense201_output2)

    dense201_model2.trainable = True

    

    enetb7_model.compile(

        optimizer='adam',

        loss = 'categorical_crossentropy',

        metrics=['categorical_accuracy']

    )

    

    enetb7_model2.compile(

        optimizer='adam',

        loss = 'categorical_crossentropy',

        metrics=['categorical_accuracy']

    )

    

    dense201_model.compile(

        optimizer='adam',

        loss = 'categorical_crossentropy',

        metrics=['categorical_accuracy']

    )



    dense201_model2.compile(

        optimizer='adam',

        loss = 'categorical_crossentropy',

        metrics=['categorical_accuracy']

    )



    #model.summary()

model = enetb7_model

#model = enetb7_model2

#model = xcept_model

#model = dense201_model



history = model.fit(get_training_dataset(), 

                    steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS,

                    validation_data=get_validation_dataset(),

                    callbacks=[early_stopping_cb, lr_callback])

 

display_training_curves(history.history['loss'], history.history['val_loss'], 'loss', 211)

display_training_curves(history.history['categorical_accuracy'], history.history['val_categorical_accuracy'], 'accuracy', 212)



# Confusion matrix

cmdataset = get_validation_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and labels, order matters.

images_ds = cmdataset.map(lambda image, label: image)

labels_ds = cmdataset.map(lambda image, label: label).unbatch()

cm_correct_labels = next(iter(labels_ds.batch(NUM_VALIDATION_IMAGES))).numpy() # get everything as one batch

cm_correct_labels = np.argmax(cm_correct_labels, axis=-1)

cm_probabilities = model.predict(images_ds)

cm_predictions = np.argmax(cm_probabilities, axis=-1)

print("Correct   labels: ", cm_correct_labels.shape, cm_correct_labels)

print("Predicted labels: ", cm_predictions.shape, cm_predictions)



cmat = confusion_matrix(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)))

score = f1_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')

precision = precision_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')

recall = recall_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')

cmat = (cmat.T / cmat.sum(axis=1)).T # normalized

display_confusion_matrix(cmat, score, precision, recall)

print('f1 score: {:.3f}, precision: {:.3f}, recall: {:.3f}'.format(score, precision, recall))
#model = enetb7_model

model = enetb7_model2

#model = xcept_model

#model = dense201_model



history = model.fit(get_training_dataset_mixup(), 

                    steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS,

                    validation_data=get_validation_dataset(),

                    callbacks=[early_stopping_cb, lr_callback])

 

display_training_curves(history.history['loss'], history.history['val_loss'], 'loss', 211)

display_training_curves(history.history['categorical_accuracy'], history.history['val_categorical_accuracy'], 'accuracy', 212)



# Confusion matrix

cmdataset = get_validation_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and labels, order matters.

images_ds = cmdataset.map(lambda image, label: image)

labels_ds = cmdataset.map(lambda image, label: label).unbatch()

cm_correct_labels = next(iter(labels_ds.batch(NUM_VALIDATION_IMAGES))).numpy() # get everything as one batch

cm_correct_labels = np.argmax(cm_correct_labels, axis=-1)

cm_probabilities = model.predict(images_ds)

cm_predictions = np.argmax(cm_probabilities, axis=-1)

print("Correct   labels: ", cm_correct_labels.shape, cm_correct_labels)

print("Predicted labels: ", cm_predictions.shape, cm_predictions)



cmat = confusion_matrix(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)))

score = f1_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')

precision = precision_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')

recall = recall_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')

cmat = (cmat.T / cmat.sum(axis=1)).T # normalized

display_confusion_matrix(cmat, score, precision, recall)

print('f1 score: {:.3f}, precision: {:.3f}, recall: {:.3f}'.format(score, precision, recall))
#model = enetb7_model

#model = xcept_model

model = dense201_model



history = model.fit(get_training_dataset(), 

                    steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS,

                    validation_data=get_validation_dataset(),

                    callbacks=[early_stopping_cb, lr_callback])

 

display_training_curves(history.history['loss'], history.history['val_loss'], 'loss', 211)

display_training_curves(history.history['categorical_accuracy'], history.history['val_categorical_accuracy'], 'accuracy', 212)



# Confusion matrix

cmdataset = get_validation_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and labels, order matters.

images_ds = cmdataset.map(lambda image, label: image)

labels_ds = cmdataset.map(lambda image, label: label).unbatch()

cm_correct_labels = next(iter(labels_ds.batch(NUM_VALIDATION_IMAGES))).numpy() # get everything as one batch

cm_correct_labels = np.argmax(cm_correct_labels, axis=-1)

cm_probabilities = model.predict(images_ds)

cm_predictions = np.argmax(cm_probabilities, axis=-1)

print("Correct   labels: ", cm_correct_labels.shape, cm_correct_labels)

print("Predicted labels: ", cm_predictions.shape, cm_predictions)



cmat = confusion_matrix(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)))

score = f1_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')

precision = precision_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')

recall = recall_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')

cmat = (cmat.T / cmat.sum(axis=1)).T # normalized

display_confusion_matrix(cmat, score, precision, recall)

print('f1 score: {:.3f}, precision: {:.3f}, recall: {:.3f}'.format(score, precision, recall))
#model = enetb7_model

#model = xcept_model

model = dense201_model2



history = model.fit(get_training_dataset_mixup(), 

                    steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS,

                    validation_data=get_validation_dataset(),

                    callbacks=[early_stopping_cb, lr_callback])

 

display_training_curves(history.history['loss'], history.history['val_loss'], 'loss', 211)

display_training_curves(history.history['categorical_accuracy'], history.history['val_categorical_accuracy'], 'accuracy', 212)



# Confusion matrix

cmdataset = get_validation_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and labels, order matters.

images_ds = cmdataset.map(lambda image, label: image)

labels_ds = cmdataset.map(lambda image, label: label).unbatch()

cm_correct_labels = next(iter(labels_ds.batch(NUM_VALIDATION_IMAGES))).numpy() # get everything as one batch

cm_correct_labels = np.argmax(cm_correct_labels, axis=-1)

cm_probabilities = model.predict(images_ds)

cm_predictions = np.argmax(cm_probabilities, axis=-1)

print("Correct   labels: ", cm_correct_labels.shape, cm_correct_labels)

print("Predicted labels: ", cm_predictions.shape, cm_predictions)



cmat = confusion_matrix(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)))

score = f1_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')

precision = precision_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')

recall = recall_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')

cmat = (cmat.T / cmat.sum(axis=1)).T # normalized

display_confusion_matrix(cmat, score, precision, recall)

print('f1 score: {:.3f}, precision: {:.3f}, recall: {:.3f}'.format(score, precision, recall))


# Confusion matrix

cmdataset = get_validation_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and labels, order matters.

images_ds = cmdataset.map(lambda image, label: image)

labels_ds = cmdataset.map(lambda image, label: label).unbatch()

cm_correct_labels = next(iter(labels_ds.batch(NUM_VALIDATION_IMAGES))).numpy() # get everything as one batch

cm_correct_labels = np.argmax(cm_correct_labels, axis=-1)



enetb7_modelcm_probabilities = enetb7_model.predict(images_ds)

enetb7_model2cm_probabilities = enetb7_model2.predict(images_ds)

dense201_modelcm_probabilities = dense201_model.predict(images_ds)

dense201_model2cm_probabilities = dense201_model2.predict(images_ds)

cm_probabilities = enetb7_modelcm_probabilities + enetb7_model2cm_probabilities + dense201_modelcm_probabilities + dense201_model2cm_probabilities



#cm_probabilities = model.predict(images_ds)

cm_predictions = np.argmax(cm_probabilities, axis=-1)

print("Correct   labels: ", cm_correct_labels.shape, cm_correct_labels)

print("Predicted labels: ", cm_predictions.shape, cm_predictions)



cmat = confusion_matrix(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)))

score = f1_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')

precision = precision_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')

recall = recall_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')

cmat = (cmat.T / cmat.sum(axis=1)).T # normalized

display_confusion_matrix(cmat, score, precision, recall)

print('f1 score: {:.3f}, precision: {:.3f}, recall: {:.3f}'.format(score, precision, recall))



print('run time: ', datetime.now() - start_time)
# Predictions

test_ds = get_test_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and ids, order matters.



print('Computing predictions...')

test_images_ds = test_ds.map(lambda image, idnum: image)



#probabilities = model.predict(test_images_ds)

enetb7_modelcm_probabilities = enetb7_model.predict(test_images_ds)

enetb7_model2cm_probabilities = enetb7_model2.predict(test_images_ds)

#xcept_modelcm_probabilities = xcept_model.predict(test_images_ds)

dense201_modelcm_probabilities = dense201_model.predict(test_images_ds)

dense201_model2cm_probabilities = dense201_model2.predict(test_images_ds)

probabilities = enetb7_modelcm_probabilities + enetb7_model2cm_probabilities + dense201_modelcm_probabilities + dense201_model2cm_probabilities



predictions = np.argmax(probabilities, axis=-1)

print(predictions)



print('Generating submission.csv file...')

test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()

test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch

np.savetxt('submission.csv', np.rec.fromarrays([test_ids, predictions]), fmt=['%s', '%d'], delimiter=',', header='id,label', comments='')

#!head submission.csv

with open('submission.csv') as myfile:

    head = [next(myfile) for x in range(10)]

print(head)