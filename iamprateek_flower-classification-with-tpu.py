# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input/flower-classification-with-tpus/'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# in this version I will try learning rate mentioned in below notebook

# https://www.kaggle.com/mgornergoogle/five-flowers-with-keras-and-xception-on-tpu

# Adding validation skip code from below kernel

# https://www.kaggle.com/wrrosa/tpu-enet-b7-densenet

# In this kernel I will try more data augmentation techniques
# fix random seed for reproducibility

seed = 7

np.random.seed(seed)
!pip install -q efficientnet



import efficientnet.tfkeras as efn
import tensorflow as tf

print("Tensorflow version " + tf.__version__)

AUTO = tf.data.experimental.AUTOTUNE # used in tf.data.Dataset API
import tensorflow.keras.backend as K
from tensorflow.keras.applications import InceptionResNetV2

from tensorflow.keras.applications import ResNet152V2

from tensorflow.keras.applications import DenseNet201
# Detect hardware, return appropriate distribution strategy

try:

    # detect and init the TPU

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    # instantiate a distribution strategy

    tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    # default distribution strategy in Tensorflow. Works on CPU and single GPU.

    tpu_strategy = tf.distribute.get_strategy()



print("No. Of REPLICAS: ", tpu_strategy.num_replicas_in_sync)
IMAGE_SIZE = [224, 224]

EPOCHS = 15

BATCH_SIZE = 16 * tpu_strategy.num_replicas_in_sync

HEIGHT = 224

WIDTH = 224

CHANNELS = 3

#FOLDS = 5



# As TPUs require access to the GCS path

from kaggle_datasets import KaggleDatasets

DATASET_PATH = KaggleDatasets().get_gcs_path()



DATASET_PATH_SELECT = { # available image sizes

    192: DATASET_PATH + '/tfrecords-jpeg-192x192',

    224: DATASET_PATH + '/tfrecords-jpeg-224x224',

    331: DATASET_PATH + '/tfrecords-jpeg-331x331',

    512: DATASET_PATH + '/tfrecords-jpeg-512x512'

}



SELECTED_DATASET = DATASET_PATH_SELECT[IMAGE_SIZE[0]]



print("SELECTED_DATASET: ", SELECTED_DATASET)



TRAINING_FILENAMES = tf.io.gfile.glob(SELECTED_DATASET + '/train/*.tfrec')

VALIDATION_FILENAMES = tf.io.gfile.glob(SELECTED_DATASET + '/val/*.tfrec')

TEST_FILENAMES = tf.io.gfile.glob(SELECTED_DATASET + '/test/*.tfrec')
# watch out for overfitting!

SKIP_VALIDATION = True

if SKIP_VALIDATION:

    TRAINING_FILENAMES = TRAINING_FILENAMES + VALIDATION_FILENAMES

    

VALIDATION_MISMATCHES_IDS = ['861282b96','df1fd14b4','b402b6acd','861282b96','741999f79','4dab7fa08','6423cd23e','617a30d60','87d91aefb','2023d3cac','5f56bcb7f','4571b9509',

'f4ec48685','f9c50db87','96379ff01','28594d9ce','6a3a28a06','fbd61ef17','55a883e16','83a80db99','9ee42218f','b5fb20185',

'868bf8b0c','d0caf04b9','ef945a176','9b8f2f5bd','f8da3867d','0bf0b39b3','bab3ef1f5','293c37e25','f739f3e83','5253af526',

'f27f9a100','077803f97','b4becad84']
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

           'trumpet creeper',  'blackberry lily',           'common tulip',     'wild rose']
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
def data_augment(image, label):

    p_spatial = tf.random.uniform([1], minval=0, maxval=1, dtype='float32', seed=seed)

    p_spatial2 = tf.random.uniform([1], minval=0, maxval=1, dtype='float32', seed=seed)

    p_pixel = tf.random.uniform([1], minval=0, maxval=1, dtype='float32', seed=seed)

    p_crop = tf.random.uniform([1], minval=0, maxval=1, dtype='float32', seed=seed)

    

    # Spatial-level transforms

    if p_spatial >= .2: # flips

        image = tf.image.random_flip_left_right(image, seed=seed)

        image = tf.image.random_flip_up_down(image, seed=seed)

        

    if p_crop >= .7: # crops

        if p_crop >= .95:

            image = tf.image.random_crop(image, size=[int(HEIGHT*.6), int(WIDTH*.6), CHANNELS], seed=seed)

        elif p_crop >= .85:

            image = tf.image.random_crop(image, size=[int(HEIGHT*.7), int(WIDTH*.7), CHANNELS], seed=seed)

        elif p_crop >= .8:

            image = tf.image.random_crop(image, size=[int(HEIGHT*.8), int(WIDTH*.8), CHANNELS], seed=seed)

        else:

            image = tf.image.random_crop(image, size=[int(HEIGHT*.9), int(WIDTH*.9), CHANNELS], seed=seed)

        image = tf.image.resize(image, size=[HEIGHT, WIDTH])

        

    if p_spatial2 >= .6: # @cdeotte's functions

        if p_spatial2 >= .9:

            image = transform_rotation(image)

        elif p_spatial2 >= .8:

            image = transform_zoom(image)

        elif p_spatial2 >= .7:

            image = transform_shift(image)

        else:

            image = transform_shear(image)

        

    # Pixel-level transforms

    if p_pixel >= .4: # pixel transformations

        if p_pixel >= .85:

            image = tf.image.random_saturation(image, lower=0, upper=2, seed=seed)

        elif p_pixel >= .65:

            image = tf.image.random_contrast(image, lower=.8, upper=2, seed=seed)

        elif p_pixel >= .5:

            image = tf.image.random_brightness(image, max_delta=.2, seed=seed)

        else:

            image = tf.image.adjust_gamma(image, gamma=.6)



    return image, label
def transform_rotation(image):

    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]

    # output - image randomly rotated

    DIM = HEIGHT

    XDIM = DIM%2 #fix for size 331

    

    rotation = 15. * tf.random.normal([1],dtype='float32')

    # CONVERT DEGREES TO RADIANS

    rotation = math.pi * rotation / 180.

    

    # ROTATION MATRIX

    c1 = tf.math.cos(rotation)

    s1 = tf.math.sin(rotation)

    one = tf.constant([1],dtype='float32')

    zero = tf.constant([0],dtype='float32')

    rotation_matrix = tf.reshape( tf.concat([c1,s1,zero, -s1,c1,zero, zero,zero,one],axis=0),[3,3] )



    # LIST DESTINATION PIXEL INDICES

    x = tf.repeat( tf.range(DIM//2,-DIM//2,-1), DIM )

    y = tf.tile( tf.range(-DIM//2,DIM//2),[DIM] )

    z = tf.ones([DIM*DIM],dtype='int32')

    idx = tf.stack( [x,y,z] )

    

    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS

    idx2 = K.dot(rotation_matrix,tf.cast(idx,dtype='float32'))

    idx2 = K.cast(idx2,dtype='int32')

    idx2 = K.clip(idx2,-DIM//2+XDIM+1,DIM//2)

    

    # FIND ORIGIN PIXEL VALUES 

    idx3 = tf.stack( [DIM//2-idx2[0,], DIM//2-1+idx2[1,]] )

    d = tf.gather_nd(image, tf.transpose(idx3))

        

    return tf.reshape(d,[DIM,DIM,3])



def transform_shear(image):

    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]

    # output - image randomly sheared

    DIM = HEIGHT

    XDIM = DIM%2 #fix for size 331

    

    shear = 5. * tf.random.normal([1],dtype='float32')

    shear = math.pi * shear / 180.

        

    # SHEAR MATRIX

    one = tf.constant([1],dtype='float32')

    zero = tf.constant([0],dtype='float32')

    c2 = tf.math.cos(shear)

    s2 = tf.math.sin(shear)

    shear_matrix = tf.reshape( tf.concat([one,s2,zero, zero,c2,zero, zero,zero,one],axis=0),[3,3] )    



    # LIST DESTINATION PIXEL INDICES

    x = tf.repeat( tf.range(DIM//2,-DIM//2,-1), DIM )

    y = tf.tile( tf.range(-DIM//2,DIM//2),[DIM] )

    z = tf.ones([DIM*DIM],dtype='int32')

    idx = tf.stack( [x,y,z] )

    

    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS

    idx2 = K.dot(shear_matrix,tf.cast(idx,dtype='float32'))

    idx2 = K.cast(idx2,dtype='int32')

    idx2 = K.clip(idx2,-DIM//2+XDIM+1,DIM//2)

    

    # FIND ORIGIN PIXEL VALUES 

    idx3 = tf.stack( [DIM//2-idx2[0,], DIM//2-1+idx2[1,]] )

    d = tf.gather_nd(image, tf.transpose(idx3))

        

    return tf.reshape(d,[DIM,DIM,3])



def transform_shift(image):

    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]

    # output - image randomly shifted

    DIM = HEIGHT

    XDIM = DIM%2 #fix for size 331

    

    height_shift = 16. * tf.random.normal([1],dtype='float32') 

    width_shift = 16. * tf.random.normal([1],dtype='float32') 

    one = tf.constant([1],dtype='float32')

    zero = tf.constant([0],dtype='float32')

        

    # SHIFT MATRIX

    shift_matrix = tf.reshape( tf.concat([one,zero,height_shift, zero,one,width_shift, zero,zero,one],axis=0),[3,3] )



    # LIST DESTINATION PIXEL INDICES

    x = tf.repeat( tf.range(DIM//2,-DIM//2,-1), DIM )

    y = tf.tile( tf.range(-DIM//2,DIM//2),[DIM] )

    z = tf.ones([DIM*DIM],dtype='int32')

    idx = tf.stack( [x,y,z] )

    

    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS

    idx2 = K.dot(shift_matrix,tf.cast(idx,dtype='float32'))

    idx2 = K.cast(idx2,dtype='int32')

    idx2 = K.clip(idx2,-DIM//2+XDIM+1,DIM//2)

    

    # FIND ORIGIN PIXEL VALUES 

    idx3 = tf.stack( [DIM//2-idx2[0,], DIM//2-1+idx2[1,]] )

    d = tf.gather_nd(image, tf.transpose(idx3))

        

    return tf.reshape(d,[DIM,DIM,3])



def transform_zoom(image):

    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]

    # output - image randomly zoomed

    DIM = HEIGHT

    XDIM = DIM%2 #fix for size 331

    

    height_zoom = 1.0 + tf.random.normal([1],dtype='float32')/10.

    width_zoom = 1.0 + tf.random.normal([1],dtype='float32')/10.

    one = tf.constant([1],dtype='float32')

    zero = tf.constant([0],dtype='float32')

        

    # ZOOM MATRIX

    zoom_matrix = tf.reshape( tf.concat([one/height_zoom,zero,zero, zero,one/width_zoom,zero, zero,zero,one],axis=0),[3,3] )



    # LIST DESTINATION PIXEL INDICES

    x = tf.repeat( tf.range(DIM//2,-DIM//2,-1), DIM )

    y = tf.tile( tf.range(-DIM//2,DIM//2),[DIM] )

    z = tf.ones([DIM*DIM],dtype='int32')

    idx = tf.stack( [x,y,z] )

    

    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS

    idx2 = K.dot(zoom_matrix,tf.cast(idx,dtype='float32'))

    idx2 = K.cast(idx2,dtype='int32')

    idx2 = K.clip(idx2,-DIM//2+XDIM+1,DIM//2)

    

    # FIND ORIGIN PIXEL VALUES 

    idx3 = tf.stack( [DIM//2-idx2[0,], DIM//2-1+idx2[1,]] )

    d = tf.gather_nd(image, tf.transpose(idx3))

        

    return tf.reshape(d,[DIM,DIM,3])
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
import re



def count_data_items(filenames):

    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]

    return np.sum(n)



NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)

NUM_VALIDATION_IMAGES = count_data_items(VALIDATION_FILENAMES)

NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

print('Dataset: {} training images, {} validation images, {} unlabeled test images'.format(NUM_TRAINING_IMAGES, NUM_VALIDATION_IMAGES, NUM_TEST_IMAGES))
import math
print("Training data shape:")

for image, label in get_training_dataset().take(3):

    print(image.numpy().shape, label.numpy().shape)

print("Training data label examples:", label.numpy())

print("-----------------------------------------------------------")

print("Validation data shape:")

for image, label in get_validation_dataset().take(3):

    print(image.numpy().shape, label.numpy().shape)

print("Validation data label examples:", label.numpy())

print("-----------------------------------------------------------")

print("Test data shape:")

for image, idnum in get_test_dataset().take(3):

    print(image.numpy().shape, idnum.numpy().shape)

print("Test data IDs:", idnum.numpy().astype('U')) # U=unicode string
# Peek at training data

training_dataset = get_training_dataset()

training_dataset = training_dataset.unbatch().batch(10)

train_batch = iter(training_dataset)
from matplotlib import pyplot as plt

import math

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

        title = '' if label is None else CLASSES[label]

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
# display some images

display_batch_of_images(next(train_batch))
# Peek at test data

test_dataset = get_test_dataset()

test_dataset = test_dataset.unbatch().batch(10)

test_batch = iter(test_dataset)
# display test images

display_batch_of_images(next(test_batch))
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

        plt.subplots(figsize=(10,10), facecolor='#FF5733')

        plt.tight_layout()

    ax = plt.subplot(subplot)

    ax.set_facecolor('#33FF60')

    ax.plot(training)

    ax.plot(validation)

    ax.set_title('model '+ title)

    ax.set_ylabel(title)

    ax.set_xlabel('epoch')

    ax.legend(['training', 'validation'])
def loss_fn(y_true,y_pred):

    ## sparse_categorical_crossentropy with logits

    return tf.keras.losses.sparse_categorical_crossentropy(y_true,y_pred,from_logits = True)



def acc_fn(y_true,y_pred):

    ## see if we need softmax or not 

    return tf.keras.metrics.sparse_categorical_accuracy(y_true,y_pred)
models = []

histories = []
with tpu_strategy.scope():

    enet = efn.EfficientNetB7(

        input_shape=[*IMAGE_SIZE, 3],

        weights='imagenet',

        include_top=False

    )



    model = tf.keras.Sequential([

        enet,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(len(CLASSES))

    ])

        

model.compile(

    optimizer=tf.keras.optimizers.Adam(lr=3e-5 * tpu_strategy.num_replicas_in_sync),

    loss = loss_fn,

    metrics=[acc_fn]

)

model.summary()
# Learning rate schedule for TPU, GPU and CPU.

# Using an LR ramp up because fine-tuning a pre-trained model.

# Starting with a high LR would break the pre-trained weights.



LR_START = 0.00001

LR_MAX = 0.00005 * tpu_strategy.num_replicas_in_sync

LR_MIN = 0.00001

LR_RAMPUP_EPOCHS = 5

LR_SUSTAIN_EPOCHS = 0

LR_EXP_DECAY = .8



def lrfn(epoch):

    if epoch < LR_RAMPUP_EPOCHS:

        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START

    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:

        lr = LR_MAX

    else:

        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN

    return lr



# Reduce learning rate when a metric has stopped improving

lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)



rng = [i for i in range(EPOCHS)]

y = [lrfn(x) for x in rng]

plt.plot(rng, y)

print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))
history = model.fit(

    get_training_dataset(), 

    steps_per_epoch=STEPS_PER_EPOCH,

    epochs=EPOCHS,

    callbacks=[lr_callback],

    validation_data=None if SKIP_VALIDATION else get_validation_dataset(),

    verbose = 1

)
if not SKIP_VALIDATION:

    display_training_curves(history.history['loss'], history.history['val_loss'], 'loss', 211)

    display_training_curves(history.history['sparse_categorical_accuracy'], history.history['val_sparse_categorical_accuracy'], 'accuracy', 212)
# with tpu_strategy.scope():

#     res = ResNet152V2(

#         input_shape=[*IMAGE_SIZE, 3],

#         weights='imagenet',

#         include_top=False

#     )



#     model2 = tf.keras.Sequential([

#         res,

#         tf.keras.layers.GlobalAveragePooling2D(),

#         tf.keras.layers.Dense(len(CLASSES), activation='softmax')

#     ])

        

# model2.compile(

#     optimizer=tf.keras.optimizers.Adam(lr=0.0001),

#     loss = 'sparse_categorical_crossentropy',

#     metrics=['sparse_categorical_accuracy']

# )

# model2.summary()
# history2 = model2.fit(

#     get_training_dataset(), 

#     steps_per_epoch=STEPS_PER_EPOCH,

#     epochs=EPOCHS, 

#     callbacks=[lr_callback],

#     validation_data=None if SKIP_VALIDATION else get_validation_dataset()

# )
# if not SKIP_VALIDATION:

#     display_training_curves(history2.history['loss'], history2.history['val_loss'], 'loss', 211)

#     display_training_curves(history2.history['sparse_categorical_accuracy'], history2.history['val_sparse_categorical_accuracy'], 'accuracy', 212)
# with tpu_strategy.scope():

#     Inct=InceptionResNetV2(

#         input_shape=[*IMAGE_SIZE, 3],

#         weights='imagenet',

#         include_top=False

      

#     )

#     model3 = tf.keras.Sequential([

#         Inct,

#         tf.keras.layers.GlobalMaxPooling2D(),

#         tf.keras.layers.Dense(len(CLASSES), activation='softmax')

#     ])

        

# model3.compile(

#     optimizer=tf.keras.optimizers.Adam(lr=0.0001),

#     loss = 'sparse_categorical_crossentropy',

#     metrics=['sparse_categorical_accuracy']

# )

# model3.summary()
# history3 = model3.fit(

#     get_training_dataset(), 

#     steps_per_epoch=STEPS_PER_EPOCH,

#     epochs=EPOCHS, 

#     callbacks=[lr_callback],

#     validation_data=None if SKIP_VALIDATION else get_validation_dataset()

# )
# if not SKIP_VALIDATION:

#     display_training_curves(history3.history['loss'], history3.history['val_loss'], 'loss', 211)

#     display_training_curves(history3.history['sparse_categorical_accuracy'], history3.history['val_sparse_categorical_accuracy'], 'accuracy', 212)
with tpu_strategy.scope():

    den=DenseNet201(

        input_shape=[*IMAGE_SIZE, 3],

        weights='imagenet',

        include_top=False

      

    )

    model2 = tf.keras.Sequential([

        den,

        tf.keras.layers.GlobalMaxPooling2D(),

        tf.keras.layers.Dense(len(CLASSES))

    ])

        

model2.compile(

    optimizer=tf.keras.optimizers.Adam(lr=3e-5 * tpu_strategy.num_replicas_in_sync),

    loss = loss_fn,

    metrics = [acc_fn]

)

model2.summary()
history4 = model2.fit(

    get_training_dataset(), 

    steps_per_epoch=STEPS_PER_EPOCH,

    epochs=EPOCHS, 

    callbacks=[lr_callback],

    validation_data=None if SKIP_VALIDATION else get_validation_dataset()

)
if not SKIP_VALIDATION:

    display_training_curves(history4.history['loss'], history4.history['val_loss'], 'loss', 211)

    display_training_curves(history4.history['sparse_categorical_accuracy'], history4.history['val_sparse_categorical_accuracy'], 'accuracy', 212)
# Finding best alpha and Beta

if not SKIP_VALIDATION:

    cmdataset = get_validation_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and labels, order matters.

    images_ds = cmdataset.map(lambda image, label: image)

    labels_ds = cmdataset.map(lambda image, label: label).unbatch()

    cm_correct_labels = next(iter(labels_ds.batch(NUM_VALIDATION_IMAGES))).numpy() # get everything as one batch

    m = model.predict(images_ds)

    m2 = model2.predict(images_ds)

    scores = []

    for alpha in np.linspace(0,1,100):

        cm_probabilities = alpha*m+(1-alpha)*m2

        cm_predictions = np.argmax(cm_probabilities, axis=-1)

        scores.append(f1_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro'))

        

    print("Correct   labels: ", cm_correct_labels.shape, cm_correct_labels)

    print("Predicted labels: ", cm_predictions.shape, cm_predictions)

    plt.plot(scores)

    best_alpha = np.argmax(scores)/100

    cm_probabilities = best_alpha*m+(1-best_alpha)*m2

    cm_predictions = np.argmax(cm_probabilities, axis=-1)

else:

    best_alpha = 0.44
print(best_alpha)
if not SKIP_VALIDATION:

    cmat = confusion_matrix(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)))

    score = f1_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')

    precision = precision_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')

    recall = recall_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')

    display_confusion_matrix(cmat, score, precision, recall)

    print('f1 score: {:.3f}, precision: {:.3f}, recall: {:.3f}'.format(score, precision, recall))
# test_ds = get_test_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and ids, order matters.



# print('Computing predictions...')

# test_images_ds = test_ds.map(lambda image, idnum: image)

# probabilities = (model.predict(test_images_ds) + model2.predict(test_images_ds)+model3.predict(test_images_ds)+model4.predict(test_images_ds))/4

# predictions = np.argmax(probabilities, axis=-1)

# print(predictions)



# # Generating submission.csv file

# test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()

# test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch

# np.savetxt('submission.csv', np.rec.fromarrays([test_ids, predictions]), fmt=['%s', '%d'], delimiter=',', header='id,label', comments='')
TTA_NUM = 10

probabilities = []

for i in range(TTA_NUM):

    print('TTA Number: ',i)

    test_ds = get_test_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and ids, order matters.

    test_images_ds = test_ds.map(lambda image, idnum: image)

    probabilities.append(model.predict(test_images_ds,verbose =1))
prob1 = np.mean(probabilities,axis =0)
TTA_NUM = 10

probabilities = []

for i in range(TTA_NUM):

    print('TTA Number: ',i)

    test_ds = get_test_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and ids, order matters.

    test_images_ds = test_ds.map(lambda image, idnum: image)

    probabilities.append(model2.predict(test_images_ds,verbose =1))
prob2 = np.mean(probabilities,axis =0)
prob = best_alpha*prob1 + (1-best_alpha)*prob2

predictions = np.argmax(prob, axis=-1)

print(predictions)



# Generating submission.csv file

test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()

test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch

np.savetxt('submission.csv', np.rec.fromarrays([test_ids, predictions]), fmt=['%s', '%d'], delimiter=',', header='id,label', comments='')