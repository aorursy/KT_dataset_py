!pip install -U --pre efficientnet
import math, re, os

import tensorflow as tf

import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

from kaggle_datasets import KaggleDatasets

import efficientnet.tfkeras as efn

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from tensorflow.keras.applications import DenseNet201

print("Tensorflow version " + tf.__version__)

'''

AUTO = tf.data.experimental.AUTOTUNE



# Create strategy from tpu

tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

tf.config.experimental_connect_to_cluster(tpu)

tf.tpu.experimental.initialize_tpu_system(tpu)

strategy = tf.distribute.experimental.TPUStrategy(tpu)



# Data access

GCS_DS_PATH = KaggleDatasets().get_gcs_path()



# Configuration

IMAGE_SIZE = [512, 512]

EPOCHS1 =25

EPOCHS2 =18

BATCH_SIZE = 2 * 16 * strategy.num_replicas_in_sync

'''
AUTO = tf.data.experimental.AUTOTUNE



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





GCS_DS_PATH = 'gs://kds-a44744780cb28b908f65e7fa3b4a473ef199bb9ffe8048dbbf29ffeb'#'gs://flowers-public' 

#GCS_DS_PATH = KaggleDatasets().get_gcs_path() # 





# Configuration

IMAGE_SIZE = [512, 512]

EPOCHS1=50

EPOCHS2=50

BATCH_SIZE = 1 * 16 * strategy.num_replicas_in_sync
LR_START = 0.0001

LR_MAX = 0.00005 * strategy.num_replicas_in_sync

LR_MIN = 0.00001

LR_RAMPUP_EPOCHS = 4

LR_SUSTAIN_EPOCHS = 4

LR_EXP_DECAY = .8



def lrfn(epoch):

    if epoch < LR_RAMPUP_EPOCHS:

        lr = np.random.random_sample() * LR_START

    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:

        lr = LR_MAX

    else:

        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN

    return lr

    

lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)
def count_data_items(filenames):

    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]

    return np.sum(n)
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



# watch out for overfitting!

SKIP_VALIDATION = False

if SKIP_VALIDATION:

    TRAINING_FILENAMES = TRAINING_FILENAMES + VALIDATION_FILENAMES


VALIDATION_FILENAMES_1 = tf.io.gfile.glob(GCS_PATH + '/val/*.tfrec')[:int(len(VALIDATION_FILENAMES)/2)]

VALIDATION_FILENAMES_2 = tf.io.gfile.glob(GCS_PATH + '/val/*.tfrec')[int(len(VALIDATION_FILENAMES)/2):]



TRAINING_FILENAMES_1 = TRAINING_FILENAMES + VALIDATION_FILENAMES_1

TRAINING_FILENAMES_2 = TRAINING_FILENAMES + VALIDATION_FILENAMES_2



NUM_TRAINING_IMAGES_1 = count_data_items(TRAINING_FILENAMES_1)

NUM_TRAINING_IMAGES_2 = count_data_items(TRAINING_FILENAMES_2)



NUM_VALIDATION_IMAGES_1 = (1 - SKIP_VALIDATION) * count_data_items(VALIDATION_FILENAMES_1)

NUM_VALIDATION_IMAGES_2 = (1 - SKIP_VALIDATION) * count_data_items(VALIDATION_FILENAMES_2)



NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)



STEPS_PER_EPOCH_1 = NUM_TRAINING_IMAGES_1 // BATCH_SIZE

STEPS_PER_EPOCH_2 = NUM_TRAINING_IMAGES_2 // BATCH_SIZE



print('Dataset: {} training_1 images, {} validation images, {} unlabeled test images'.format(NUM_TRAINING_IMAGES_1, NUM_VALIDATION_IMAGES_1, NUM_TEST_IMAGES))

print('Dataset: {} training_2 images, {} validation images, {} unlabeled test images'.format(NUM_TRAINING_IMAGES_2, NUM_VALIDATION_IMAGES_2, NUM_TEST_IMAGES))
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
img_size=512

def random_blockout(img, sl=0.1, sh=0.2, rl=0.4):



    h, w, c = img_size, img_size, 3

    origin_area = tf.cast(h*w, tf.float32)



    e_size_l = tf.cast(tf.round(tf.sqrt(origin_area * sl * rl)), tf.int32)

    e_size_h = tf.cast(tf.round(tf.sqrt(origin_area * sh / rl)), tf.int32)



    e_height_h = tf.minimum(e_size_h, h)

    e_width_h = tf.minimum(e_size_h, w)



    erase_height = tf.random.uniform(shape=[], minval=e_size_l, maxval=e_height_h, dtype=tf.int32)

    erase_width = tf.random.uniform(shape=[], minval=e_size_l, maxval=e_width_h, dtype=tf.int32)



    erase_area = tf.zeros(shape=[erase_height, erase_width, c])

    erase_area = tf.cast(erase_area, tf.uint8)



    pad_h = h - erase_height

    pad_top = tf.random.uniform(shape=[], minval=0, maxval=pad_h, dtype=tf.int32)

    pad_bottom = pad_h - pad_top



    pad_w = w - erase_width

    pad_left = tf.random.uniform(shape=[], minval=0, maxval=pad_w, dtype=tf.int32)

    pad_right = pad_w - pad_left



    erase_mask = tf.pad([erase_area], [[0,0],[pad_top, pad_bottom], [pad_left, pad_right], [0,0]], constant_values=1)

    erase_mask = tf.squeeze(erase_mask, axis=0)

    erased_img = tf.multiply(tf.cast(img,tf.float32), tf.cast(erase_mask, tf.float32))



    return tf.cast(erased_img, img.dtype)
def decode_image(image_data):

    image = tf.image.decode_jpeg(image_data, channels=3)

    #image = tf.io.decode_image(image_data, channels=3)

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

    #label = tf.one_hot(label, len(CLASSES))

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



def data_augment(image, label):

    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),

    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part

    # of the TPU while the TPU itself is computing gradients.

    image = tf.image.random_flip_left_right(image)

    #image = tf.image.random_saturation(image, 0, 2)

    image= random_blockout(image)

    return image, label   





def get_training_dataset():

    dataset = load_dataset(TRAINING_FILENAMES, labeled=True)

    #dataset = dataset.map(data_augment, num_parallel_calls=AUTO)    

    dataset = dataset.repeat() # the training dataset must repeat for several epochs

    dataset = dataset.shuffle(3456)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def get_training_dataset1():

    dataset = load_dataset(TRAINING_FILENAMES_1, labeled=True)

    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)    

    dataset = dataset.repeat() # the training dataset must repeat for several epochs

    dataset = dataset.shuffle(3456)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def get_training_dataset2():

    dataset = load_dataset(TRAINING_FILENAMES_2, labeled=True)

    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)    

    dataset = dataset.repeat() # the training dataset must repeat for several epochs

    dataset = dataset.shuffle(3456)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def get_validation_dataset1(ordered=False):

    dataset = load_dataset(VALIDATION_FILENAMES_1, labeled=True, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.cache()

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def get_validation_dataset2(ordered=False):

    dataset = load_dataset(VALIDATION_FILENAMES_2, labeled=True, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.cache()

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def get_validation_dataset3(ordered=False):

    dataset = load_dataset(VALIDATION_FILENAMES, labeled=True, ordered=ordered)

    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)  

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.cache()

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def get_test_dataset(ordered=False):

    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def data_augment2(image, label):

    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),

    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part

    # of the TPU while the TPU itself is computing gradients.

    image = tf.image.flip_left_right(image)

    #image = tf.image.random_saturation(image, 0, 2)

    #image= random_blockout(image)

    return image, label   



def get_test_dataset2(ordered=False):

    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)

    dataset = dataset.map(data_augment2, num_parallel_calls=AUTO)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset





# Need this line so Google will recite some incantations

# for Turing to magically load the model onto the TPU

with strategy.scope():

    rnet = DenseNet201(

        input_shape=(512, 512, 3),

        weights='imagenet',

        include_top=False

    )



    model = tf.keras.Sequential([

        rnet,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(len(CLASSES), activation='softmax')

    ])

        

model.compile(

    optimizer=tf.keras.optimizers.Adam(lr=0.0001),

    loss = 'sparse_categorical_crossentropy',

    metrics=[]

)

model.summary()
#scheduler = tf.keras.callbacks.ReduceLROnPlateau(patience=3, verbose=1)



#patience=5 

#lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', factor=(5/10), patience=patience, min_lr=0.00001, verbose= 1)    

#es = tf.keras.callbacks.EarlyStopping(monitor='val_f1', mode='max', verbose=1, patience=patience) #restore_best_weights=True,

#mc = tf.keras.callbacks.ModelCheckpoint('EfficientNetB7.h5', verbose=1, monitor='val_f1_score', mode='max', 

#                                        save_best_only=True, save_weights_only=True) 



#es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience) #restore_best_weights=True,

#mc = tf.keras.callbacks.ModelCheckpoint('EfficientNetB7_Fold0_25.h5', verbose=1, monitor='val_loss', mode='min', 

#                                        save_best_only=True, save_weights_only=True) 

mc = tf.keras.callbacks.ModelCheckpoint('DenseNet201_validation2_50.h5', verbose=1, monitor='val_loss', mode='min', 

                                        save_best_only=True, save_weights_only=True) 



history = model.fit(

    get_training_dataset1(), 

    steps_per_epoch=STEPS_PER_EPOCH_1,

    epochs=EPOCHS1,

    callbacks=[lr_callback,mc],

    validation_data=None if SKIP_VALIDATION else get_validation_dataset2()

)



#model.save_weights('DenseNet201_25.h5')

#model.load_weights('../input/104-flower-classification-tpu/DenseNet201_25.h5')
with strategy.scope():

    rnet = DenseNet201(

        input_shape=(512, 512, 3),

        weights='imagenet',

        include_top=False

    )



    model2 = tf.keras.Sequential([

        rnet,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(len(CLASSES), activation='softmax')

    ])

        

model2.compile(

    optimizer=tf.keras.optimizers.Adam(lr=0.0001),

    loss = 'sparse_categorical_crossentropy',

    metrics=[]

)

model2.summary()
mc = tf.keras.callbacks.ModelCheckpoint('DenseNet201_validation1_50.h5', verbose=1, monitor='val_loss', mode='min', 

                                        save_best_only=True, save_weights_only=True) 



history2 = model2.fit(

    get_training_dataset2(), 

    steps_per_epoch=STEPS_PER_EPOCH_2,

    epochs=EPOCHS2,

    callbacks=[lr_callback,mc],

    validation_data=None if SKIP_VALIDATION else get_validation_dataset1()

)



#model.save_weights('EfficientNetB7_25.h5')

#model.load_weights('../input/104-flower-classification-tpu/EfficientNetB7_25.h5')