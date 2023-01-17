!pip install -q efficientnet
import math, re, os

import tensorflow as tf

import numpy as np

import pandas as pd

import efficientnet.tfkeras as efn



from kaggle_datasets import KaggleDatasets

print("Tensorflow version " + tf.__version__)



AUTO = tf.data.experimental.AUTOTUNE
# Detect hardware, return appropriate distribution strategy

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

    startegy = tf.distribute.get_strategy()

    

print("REPLICAS", strategy.num_replicas_in_sync)

    
GCS_DS_PATH = KaggleDatasets().get_gcs_path('flower-classification-with-tpus')



IMAGE_SIZE = [512, 512]



BATCH_SIZE = 16 * strategy.num_replicas_in_sync



GCS_PATH_SELECT = { # available image sizes

    192: GCS_DS_PATH + '/tfrecords-jpeg-192x192/',

    224: GCS_DS_PATH + '/tfrecords-jpeg-224x224/',

    331: GCS_DS_PATH + '/tfrecords-jpeg-331x331/',

    512: GCS_DS_PATH + '/tfrecords-jpeg-512x512/'

}

GCS_PATH = GCS_PATH_SELECT[IMAGE_SIZE[0]]



TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + 'train/*.tfrec')

VALIDATION_FILENAMES = tf.io.gfile.glob(GCS_PATH + 'val/*.tfrec')

TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + 'test/*.tfrec')
GCS_DS_PATH = KaggleDatasets().get_gcs_path('oxford-flowers-tfrecords')



IMAGE_SIZE = [512, 512] 



BATCH_SIZE = 32 * strategy.num_replicas_in_sync



GCS_PATH_SELECT = { # available image sizes

    192: GCS_DS_PATH + '/tfrecords-png-192x192/',

    224: GCS_DS_PATH + '/tfrecords-png-224x224/',

    331: GCS_DS_PATH + '/tfrecords-png-331x331/',

    512: GCS_DS_PATH + '/tfrecords-png-512x512/'

}

GCS_PATH = GCS_PATH_SELECT[IMAGE_SIZE[0]]



TRAINING_FILENAMES += tf.io.gfile.glob(GCS_PATH + '*.tfrec')
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



def data_augment(image, label):

    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),

    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part

    # of the TPU while the TPU itself is computing gradients.

    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_saturation(image, 0, 2)

    image = tf.image.random_brightness(image, 3)

    image = tf.image.random_contrast(image, 0, 2)

    image = tf.image.random_flip_up_down(image)

    image = tf.image.random_hue(image, 0.25)

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

NUM_VALIDATION_IMAGES = count_data_items(VALIDATION_FILENAMES)

NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

print('Dataset: {} training images, {} validation images, {} unlabeled test images'.format(NUM_TRAINING_IMAGES, NUM_VALIDATION_IMAGES, NUM_TEST_IMAGES))
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

print("Test data IDs:", idnum.numpy().astype('U'))
with strategy.scope():

    pretrained_model1 = tf.keras.applications.DenseNet201(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])

    pretrained_model1.trainable = True # False = transfer learning, True = fine-tuning

    

    model1 = tf.keras.Sequential([

        pretrained_model1,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(len(CLASSES), activation='softmax')

    ])

        

model1.compile(

    optimizer='adam',

    loss = 'sparse_categorical_crossentropy',

    metrics=['sparse_categorical_accuracy']

)

model1.summary()
with strategy.scope():

    pretrained_model2 = efn.EfficientNetB7(weights='noisy-student', include_top=False, input_shape=[*IMAGE_SIZE, 3])

    pretrained_model2.trainable = True # Full Training

    

    model2 = tf.keras.Sequential([

        pretrained_model2,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(len(CLASSES), activation='softmax')

    ])

        

model2.compile(

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False),

    loss = 'sparse_categorical_crossentropy',

    metrics=['sparse_categorical_accuracy']

)

model2.summary()
LR_START = 0.0001

LR_MAX = 0.00005 * strategy.num_replicas_in_sync

LR_MIN = 0.00001

LR_RAMPUP_EPOCHS = 4

LR_SUSTAIN_EPOCHS = 6

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



es_callback = tf.keras.callbacks.EarlyStopping(min_delta=0, patience=10, verbose=1, mode='auto', restore_best_weights=True)
EPOCHS = 30
model1.fit(get_training_dataset(), steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, callbacks=[lr_callback], validation_data=get_validation_dataset())
model2.fit(get_training_dataset(), steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, callbacks=[lr_callback], validation_data=get_validation_dataset())
test_ds = get_test_dataset(ordered=True)



print('Computing predictions...')

test_images_ds = test_ds.map(lambda image, idnum: image)

probabilities = (model1.predict(test_images_ds)*0.3)+model2.predict(test_images_ds)*0.7

predictions = np.argmax(probabilities, axis=-1)

print(predictions)



print('Generating submission.csv file...')

test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()

test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch

np.savetxt('submission.csv', np.rec.fromarrays([test_ids, predictions]), fmt=['%s', '%d'], delimiter=',', header='id,label', comments='')

!head submission.csv