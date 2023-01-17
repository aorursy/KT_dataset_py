!pip install -U efficientnet
import re, math

# import math, re, os

import tensorflow as tf

import numpy as np

from matplotlib import pyplot as plt

print("Tensorflow version " + tf.__version__)

AUTO = tf.data.experimental.AUTOTUNE

from kaggle_datasets import KaggleDatasets

# from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix



import efficientnet.tfkeras
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
GCS_DS_PATH = KaggleDatasets().get_gcs_path('flower-classification-with-tpus') # you can list the bucket with "!gsutil ls $GCS_DS_PATH"
IMAGE_SIZE = [512, 512] # at this size, a GPU will run out of memory. Use the TPU

BATCH_SIZE = 16 * strategy.num_replicas_in_sync



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





GCS_PATH_SELECT = { # available image sizes

    192: GCS_DS_PATH + '/tfrecords-jpeg-192x192',

    224: GCS_DS_PATH + '/tfrecords-jpeg-224x224',

    331: GCS_DS_PATH + '/tfrecords-jpeg-331x331',

    512: GCS_DS_PATH + '/tfrecords-jpeg-512x512'

}

GCS_PATH = GCS_PATH_SELECT[IMAGE_SIZE[0]]



TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/test/*.tfrec') # predictions on this dataset should be submitted for the competition
# TEST_FILENAMES = tf.io.gfile.glob('/kaggle/input/flower-classification-with-tpus/tfrecords-jpeg-512x512/test/*.tfrec') # predictions on this dataset should be submitted for the competition
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



# def title_from_label_and_target(label, correct_label):

#     if correct_label is None:

#         return CLASSES[label], True

#     correct = (label == correct_label)

#     return "{} [{}{}{}]".format(CLASSES[label], 'OK' if correct else 'NO', u"\u2192" if not correct else '',

#                                 CLASSES[correct_label] if not correct else ''), correct



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

        title = '' if label is None else CLASSES[np.argmax(label)]

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



# def display_confusion_matrix(cmat, score, precision, recall):

#     plt.figure(figsize=(15,15))

#     ax = plt.gca()

#     ax.matshow(cmat, cmap='Reds')

#     ax.set_xticks(range(len(CLASSES)))

#     ax.set_xticklabels(CLASSES, fontdict={'fontsize': 7})

#     plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")

#     ax.set_yticks(range(len(CLASSES)))

#     ax.set_yticklabels(CLASSES, fontdict={'fontsize': 7})

#     plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")

#     titlestring = ""

#     if score is not None:

#         titlestring += 'f1 = {:.3f} '.format(score)

#     if precision is not None:

#         titlestring += '\nprecision = {:.3f} '.format(precision)

#     if recall is not None:

#         titlestring += '\nrecall = {:.3f} '.format(recall)

#     if len(titlestring) > 0:

#         ax.text(101, 1, titlestring, fontdict={'fontsize': 18, 'horizontalalignment':'right', 'verticalalignment':'top', 'color':'#804040'})

#     plt.show()

    

# def display_training_curves(training, validation, title, subplot):

#     if subplot%10==1: # set up the subplots on the first call

#         plt.subplots(figsize=(10,10), facecolor='#F0F0F0')

#         plt.tight_layout()

#     ax = plt.subplot(subplot)

#     ax.set_facecolor('#F8F8F8')

#     ax.plot(training)

#     ax.plot(validation)

#     ax.set_title('model '+ title)

#     ax.set_ylabel(title)

#     #ax.set_ylim(0.28,1.05)

#     ax.set_xlabel('epoch')

#     ax.legend(['train', 'valid.'])
def count_data_items(filenames):

    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]

    return np.sum(n)



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

    one_hot_class = tf.one_hot(label, depth=len(CLASSES))

    return image, one_hot_class # returns a dataset of (image, label) pairs

    

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

    

# def force_image_sizes(dataset, image_size):

#     # explicit size needed for TPU

#     reshape_images = lambda image, label: (tf.reshape(image, [*image_size, 3]), label)

#     dataset = dataset.map(reshape_images, num_parallel_calls=AUTO)

#     return dataset



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



def data_augment(image, one_hot_class):

    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),

    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part

    # of the TPU while the TPU itself is computing gradients.

    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_saturation(image, 0.8, 1)

#     image = tf.image.random_jpeg_quality(image, 80, 100)

    image = tf.image.random_brightness(image, 0.1)

    image = tf.image.random_contrast(image, 0.8, 1)

    return image, one_hot_class



# def get_training_dataset():

#     dataset = load_dataset(TRAINING_FILENAMES, labeled=True)

#     dataset = dataset.map(data_augment, num_parallel_calls=AUTO)

#     dataset = dataset.repeat() # the training dataset must repeat for several epochs

#     dataset = dataset.shuffle(2048)

#     dataset = dataset.batch(BATCH_SIZE)

#     dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

#     return dataset



# def get_validation_dataset(ordered=False):

#     dataset = load_dataset(VALIDATION_FILENAMES, labeled=True, ordered=ordered)

#     dataset = dataset.batch(BATCH_SIZE)

#     dataset = dataset.cache()

#     dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

#     return dataset



def get_test_dataset(ordered=False):

    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



# NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)

# NUM_VALIDATION_IMAGES = count_data_items(VALIDATION_FILENAMES)

NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)

# STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

# print('Dataset: {} training images, {} validation images, {} unlabeled test images'.format(NUM_TRAINING_IMAGES, NUM_VALIDATION_IMAGES, NUM_TEST_IMAGES))

print('Dataset: {} unlabeled test images'.format(NUM_TEST_IMAGES))
print("Test data shapes:")

for image, idnum in get_test_dataset().take(3):

    print(image.numpy().shape, idnum.numpy().shape)

print("Test data IDs:", idnum.numpy().astype('U')) # U=unicode string
# peer at test data

test_dataset = get_test_dataset()

test_dataset = test_dataset.unbatch().batch(20)

test_batch = iter(test_dataset)
# run this cell again for next set of images

display_batch_of_images(next(test_batch))
EFFICIENTNETB7_CKPT = '/kaggle/input/flowers-with-tpu-efficientnetb7-focalloss/model.h5'
with strategy.scope():

    pretrained_model = efficientnet.tfkeras.EfficientNetB7(weights=None, include_top=False)



    effb7 = tf.keras.Sequential([

        pretrained_model,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(len(CLASSES), activation='softmax')

    ])



    effb7.load_weights(EFFICIENTNETB7_CKPT)
DENSENET201_CKPT = '/kaggle/input/flowers-with-tpu-densenet201-focalloss/model.h5'
with strategy.scope():

    pretrained_model = tf.keras.applications.DenseNet201(weights=None, input_shape=[*IMAGE_SIZE, 3], include_top=False)



    den201 = tf.keras.Sequential([

        pretrained_model,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(len(CLASSES), activation='softmax')

    ])



    den201.load_weights(DENSENET201_CKPT)
RESNET152_CKPT = '/kaggle/input/flowers-with-tpu-resnet152-focalloss/model.h5'
with strategy.scope():

    pretrained_model = tf.keras.applications.ResNet152V2(weights=None, input_shape=[*IMAGE_SIZE, 3], include_top=False)



    res152 = tf.keras.Sequential([

        pretrained_model,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(len(CLASSES), activation='softmax')

    ])



    res152.load_weights(RESNET152_CKPT)
INCEPTIONRESNET_CKPT = '/kaggle/input/flowers-with-tpu-inceptionresnet-focalloss/model.h5'
with strategy.scope():

    pretrained_model = tf.keras.applications.InceptionResNetV2(weights=None, input_shape=[*IMAGE_SIZE, 3], include_top=False)



    inres = tf.keras.Sequential([

        pretrained_model,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(len(CLASSES), activation='softmax')

    ])



    inres.load_weights(INCEPTIONRESNET_CKPT)
XCEPTION_CKPT = '/kaggle/input/flowers-with-tpu-xception-focalloss/model.h5'
with strategy.scope():

    pretrained_model = tf.keras.applications.Xception(weights=None, input_shape=[*IMAGE_SIZE, 3], include_top=False)



    xcep = tf.keras.Sequential([

        pretrained_model,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(len(CLASSES), activation='softmax')

    ])



    xcep.load_weights(XCEPTION_CKPT)
INCEPTION_CKPT = '/kaggle/input/flowers-with-tpu-inception-focalloss/model.h5'
with strategy.scope():

    pretrained_model = tf.keras.applications.InceptionV3(weights=None, input_shape=[*IMAGE_SIZE, 3], include_top=False)



    incep = tf.keras.Sequential([

        pretrained_model,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(len(CLASSES), activation='softmax')

    ])



    incep.load_weights(INCEPTION_CKPT)
MOBILENET_CKPT = '/kaggle/input/flowers-with-tpu-mobilenet-focalloss/model.h5'
with strategy.scope():

    pretrained_model = tf.keras.applications.MobileNetV2(weights=None, input_shape=[*IMAGE_SIZE, 3], include_top=False)



    mobi = tf.keras.Sequential([

        pretrained_model,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(len(CLASSES), activation='softmax')

    ])



    mobi.load_weights(MOBILENET_CKPT)
VGG19_CKPT = '/kaggle/input/flowers-with-tpu-vgg19-focalloss/model.h5'
with strategy.scope():

    pretrained_model = tf.keras.applications.VGG19(weights=None, input_shape=[*IMAGE_SIZE, 3], include_top=False)



    vgg19 = tf.keras.Sequential([

        pretrained_model,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(len(CLASSES), activation='softmax')

    ])



    vgg19.load_weights(VGG19_CKPT)
test_ds = get_test_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and ids, order matters.



print('Computing predictions...')

test_images_ds = test_ds.map(lambda image, idnum: image)
effb7_probabilities = effb7.predict(test_images_ds)
den201_probabilities = den201.predict(test_images_ds)
res152_probabilities = res152.predict(test_images_ds)
inres_probabilities = inres.predict(test_images_ds)
xcep_probabilities = xcep.predict(test_images_ds)
incep_probabilities = incep.predict(test_images_ds)
mobi_probabilities = mobi.predict(test_images_ds)
vgg19_probabilities = vgg19.predict(test_images_ds)
probabilities = np.mean(

    [

        effb7_probabilities,

        den201_probabilities,

        res152_probabilities,

        inres_probabilities,

        xcep_probabilities,

        incep_probabilities,

        mobi_probabilities,

        vgg19_probabilities

    ],

    axis=0

)



predictions = np.argmax(probabilities, axis=-1)

print(predictions)
print('Generating submission.csv file...')

test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()

test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch

np.savetxt('submission.csv', np.rec.fromarrays([test_ids, predictions]), fmt=['%s', '%d'], delimiter=',', header='id,label', comments='')

!head submission.csv