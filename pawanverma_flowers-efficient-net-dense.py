# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import math, re, os

import tensorflow as tf

import tensorflow.keras.backend as K

import numpy as np

from matplotlib import pyplot as plt

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

print("Tensorflow version " + tf.__version__)

AUTO = tf.data.experimental.AUTOTUNE
import logging, sys



sys.stdout = open("/kaggle/working/logoutput.txt", "w")



def setup_tf_logging():

    tf.get_logger().handlers = []

    tf.get_logger().addHandler(logging.StreamHandler(sys.stdout))



setup_tf_logging()
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
from kaggle_datasets import KaggleDatasets
GCS_DS_PATH = KaggleDatasets().get_gcs_path()
!gsutil ls $GCS_DS_PATH/
IMAGE_SIZE = [331, 331] # At this size, a GPU will run out of memory. Use the TPU.

                        # For GPU training, please select 224 x 224 px image size.

IMAGE_SIZE_1 = [512,512]

#IMAGE_SIZE_2 = [331,331]

EPOCHS = 40

BATCH_SIZE = 16 * strategy.num_replicas_in_sync





GCS_PATH_SELECT = { # available image sizes

    192: GCS_DS_PATH + '/tfrecords-jpeg-192x192',

    224: GCS_DS_PATH + '/tfrecords-jpeg-224x224',

    331: GCS_DS_PATH + '/tfrecords-jpeg-331x331',

    512: GCS_DS_PATH + '/tfrecords-jpeg-512x512'

}

GCS_PATH = GCS_PATH_SELECT[IMAGE_SIZE[0]]

GCS_PATH_1 = GCS_PATH_SELECT[IMAGE_SIZE_1[0]]

#GCS_PATH_2 = GCS_PATH_SELECT[IMAGE_SIZE_2[0]]



TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/train/*.tfrec')

TRAINING_FILENAMES2 = tf.io.gfile.glob(GCS_PATH_1 + '/train/*.tfrec')

#TRAINING_FILENAMES3 = tf.io.gfile.glob(GCS_PATH_2 + '/train/*.tfrec')

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

           'trumpet creeper',  'blackberry lily',           'common tulip',     'wild rose']                                                                                                                                               #
FULL_TRAINING=True

if FULL_TRAINING:

    TRAINING_FILENAMES+=VALIDATION_FILENAMES

TRAINING_FILENAMES
np.set_printoptions(threshold=15, linewidth=80)

# Y=[]

# def count_flower_classes(databatch):

#     images, labels = batch_to_numpy_images_and_labels(databatch)

#     for label in labels:

#         Y.append(CLASSES[label])

#     return Y

    

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
from sklearn.utils import class_weight
def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):

    # returns 3x3 transformmatrix which transforms indicies

        

    # CONVERT DEGREES TO RADIANS

    rotation = math.pi * rotation / 180.

    shear = math.pi * shear / 180.

    

    # ROTATION MATRIX

    c1 = tf.math.cos(rotation)

    s1 = tf.math.sin(rotation)

    one = tf.constant([1],dtype='float32')

    zero = tf.constant([0],dtype='float32')

    rotation_matrix = tf.reshape( tf.concat([c1,s1,zero, -s1,c1,zero, zero,zero,one],axis=0),[3,3] )

        

    # SHEAR MATRIX

    c2 = tf.math.cos(shear)

    s2 = tf.math.sin(shear)

    shear_matrix = tf.reshape( tf.concat([one,s2,zero, zero,c2,zero, zero,zero,one],axis=0),[3,3] )    

    

    # ZOOM MATRIX

    zoom_matrix = tf.reshape( tf.concat([one/height_zoom,zero,zero, zero,one/width_zoom,zero, zero,zero,one],axis=0),[3,3] )

    

    # SHIFT MATRIX

    shift_matrix = tf.reshape( tf.concat([one,zero,height_shift, zero,one,width_shift, zero,zero,one],axis=0),[3,3] )

    

    return K.dot(K.dot(rotation_matrix, shear_matrix), K.dot(zoom_matrix, shift_matrix))



def transform(image,label):

    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]

    # output - image randomly rotated, sheared, zoomed, and shifted

    DIM = IMAGE_SIZE[0]

    XDIM = DIM%2 #fix for size 331

    

    rot = 15. * tf.random.normal([1],dtype='float32')

    shr = 5. * tf.random.normal([1],dtype='float32') 

    h_zoom = 1.0 + tf.random.normal([1],dtype='float32')/10.

    w_zoom = 1.0 + tf.random.normal([1],dtype='float32')/10.

    h_shift = 16. * tf.random.normal([1],dtype='float32') 

    w_shift = 16. * tf.random.normal([1],dtype='float32') 

  

    # GET TRANSFORMATION MATRIX

    m = get_mat(rot,shr,h_zoom,w_zoom,h_shift,w_shift) 



    # LIST DESTINATION PIXEL INDICES

    x = tf.repeat( tf.range(DIM//2,-DIM//2,-1), DIM )

    y = tf.tile( tf.range(-DIM//2,DIM//2),[DIM] )

    z = tf.ones([DIM*DIM],dtype='int32')

    idx = tf.stack( [x,y,z] )

    

    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS

    idx2 = K.dot(m,tf.cast(idx,dtype='float32'))

    idx2 = K.cast(idx2,dtype='int32')

    idx2 = K.clip(idx2,-DIM//2+XDIM+1,DIM//2)

    

    # FIND ORIGIN PIXEL VALUES           

    idx3 = tf.stack( [DIM//2-idx2[0,], DIM//2-1+idx2[1,]] )

    d = tf.gather_nd(image,tf.transpose(idx3))

        

    return tf.reshape(d,[DIM,DIM,3]),label


def decode_image(image_data):

    image = tf.image.decode_jpeg(image_data, channels=3)

    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range

    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU    

    return image

def decode_image2(image_data):

    image = tf.image.decode_jpeg(image_data, channels=3)

    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range

    image = tf.reshape(image, [*IMAGE_SIZE_1, 3]) # explicit size needed for TPU

    return image

def decode_image3(image_data):

    image = tf.image.decode_jpeg(image_data, channels=3)

    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range

    image = tf.reshape(image, [*IMAGE_SIZE_2, 3]) # explicit size needed for TPU

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

def read_labeled_tfrecord2(example):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element

    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image2(example['image'])

    label = tf.cast(example['class'], tf.int32)

    

    return image, label # returns a dataset of (image, label) pairs



def read_labeled_tfrecord3(example):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element

    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image3(example['image'])

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

def load_dataset2(filenames, labeled=True, ordered=False):

    # Read from TFRecords. For optimal performance, reading from multiple files at once and

    # disregarding data order. Order does not matter since we will be shuffling the data anyway.



    ignore_order = tf.data.Options()

    if not ordered:

        ignore_order.experimental_deterministic = False # disable order, increase speed



    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files

    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order

    dataset = dataset.map(read_labeled_tfrecord2 if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)

    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False

    return dataset



def load_dataset3(filenames, labeled=True, ordered=False):

    # Read from TFRecords. For optimal performance, reading from multiple files at once and

    # disregarding data order. Order does not matter since we will be shuffling the data anyway.



    ignore_order = tf.data.Options()

    if not ordered:

        ignore_order.experimental_deterministic = False # disable order, increase speed



    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files

    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order

    dataset = dataset.map(read_labeled_tfrecord3 if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)

    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False

    return dataset



def data_augment(image, label):

    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),

    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part

    # of the TPU while the TPU itself is computing gradients.

    image = tf.image.random_flip_left_right(image)

    

   

        

    #image = tf.image.random_saturation(image, 0, 2)

    #image = tf.image.resize_with_crop_or_pad(image, IMAGE_SIZE[0]+6, IMAGE_SIZE[0]+6) # Add 6 pixels of padding

    #image = tf.image.random_crop(image, size=[IMAGE_SIZE[0], IMAGE_SIZE[0],3]) # Random crop back to 512x512

    #image = tf.image.random_brightness(image, max_delta=0.5)

   

    return image, label  



def data_augment2(image, label):

    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),

    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part

    # of the TPU while the TPU itself is computing gradients.

    image = tf.image.random_flip_left_right(image)

    

    print(f"Resizing image with size {image.shape[1]} to {IMAGE_SIZE[0]}")

    image = tf.image.random_crop(image, size=[IMAGE_SIZE[0], IMAGE_SIZE[0],3])

        

    #image = tf.image.random_saturation(image, 0, 2)

    #image = tf.image.resize_with_crop_or_pad(image, IMAGE_SIZE[0]+6, IMAGE_SIZE[0]+6) # Add 6 pixels of padding

    #image = tf.image.random_crop(image, size=[IMAGE_SIZE[0], IMAGE_SIZE[0],3]) # Random crop back to 512x512

    #image = tf.image.random_brightness(image, max_delta=0.5)

   

    return image, label   



def get_training_dataset():

    dataset = load_dataset(TRAINING_FILENAMES, labeled=True)

    dataset2 = load_dataset2(TRAINING_FILENAMES2, labeled=True)

    #dataset3 = load_dataset3(TRAINING_FILENAMES3, labeled=True)

    

    #dataset = dataset.map(transform, num_parallel_calls=AUTO)

    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)

    dataset2 = dataset2.map(data_augment2, num_parallel_calls=AUTO)

    #dataset3 = dataset3.map(data_augment2, num_parallel_calls=AUTO)

    dataset = dataset.concatenate(dataset2)

    #dataset = dataset.concatenate(dataset3)

    dataset = dataset.repeat() # the training dataset must repeat for several epochs

    dataset = dataset.shuffle(2048)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset

def get_class_weights():

    dataset = load_dataset(TRAINING_FILENAMES, labeled=True)

    dataset2 = load_dataset2(TRAINING_FILENAMES2, labeled=True)

    #dataset3 = load_dataset3(TRAINING_FILENAMES3, labeled=True)

    dataset = dataset.concatenate(dataset2)

    #dataset = dataset.concatenate(dataset3)

    flower_counts=[]

    for images, labels in dataset:

       flower_counts.append(labels) 



    print(np.array(flower_counts))

    print(np.unique(flower_counts,return_counts=True))

    

    class_weights=class_weight.compute_class_weight('balanced',np.unique(flower_counts),np.array(flower_counts)  )

    print(f'class_weights:{class_weights}')

    return class_weights

    

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



NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)+count_data_items(TRAINING_FILENAMES2)#+count_data_items(TRAINING_FILENAMES3)

NUM_VALIDATION_IMAGES = count_data_items(VALIDATION_FILENAMES)

NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

print('Dataset: {} training images, {} validation images, {} unlabeled test images'.format(NUM_TRAINING_IMAGES, NUM_VALIDATION_IMAGES, NUM_TEST_IMAGES))

get_training_dataset()
# print("Training data shapes:")

# for image, label in get_training_dataset().take(3):

#     print(image.numpy().shape, label.numpy().shape)

# print("Training data label examples:", label.numpy())

# print("Validation data shapes:")

# for image, label in get_validation_dataset().take(3):

#     print(image.numpy().shape, label.numpy().shape)

# print("Validation data label examples:", label.numpy())

# print("Test data shapes:")

# for image, idnum in get_test_dataset().take(3):

#     print(image.numpy().shape, idnum.numpy().shape)

# print("Test data IDs:", idnum.numpy().astype('U')) # U=unicode string
training_dataset = get_training_dataset()

training_dataset = training_dataset.unbatch().batch(20)

train_batch = iter(training_dataset)
display_batch_of_images(next(train_batch))
# test_dataset = get_test_dataset()

# test_dataset = test_dataset.unbatch().batch(20)

# test_batch = iter(test_dataset)
# display_batch_of_images(next(test_batch))
class_weights=get_class_weights()
!pip install -q efficientnet
from efficientnet.tfkeras import EfficientNetB7
with strategy.scope():

    #pretrained_model = tf.keras.applications.DenseNet201(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])

    #pretrained_model = tf.keras.applications.Xception(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])

    #pretrained_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])

    pretrained_model_1  = EfficientNetB7(weights='noisy-student', include_top=False, input_shape=[*IMAGE_SIZE, 3])

    pretrained_model_1.trainable = True # False = transfer learning, True = fine-tuning

    

    model = tf.keras.Sequential([

        pretrained_model_1,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(len(CLASSES), activation='softmax')

    ])

        

model.compile(

    tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False),

    loss = 'sparse_categorical_crossentropy',

    metrics=['sparse_categorical_accuracy']

)

model.summary()
# print("Number of layers in the base model: ", len(pretrained_model_1.layers))

# #pretrained_model.summary()
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
if not FULL_TRAINING:

    es_callback = tf.keras.callbacks.EarlyStopping(monitor="val_sparse_categorical_accuracy",min_delta=0, patience=10, verbose=1, mode='auto', restore_best_weights=True)

else:

    es_callback = tf.keras.callbacks.EarlyStopping(monitor="sparse_categorical_accuracy",min_delta=0, patience=10, verbose=1, mode='auto', restore_best_weights=True)

    
if not FULL_TRAINING:

    history = model.fit(get_training_dataset(), steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, validation_data=get_validation_dataset(),callbacks = [lr_callback, es_callback],class_weight=class_weights)

else:

    history = model.fit(get_training_dataset(), steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, callbacks = [lr_callback],class_weight=class_weights)

    
# !mkdir -p saved_model


# model.save('saved_model/Flower_EFN.h5')
display_training_curves(history.history['loss'], history.history['val_loss'], 'loss', 211)

display_training_curves(history.history['sparse_categorical_accuracy'], history.history['val_sparse_categorical_accuracy'], 'accuracy', 212)
# with strategy.scope():

#     #pretrained_model = tf.keras.applications.DenseNet201(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])

#     pretrained_model_2 = tf.keras.applications.Xception(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])

#     #pretrained_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])

#     #pretrained_model  = EfficientNetB7(weights='noisy-student', include_top=False, input_shape=[*IMAGE_SIZE, 3])

#     pretrained_model_2.trainable = True # False = transfer learning, True = fine-tuning

    

#     model_2 = tf.keras.Sequential([

#         pretrained_model_2,

#         tf.keras.layers.GlobalAveragePooling2D(),

#         tf.keras.layers.Dense(len(CLASSES), activation='softmax')

#     ])

        

# model_2.compile(

#     tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False),

#     loss = 'sparse_categorical_crossentropy',

#     metrics=['sparse_categorical_accuracy']

# )

# model_2.summary()
# if not FULL_TRAINING:

#     history_2 = model_2.fit(get_training_dataset(), steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, validation_data=get_validation_dataset(),callbacks = [lr_callback, es_callback])

# else:

#     history_2 = model_2.fit(get_training_dataset(), steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, callbacks = [lr_callback])
# model_2.save('Flower_XCE.h5')
# print("Number of layers in the base model: ", len(pretrained_model_2.layers))

#pretrained_model.summary()
# pretrained_model_2.summary()
# pretrained_model_2.trainable = True

# for layer in pretrained_model_2.layers[:-10]:

#     layer.trainable = False

# for layer in pretrained_model_2.layers:

#     print(layer, layer.trainable)
# model_2.compile( tf.keras.optimizers.Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999, amsgrad=False), loss = 'sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'] ) 

# model_2.summary()
# fine_tune_epochs = 50

# total_epochs = EPOCHS + fine_tune_epochs

# history_2_fine = model_2.fit(get_training_dataset(), steps_per_epoch=STEPS_PER_EPOCH, epochs=total_epochs,initial_epoch = history_2.epoch[-1], validation_data=get_validation_dataset(),callbacks = [lr_callback, es_callback])
with strategy.scope():

    pretrained_model_3 = tf.keras.applications.DenseNet201(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])

    #pretrained_model_2 = tf.keras.applications.Xception(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])

    #pretrained_model_4 = tf.keras.applications.VGG16(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])

    #pretrained_model_1  = EfficientNetB7(weights='noisy-student', include_top=False, input_shape=[*IMAGE_SIZE, 3])

    pretrained_model_3.trainable = True # False = transfer learning, True = fine-tuning

    

    model_3 = tf.keras.Sequential([

        pretrained_model_3,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(len(CLASSES), activation='softmax')

    ])

        

model_3.compile(

    tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False),

    loss = 'sparse_categorical_crossentropy',

    metrics=['sparse_categorical_accuracy']

)

model_3.summary()
if not FULL_TRAINING:

    history_3 = model_3.fit(get_training_dataset(), steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, validation_data=get_validation_dataset(),callbacks = [lr_callback, es_callback], class_weight=class_weights)

else:

    history_3 = model_3.fit(get_training_dataset(), steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, callbacks = [lr_callback],class_weight=class_weights)
# model_3.save('Flower_Dense.h5')
# with strategy.scope():

#     #pretrained_model = tf.keras.applications.DenseNet201(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])

#     #pretrained_model = tf.keras.applications.Xception(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])

#     pretrained_model_4 = tf.keras.applications.VGG16(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])

#     #pretrained_model  = EfficientNetB7(weights='noisy-student', include_top=False, input_shape=[*IMAGE_SIZE, 3])

#     pretrained_model.trainable = False # False = transfer learning, True = fine-tuning

    

#     model_4 = tf.keras.Sequential([

#         pretrained_model_4,

#         tf.keras.layers.GlobalAveragePooling2D(),

#         tf.keras.layers.Dense(len(CLASSES), activation='softmax')

#     ])

        

# model_4.compile(

#     tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False),

#     loss = 'sparse_categorical_crossentropy',

#     metrics=['sparse_categorical_accuracy']

# )

# model_4.summary()
# if not FULL_TRAINING:

#     history_4 = model_4.fit(get_training_dataset(), steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, validation_data=get_validation_dataset(),callbacks = [es_callback,lr_callback])

# else:

#     history_4 = model_4.fit(get_training_dataset(), steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS,callbacks = [lr_callback])
if not FULL_TRAINING:

    cmdataset = get_validation_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and labels, order matters.

    images_ds = cmdataset.map(lambda image, label: image)

    labels_ds = cmdataset.map(lambda image, label: label).unbatch()

    cm_correct_labels = next(iter(labels_ds.batch(NUM_VALIDATION_IMAGES))).numpy() # get everything as one batch

    # cm_probabilities = (model_4.predict(images_ds)*0.1)+(model_3.predict(images_ds)*0.35)+(model_2.predict(images_ds)*0.25)+(model.predict(images_ds)*0.4)

    pred_score=[]

    dense_pred=model_3.predict(images_ds)

    efn_pred=model.predict(images_ds)

    for model_wt in np.linspace(0,1,100):

        cm_probabilities= model_wt*dense_pred +(1-model_wt)*efn_pred

        cm_predictions = np.argmax(cm_probabilities, axis=-1)

        pred_score.append(f1_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro'))

    #cm_probabilities = (model_3.predict(images_ds)*0.65)+(model.predict(images_ds)*0.35)

    #cm_predictions = np.argmax(cm_probabilities, axis=-1)

    plt.plot(pred_score)

    best_model_wt=np.argmax(pred_score)/100

    cm_probabilities= best_model_wt*dense_pred +(1-best_model_wt)*efn_pred

    cm_predictions = np.argmax(cm_probabilities, axis=-1)

    print("Correct   labels: ", cm_correct_labels.shape, cm_correct_labels)

    print("Predicted labels: ", cm_predictions.shape, cm_predictions)

else:

    best_model_wt=0.56
print(f'Best model weight {best_model_wt}')
cmat = confusion_matrix(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)))

score = f1_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')

precision = precision_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')

recall = recall_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')

cmat = (cmat.T / cmat.sum(axis=1)).T # normalized

display_confusion_matrix(cmat, score, precision, recall)

print('f1 score: {:.3f}, precision: {:.3f}, recall: {:.3f}'.format(score, precision, recall))
test_ds = get_test_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and ids, order matters.



print('Computing predictions...')

test_images_ds = test_ds.map(lambda image, idnum: image)

# probabilities =(model_4.predict(test_ds)*0.1)+(model_3.predict(test_ds)*0.3)+(model_2.predict(test_ds)*0.2)+(model.predict(test_ds)*0.4)

probabilities =(model_3.predict(test_ds)*best_model_wt)+(model.predict(test_ds)*best_model_wt)

predictions = np.argmax(probabilities, axis=-1)

print(predictions)



print('Generating submission.csv file...')

test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()

test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch

np.savetxt('submission.csv', np.rec.fromarrays([test_ids, predictions]), fmt=['%s', '%d'], delimiter=',', header='id,label', comments='')

!head submission.csv
# dataset = get_validation_dataset()

# dataset = dataset.unbatch().batch(20)

# batch = iter(dataset)
# # run this cell again for next set of images

# images, labels = next(batch)

# probabilities = model.predict(images)

# predictions = np.argmax(probabilities, axis=-1)

# display_batch_of_images((images, labels), predictions)