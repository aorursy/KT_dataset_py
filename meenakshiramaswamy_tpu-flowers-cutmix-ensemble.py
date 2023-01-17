!pip install -q efficientnet
#!pip install keras-rectified-adam
import efficientnet.tfkeras as efn

from tensorflow.keras.applications import InceptionResNetV2

from tensorflow.keras.applications import ResNet152V2

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

#import keras_radam

#from keras_radam import RAdam
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

    #for filename in filenames:

        #print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from IPython.display import display

from IPython.core.interactiveshell import InteractiveShell

#InteractiveShell.ast_node_interactivity = "all"

from sklearn.model_selection import KFold

import math, re

import tensorflow as tf

import numpy as np

from matplotlib import pyplot as plt

from kaggle_datasets import KaggleDatasets

print("Tensorflow version " + tf.__version__)

AUTO = tf.data.experimental.AUTOTUNE
# Data access

GCS_DS_PATH = KaggleDatasets().get_gcs_path('flower-classification-with-tpus')

# Configuration

IMAGE_SIZE = [512,512]

#PATH2 = KaggleDatasets().get_gcs_path('tf-flower-photo-tfrec')
'''

External dataset is disallowed in this competition

#When using external DS

GCS_PATH_SELECT = { # available image sizes

    192: GCS_DS_PATH + '/tfrecords-jpeg-192x192',

    224: GCS_DS_PATH + '/tfrecords-jpeg-224x224',

    331: GCS_DS_PATH + '/tfrecords-jpeg-331x331',

    512: GCS_DS_PATH + '/tfrecords-jpeg-512x512'

}

GCS_PATH = GCS_PATH_SELECT[IMAGE_SIZE[0]]



TRAINING_FILENAMES = tf.io.gfile.glob(PATH2 + '/imagenet/tfrecords-jpeg-512x512/*.tfrec')

TRAINING_FILENAMES = TRAINING_FILENAMES + tf.io.gfile.glob(PATH2 + '/inaturalist_1/tfrecords-jpeg-512x512/*.tfrec')

TRAINING_FILENAMES = TRAINING_FILENAMES + tf.io.gfile.glob(PATH2 + '/openimage/tfrecords-jpeg-512x512/*.tfrec')

TRAINING_FILENAMES = TRAINING_FILENAMES + tf.io.gfile.glob(PATH2 + '/oxford_102/tfrecords-jpeg-512x512/*.tfrec')

TRAINING_FILENAMES = TRAINING_FILENAMES + tf.io.gfile.glob(PATH2 + '/tf_flowers/tfrecords-jpeg-512x512/*.tfrec')



VALIDATION_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/val/*.tfrec')

TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/test/*.tfrec') # predictions on this dataset should be submitted for the competition



# watch out for overfitting!

SKIP_VALIDATION = False

if SKIP_VALIDATION:

    TRAINING_FILENAMES = TRAINING_FILENAMES + VALIDATION_FILENAMES

    

'''  


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

    

 
len(TRAINING_FILENAMES)

len(VALIDATION_FILENAMES)

len(TEST_FILENAMES)
CLASSES = ['pink primrose',

           'hard-leaved pocket orchid',

           'canterbury bells', 

           'sweet pea',    

           'wild geranium',  

           'tiger lily',       

           'moon orchid',     

           'bird of paradise',

           'monkshood',     

           'globe thistle',         # 00 - 09

           'snapdragon', 

           "colt's foot",      

           'king protea',   

           'spear thistle',

           'yellow iris',     

           'globe-flower',  

           'purple coneflower',  

           'peruvian lily',   

           'balloon flower',  

           'giant white arum lily', # 10 - 19

           'fire lily',  

           'pincushion flower',  

           'fritillary',    

           'red ginger', 

           'grape hyacinth', 

           'corn poppy',     

           'prince of wales feathers',

           'stemless gentian',

           'artichoke',       

           'sweet william',         # 20 - 29

           'carnation',   

           'garden phlox',     

           'love in the mist',

           'cosmos',       

           'alpine sea holly',

           'ruby-lipped cattleya',

           'cape flower',        

           'great masterwort', 

           'siam tulip',      

           'lenten rose',           # 30 - 39

           'barberton daisy',

           'daffodil',        

           'sword lily',     

           'poinsettia',   

           'bolero deep blue',  

           'wallflower',       

           'marigold',         

           'buttercup',       

           'daisy',        

           'common dandelion',      # 40 - 49

           'petunia',     

           'wild pansy',        

           'primula',        

           'sunflower',     

           'lilac hibiscus',  

           'bishop of llandaff', 

           'gaura',              

           'geranium',       

           'orange dahlia',  

           'pink-yellow dahlia',    # 50 - 59

           'cautleya spicata',

           'japanese anemone', 

           'black-eyed susan',

           'silverbush',  

           'californian poppy',

           'osteospermum',   

           'spring crocus',  

           'iris',       

           'windflower',   

           'tree poppy',            # 60 - 69

           'gazania',    

           'azalea',   

           'water lily', 

           'rose',          

           'thorn apple',   

           'morning glory',  

           'passion flower',  

           'lotus',           

           'toad lily',      

           'anthurium',             # 70 - 79

           'frangipani', 

           'clematis',      

           'hibiscus',      

           'columbine',   

           'desert-rose',     

           'tree mallow',   

           'magnolia',       

           'cyclamen ',      

           'watercress',     

           'canna lily',            # 80 - 89

           'hippeastrum ', 

           'bee balm',       

           'pink quill',     

           'foxglove',    

           'bougainvillea', 

           'camellia',      

           'mallow',          

           'mexican petunia', 

           'bromelia',         

           'blanket flower',        # 90 - 99

           'trumpet creeper', 

           'blackberry lily',   

           'common tulip',    

           'wild rose']                                                                                                                                               # 100 - 102
MIXED_PRECISION = False

XLA_ACCELERATE = False



if MIXED_PRECISION:

    from tensorflow.keras.mixed_precision import experimental as mixed_precision

    if tpu: policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')

    else: policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')

    mixed_precision.set_policy(policy)

    print('Mixed precision enabled')



if XLA_ACCELERATE:

    tf.config.optimizer.set_jit(True)

    print('Accelerated Linear Algebra enabled')
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
EPOCHS = 20
# Learning rate schedule for TPU, GPU and CPU.

# Using an LR ramp up because fine-tuning a pre-trained model.

# Starting with a high LR would break the pre-trained weights.



LR_START = 0.00001

LR_MAX = 0.00005 * strategy.num_replicas_in_sync

LR_MIN = 0.00001

LR_RAMPUP_EPOCHS = 4

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

    

lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose = True)



rng = [i for i in range(50 if EPOCHS<50 else EPOCHS)]

y = [lrfn(x) for x in rng]

plt.plot(rng, y)

print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))
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



def data_augment(image, label, seed=2020):

    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),

    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part

    # of the TPU while the TPU itself is computing gradients.

    image = tf.image.random_flip_left_right(image, seed=seed)

#     image = tf.image.random_flip_up_down(image, seed=seed)

#     image = tf.image.random_brightness(image, 0.1, seed=seed)

    

#     image = tf.image.random_jpeg_quality(image, 85, 100, seed=seed)

#     image = tf.image.resize(image, [530, 530])

#     image = tf.image.random_crop(image, [512, 512], seed=seed)

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



def get_training_dataset(dataset, do_aug=True):

    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)

    dataset = dataset.repeat() # the training dataset must repeat for several epochs

    dataset = dataset.batch(AUG_BATCH)

    if do_aug: dataset = dataset.map(transform, num_parallel_calls=AUTO) # note we put AFTER batching

    dataset = dataset.unbatch()

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



def get_train_valid_datasets():

    dataset = load_dataset(TRAINING_FILENAMES + VALIDATION_FILENAMES, labeled=True)

    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)

    dataset = dataset.repeat() # the training dataset must repeat for several epochs

    dataset = dataset.shuffle(2048)

    dataset = dataset.batch(BATCH_SIZE)

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
#EPOCHS = 15

BATCH_SIZE = 8 * strategy.num_replicas_in_sync

AUG_BATCH = BATCH_SIZE

NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)

NUM_VALIDATION_IMAGES = count_data_items(VALIDATION_FILENAMES)

NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

print('Dataset: {} training images, {} validation images, {} unlabeled test images'.format(NUM_TRAINING_IMAGES, NUM_VALIDATION_IMAGES, NUM_TEST_IMAGES))

#AUTO = tf.data.experimental.AUTOTUNE
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



def display_confusion_matrix(cmat, score, precision, recall):

    plt.figure(figsize=(15,15))

    ax = plt.gca()

    ax.matshow(cmat, cmap='BuPu')

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
# Peek at training data

dataset = load_dataset(TRAINING_FILENAMES, labeled=True)

training_dataset = get_training_dataset(dataset, do_aug=False)

training_dataset = training_dataset.unbatch().batch(20)

train_batch = iter(training_dataset)
print("Training data shapes:")

for image, label in get_training_dataset(dataset, do_aug=False).take(2):

    print(image.numpy().shape, label.numpy().shape)

    train_image = image.numpy()

    train_label = label.numpy()

print("Training data label examples:", label.numpy())

print("Validation data shapes:")

for image, label in get_validation_dataset().take(2):

    print(image.numpy().shape, label.numpy().shape)

print("Validation data label examples:", label.numpy())

print("Test data shapes:")

for image, idnum in get_test_dataset().take(2):

    print(image.numpy().shape, idnum.numpy().shape)

print("Test data IDs:", idnum.numpy().astype('U')) # U=unicode string
# Peek at training data

valid_dataset = get_validation_dataset()

valid_dataset = valid_dataset.unbatch().batch(20)

val_batch = iter(valid_dataset)
# run this cell again for next set of images

display_batch_of_images(next(train_batch))
# run this cell again for next set of images

display_batch_of_images(next(val_batch))
# peer at test data

test_dataset = get_test_dataset()

test_dataset = test_dataset.unbatch().batch(20)

test_batch = iter(test_dataset)
# run this cell again for next set of images

display_batch_of_images(next(test_batch))

def onehot(image,label):

    CLASSES = 104

    return image,tf.one_hot(label,CLASSES)
def cutmix(image, label, PROBABILITY = 1.0):

    # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]

    # output - a batch of images with cutmix applied

    DIM = IMAGE_SIZE[0]

    CLASSES = 104

    

    imgs = []; labs = []

    for j in range(AUG_BATCH):

        # DO CUTMIX WITH PROBABILITY DEFINED ABOVE

        P = tf.cast( tf.random.uniform([],0,1)<=PROBABILITY, tf.int32)

        # CHOOSE RANDOM IMAGE TO CUTMIX WITH

        k = tf.cast( tf.random.uniform([],0,AUG_BATCH),tf.int32)

        # CHOOSE RANDOM LOCATION

        x = tf.cast( tf.random.uniform([],0,DIM),tf.int32)

        y = tf.cast( tf.random.uniform([],0,DIM),tf.int32)

        b = tf.random.uniform([],0,1) # this is beta dist with alpha=1.0

        WIDTH = tf.cast( DIM * tf.math.sqrt(1-b),tf.int32) * P

        ya = tf.math.maximum(0,y-WIDTH//2)

        yb = tf.math.minimum(DIM,y+WIDTH//2)

        xa = tf.math.maximum(0,x-WIDTH//2)

        xb = tf.math.minimum(DIM,x+WIDTH//2)

        # MAKE CUTMIX IMAGE

        one = image[j,ya:yb,0:xa,:]

        two = image[k,ya:yb,xa:xb,:]

        three = image[j,ya:yb,xb:DIM,:]

        middle = tf.concat([one,two,three],axis=1)

        img = tf.concat([image[j,0:ya,:,:],middle,image[j,yb:DIM,:,:]],axis=0)

        imgs.append(img)

        # MAKE CUTMIX LABEL

        a = tf.cast(WIDTH*WIDTH/DIM/DIM,tf.float32)

        if len(label.shape)==1:

            lab1 = tf.one_hot(label[j],CLASSES)

            lab2 = tf.one_hot(label[k],CLASSES)

        else:

            lab1 = label[j,]

            lab2 = label[k,]

        labs.append((1-a)*lab1 + a*lab2)

            

    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)

    image2 = tf.reshape(tf.stack(imgs),(AUG_BATCH,DIM,DIM,3))

    label2 = tf.reshape(tf.stack(labs),(AUG_BATCH,CLASSES))

    return image2,label2
row = 6; col = 4;

row = min(row,AUG_BATCH//col)

all_elements = get_training_dataset(load_dataset(TRAINING_FILENAMES),do_aug=False).unbatch()

augmented_element = all_elements.repeat().batch(AUG_BATCH).map(cutmix)



for (img,label) in augmented_element:

    plt.figure(figsize=(15,int(15*row/col)))

    for j in range(row*col):

        plt.subplot(row,col,j+1)

        plt.axis('off')

        plt.imshow(img[j,])

    plt.show()

    break
def mixup(image, label, PROBABILITY = 1.0):

    # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]

    # output - a batch of images with mixup applied

    DIM = IMAGE_SIZE[0]

    CLASSES = 104

    

    imgs = []; labs = []

    for j in range(AUG_BATCH):

        # DO MIXUP WITH PROBABILITY DEFINED ABOVE

        P = tf.cast( tf.random.uniform([],0,1)<=PROBABILITY, tf.float32)

        # CHOOSE RANDOM

        k = tf.cast( tf.random.uniform([],0,AUG_BATCH),tf.int32)

        a = tf.random.uniform([],0,1)*P # this is beta dist with alpha=1.0

        # MAKE MIXUP IMAGE

        img1 = image[j,]

        img2 = image[k,]

        imgs.append((1-a)*img1 + a*img2)

        # MAKE CUTMIX LABEL

        if len(label.shape)==1:

            lab1 = tf.one_hot(label[j],CLASSES)

            lab2 = tf.one_hot(label[k],CLASSES)

        else:

            lab1 = label[j,]

            lab2 = label[k,]

        labs.append((1-a)*lab1 + a*lab2)

            

    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)

    image2 = tf.reshape(tf.stack(imgs),(AUG_BATCH,DIM,DIM,3))

    label2 = tf.reshape(tf.stack(labs),(AUG_BATCH,CLASSES))

    return image2,label2
def transform(image,label):

    # THIS FUNCTION APPLIES BOTH CUTMIX AND MIXUP

    DIM = IMAGE_SIZE[0]

    CLASSES = 104

    SWITCH = 0.5

    CUTMIX_PROB = 0.666

    MIXUP_PROB = 0.666

    # FOR SWITCH PERCENT OF TIME WE DO CUTMIX AND (1-SWITCH) WE DO MIXUP

    image2, label2 = cutmix(image, label, CUTMIX_PROB)

    image3, label3 = mixup(image, label, MIXUP_PROB)

    imgs = []; labs = []

    for j in range(AUG_BATCH):

        P = tf.cast( tf.random.uniform([],0,1)<=SWITCH, tf.float32)

        imgs.append(P*image2[j,]+(1-P)*image3[j,])

        labs.append(P*label2[j,]+(1-P)*label3[j,])

    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)

    image4 = tf.reshape(tf.stack(imgs),(AUG_BATCH,DIM,DIM,3))

    label4 = tf.reshape(tf.stack(labs),(AUG_BATCH,CLASSES))

    return image4,label4
row = 6; col = 4;

row = min(row,AUG_BATCH//col)

all_elements = get_training_dataset(load_dataset(TRAINING_FILENAMES),do_aug=False).unbatch()

augmented_element = all_elements.repeat().batch(AUG_BATCH).map(mixup)



for (img,label) in augmented_element:

    plt.figure(figsize=(15,int(15*row/col)))

    for j in range(row*col):

        plt.subplot(row,col,j+1)

        plt.axis('off')

        plt.imshow(img[j,])

    plt.show()

    break
row = 6; col = 4;

row = min(row,AUG_BATCH//col)

all_elements = get_training_dataset(load_dataset(TRAINING_FILENAMES),do_aug=False).unbatch()

augmented_element = all_elements.repeat().batch(AUG_BATCH).map(transform)



for (img,label) in augmented_element:

    plt.figure(figsize=(15,int(15*row/col)))

    for j in range(row*col):

        plt.subplot(row,col,j+1)

        plt.axis('off')

        plt.imshow(img[j,])

    plt.show()

    break
'''

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

weight_path="{}_tpu_weights.h5".format('flower_classify')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 

                             save_best_only=True, mode='min', save_weights_only = True)





reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=3, verbose=1, mode='auto', 

                                   min_delta=0.01, cooldown=3, min_lr=0.0001)

early = EarlyStopping(monitor="val_loss", 

                      mode="min",restore_best_weights=True, 

                      patience=10) # probably needs to be more patient

callbacks_list = [checkpoint, early, reduceLROnPlat]

'''
def call_model_efn():

    enet = efn.EfficientNetB7(

        input_shape=(512, 512, 3),

        weights = 'noisy-student', #weights='imagenet',

        include_top=False

    )



    enet.trainable = True



    model = tf.keras.Sequential([

        enet,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(len(CLASSES), activation='softmax')

    ])

    

    #opt = RAdam(total_steps=5000, warmup_proportion=0.1, min_lr=1e-5)

    

    model.compile(

        optimizer='adam',

        loss = 'sparse_categorical_crossentropy',

        metrics=['sparse_categorical_accuracy']

        #tf.keras.optimizers.Adam(lr=1e-3),

        #loss = 'sparse_categorical_crossentropy',

        #metrics=['sparse_categorical_accuracy']

    )

    return model
'''

train_inputs, train_labels = batch_to_numpy_images_and_labels(next(train_batch))

val_inputs, val_labels = batch_to_numpy_images_and_labels(next(val_batch))



inputs = np.concatenate((train_inputs, val_inputs), axis=0)

targets = np.concatenate((train_labels, val_labels), axis=0)



#print (inputs, targets)

'''
#weights_model='/kaggle/input/tpu-flowers-kfold-cv/flower_classify_tpu_weights.h5'
'''

num_folds = 2



# Define per-fold score containers

acc_per_fold = []

loss_per_fold = []





# Define the K-fold Cross Validator

kfold = KFold(n_splits=num_folds, shuffle=True, random_state = 42)



# K-fold Cross Validation model evaluation

fold_no = 1

for train, val in kfold.split(inputs, targets):

    with strategy.scope():

        enet_model = call_model_efn()

    enet_model.summary()

    #enet_model.load_weights(weights_model)

    # Generate a print

    print('------------------------------------------------------------------------')

    print(f'Training for fold {fold_no} ...')

    print('------------------------------------------------------------------------')

    # scheduler = tf.keras.callbacks.ReduceLROnPlateau(patience=3, verbose=1)

    #lr_schedule = tf.keras.callbacks.LearningRateScheduler(reduceLROnPlat, verbose=1)

    history_enet = enet_model.fit(

        #get_training_dataset(),#get_train_valid_datasets(),

        get_training_dataset(load_dataset(TRAINING_FILENAMES),do_aug=False),

        steps_per_epoch=STEPS_PER_EPOCH,

        epochs=EPOCHS, 

        callbacks = [lr_callback],

        #callbacks=[checkpoint, early, reduceLROnPlat],

        validation_data = get_validation_dataset(),

        verbose=2

    )

    # Generate generalization metrics

    scores = enet_model.evaluate(get_validation_dataset())

    print(f'Score for fold {fold_no}: {enet_model.metrics_names[0]} of {scores[0]}; {enet_model.metrics_names[1]} of {scores[1]*100}%')

    acc_per_fold.append(scores[1] * 100)

    loss_per_fold.append(scores[0])

    # Increase fold number

    fold_no = fold_no + 1

    

    

# == Provide average scores ==

print('\n------------------------------------------------------------------------')

print('Score per fold')

for i in range(0, len(acc_per_fold)):

  print('------------------------------------------------------------------------')

  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')

print('------------------------------------------------------------------------')

print('Average scores for all folds:')

print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')

print(f'> Loss: {np.mean(loss_per_fold)}')

print('------------------------------------------------------------------------')    

'''


#Skip this cell if you use k-fold

#use for ensemble

with strategy.scope():

    enet_model = call_model_efn()

enet_model.summary()



history_enet = enet_model.fit(

    #get_training_dataset(),#get_train_valid_datasets(),

    get_training_dataset(load_dataset(TRAINING_FILENAMES),do_aug=False),

    steps_per_epoch=STEPS_PER_EPOCH,

    epochs=EPOCHS, 

    #callbacks=[checkpoint, early, reduceLROnPlat],

    callbacks = [lr_callback],

    validation_data = get_validation_dataset(),

    verbose=2

)



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

    

display_training_curves(history_enet.history['loss'], history_enet.history['val_loss'], 'loss', 211)

display_training_curves(history_enet.history['sparse_categorical_accuracy'], history_enet.history['val_sparse_categorical_accuracy'], 'accuracy', 212)    
cmdataset = get_validation_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and labels, order matters.

images_ds = cmdataset.map(lambda image, label: image)

labels_ds = cmdataset.map(lambda image, label: label).unbatch()

cm_correct_labels = next(iter(labels_ds.batch(NUM_VALIDATION_IMAGES))).numpy() # get everything as one batch

cm_probabilities = enet_model.predict(images_ds)

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
test_ds = get_test_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and ids, order matters.



print('Computing predictions...')

test_images_ds = test_ds.map(lambda image, idnum: image)

enet_probabilities = enet_model.predict(test_images_ds)

enet_predictions = np.argmax(enet_probabilities, axis=-1)

print(enet_predictions)

'''

print('Generating submission.csv file...')

test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()

test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch

np.savetxt('submission.csv', np.rec.fromarrays([test_ids, enet_predictions]), fmt=['%s', '%d'], delimiter=',', header='id,label', comments='')

'''
'''

# run this cell again for next set of images

images, labels = next(val_batch)

enet_probabilities = enet_model.predict(images)

enet_predictions = np.argmax(enet_probabilities, axis=-1)

display_batch_of_images((images, labels), enet_predictions)

'''
#EPOCHS = 20


def call_model_rn152v2():

    rnet = ResNet152V2(

        input_shape=(512,512,3),

        weights = 'imagenet', #weights='imagenet',

        include_top=False

    )



    rnet.trainable = True



    model = tf.keras.Sequential([

        rnet,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(len(CLASSES), activation='softmax')

    ])

    

    #opt = RAdam(total_steps=5000, warmup_proportion=0.1, min_lr=1e-5)

    

    model.compile(

        #optimizer=tf.optimizers.RectifiedAdam(),

        #tf.keras.optimizers.Adam(lr=1e-5),

        optimizer = 'adam',

        loss = 'sparse_categorical_crossentropy',

        metrics=['sparse_categorical_accuracy']

    )

    return model



with strategy.scope():

    rnet_model = call_model_rn152v2()

rnet_model.summary()



history_resnet = rnet_model.fit(

    #get_training_dataset(),#get_train_valid_datasets(),

    get_training_dataset(load_dataset(TRAINING_FILENAMES),do_aug=False),

    steps_per_epoch=STEPS_PER_EPOCH,

    epochs=EPOCHS, 

    #callbacks=[checkpoint, early, reduceLROnPlat],

    callbacks = [lr_callback], 

    validation_data = get_validation_dataset(),

    verbose=2

)



display_training_curves(history_resnet.history['loss'], history_resnet.history['val_loss'], 'loss', 211)

display_training_curves(history_resnet.history['sparse_categorical_accuracy'], history_resnet.history['val_sparse_categorical_accuracy'], 'accuracy', 212) 



cmdataset = get_validation_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and labels, order matters.

images_ds = cmdataset.map(lambda image, label: image)

labels_ds = cmdataset.map(lambda image, label: label).unbatch()

rm_correct_labels = next(iter(labels_ds.batch(NUM_VALIDATION_IMAGES))).numpy() # get everything as one batch

rm_probabilities = rnet_model.predict(images_ds)

rm_predictions = np.argmax(rm_probabilities, axis=-1)

print("Correct   labels: ", rm_correct_labels.shape, cm_correct_labels)

print("Predicted labels: ", rm_predictions.shape, cm_predictions)



rmat = confusion_matrix(rm_correct_labels, rm_predictions, labels=range(len(CLASSES)))

score = f1_score(rm_correct_labels, rm_predictions, labels=range(len(CLASSES)), average='macro')

precision = precision_score(rm_correct_labels, rm_predictions, labels=range(len(CLASSES)), average='macro')

recall = recall_score(rm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')

rmat = (rmat.T / rmat.sum(axis=1)).T # normalized

display_confusion_matrix(rmat, score, precision, recall)

print('f1 score: {:.3f}, precision: {:.3f}, recall: {:.3f}'.format(score, precision, recall))



test_ds = get_test_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and ids, order matters.



print('Computing predictions...')

test_images_ds = test_ds.map(lambda image, idnum: image)

rnet_probabilities = rnet_model.predict(test_images_ds)

rnet_predictions = np.argmax(rnet_probabilities, axis=-1)

print(rnet_predictions)



def call_model_IRv2():

    irnet = InceptionResNetV2(

        input_shape=(512,512,3),

        weights = 'imagenet', #weights='imagenet',

        include_top=False

    )



    irnet.trainable = True



    model = tf.keras.Sequential([

        irnet,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(len(CLASSES), activation='softmax')

    ])

    

    #opt = RAdam(total_steps=5000, warmup_proportion=0.1, min_lr=1e-5)

    

    model.compile(

        #optimizer=tf.optimizers.RectifiedAdam(),

        #tf.keras.optimizers.Adam(lr=1e-5),

        optimizer = 'adam',

        loss = 'sparse_categorical_crossentropy',

        metrics=['sparse_categorical_accuracy']

    )

    return model



with strategy.scope():

    irnet_model = call_model_IRv2()

irnet_model.summary()



history_incepresnet = irnet_model.fit(

    #get_training_dataset(),#get_train_valid_datasets(),

    get_training_dataset(load_dataset(TRAINING_FILENAMES),do_aug=False),

    steps_per_epoch=STEPS_PER_EPOCH,

    epochs=EPOCHS, 

    callbacks = [lr_callback],

    #callbacks=[checkpoint, early, reduceLROnPlat],

    validation_data = get_validation_dataset(),

    verbose=2

)



display_training_curves(history_incepresnet.history['loss'], history_incepresnet.history['val_loss'], 'loss', 211)

display_training_curves(history_incepresnet.history['sparse_categorical_accuracy'], history_incepresnet.history['val_sparse_categorical_accuracy'], 'accuracy', 212)   



cmdataset = get_validation_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and labels, order matters.

images_ds = cmdataset.map(lambda image, label: image)

labels_ds = cmdataset.map(lambda image, label: label).unbatch()

cm_correct_labels = next(iter(labels_ds.batch(NUM_VALIDATION_IMAGES))).numpy() # get everything as one batch

cm_probabilities = irnet_model.predict(images_ds)

cm_predictions = np.argmax(cm_probabilities, axis=-1)

print("Correct   labels: ", cm_correct_labels.shape, cm_correct_labels)

print("Predicted labels: ", cm_predictions.shape, cm_predictions)



cmat = confusion_matrix(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)))

rscore = f1_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')

rprecision = precision_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')

rrecall = recall_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')

cmat = (cmat.T / cmat.sum(axis=1)).T # normalized

display_confusion_matrix(cmat, rscore, rprecision, rrecall)

print('f1 score: {:.3f}, precision: {:.3f}, recall: {:.3f}'.format(rscore, rprecision, rrecall))



test_ds = get_test_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and ids, order matters.



print('Computing predictions...')

test_images_ds = test_ds.map(lambda image, idnum: image)

irnet_probabilities = irnet_model.predict(test_images_ds)

irnet_predictions = np.argmax(irnet_probabilities, axis=-1)

print(irnet_predictions)

from tensorflow.keras.applications import DenseNet201


def call_model_Dnet201():

    dnet = DenseNet201(

        input_shape=(512,512,3),

        weights='imagenet',

        include_top=False

    )



    dnet.trainable = True



    model = tf.keras.Sequential([

        dnet,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(len(CLASSES), activation='softmax')

    ])

    

    #opt = RAdam(total_steps=5000, warmup_proportion=0.1, min_lr=1e-5)

    

    model.compile(

        #optimizer=tf.optimizers.RectifiedAdam(),

        #tf.keras.optimizers.Adam(),

        optimizer = 'adam',

        loss = 'sparse_categorical_crossentropy',

        metrics=['sparse_categorical_accuracy']

    )

    return model



with strategy.scope():

    dnet_model = call_model_Dnet201()

dnet_model.summary()



history_dnet = dnet_model.fit(

    #get_training_dataset(),#get_train_valid_datasets(),

    get_training_dataset(load_dataset(TRAINING_FILENAMES),do_aug=False),

    steps_per_epoch=STEPS_PER_EPOCH,

    epochs=EPOCHS, 

    callbacks = [lr_callback],

    #callbacks=[checkpoint, early, reduceLROnPlat],

    validation_data = get_validation_dataset(),

    verbose=2

)



display_training_curves(history_dnet.history['loss'], history_dnet.history['val_loss'], 'loss', 211)

display_training_curves(history_dnet.history['sparse_categorical_accuracy'], history_dnet.history['val_sparse_categorical_accuracy'], 'accuracy', 212) 



cmdataset = get_validation_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and labels, order matters.

images_ds = cmdataset.map(lambda image, label: image)

labels_ds = cmdataset.map(lambda image, label: label).unbatch()

cm_correct_labels = next(iter(labels_ds.batch(NUM_VALIDATION_IMAGES))).numpy() # get everything as one batch

cm_probabilities = dnet_model.predict(images_ds)

cm_predictions = np.argmax(cm_probabilities, axis=-1)

print("Correct   labels: ", cm_correct_labels.shape, cm_correct_labels)

print("Predicted labels: ", cm_predictions.shape, cm_predictions)



cmat = confusion_matrix(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)))

rscore = f1_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')

rprecision = precision_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')

rrecall = recall_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')

cmat = (cmat.T / cmat.sum(axis=1)).T # normalized

display_confusion_matrix(cmat, rscore, rprecision, rrecall)

print('f1 score: {:.3f}, precision: {:.3f}, recall: {:.3f}'.format(rscore, rprecision, rrecall))



test_ds = get_test_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and ids, order matters.



print('Computing predictions...')

test_images_ds = test_ds.map(lambda image, idnum: image)

dnet_probabilities = dnet_model.predict(test_images_ds)

dnet_predictions = np.argmax(dnet_probabilities, axis=-1)

print(dnet_predictions)

probabilities = (enet_probabilities + dnet_probabilities + irnet_probabilities + rnet_probabilities)/4

ensemble_predict = np.argmax(probabilities, axis=-1)

print(ensemble_predict)
'''

probabilities = np.mean(

    [

        enet_probabilities,

        dnet_probabilities

        ,irnet_probabilities

        ,rnet_probabilities

    ],

    axis=0

)



ensemble_predict = np.argmax(probabilities, axis=-1)

print(ensemble_predict)

'''


print('Generating submission.csv file...')

test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()

test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch



#ensemble_predict = (np.mean [enet_predictions , rnet_predictions, irnet_predictions,dnet_predictions ], axis = 0)

np.savetxt('submission.csv', np.rec.fromarrays([test_ids, ensemble_predict]), fmt=['%s', '%d'], delimiter=',', header='id,label', comments='')

print ('Submission Ensemble saved.....')
