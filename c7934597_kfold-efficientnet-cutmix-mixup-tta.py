#the basics

from matplotlib import pyplot as plt

import math, os, re

import numpy as np, pandas as pd



#deep learning basics

import tensorflow as tf

import tensorflow.keras.backend as K



#get current TensorFlow version fo

print("Currently using Tensorflow version " + tf.__version__)
DEVICE = 'TPU'   #or GPU



if DEVICE == "TPU":

    print("connecting to TPU...")

    try:

        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

        print('Running on TPU ', tpu.master())

    except ValueError:

        print("Could not connect to TPU")

        tpu = None



    if tpu:

        try:

            print("initializing  TPU ...")

            tf.config.experimental_connect_to_cluster(tpu)

            tf.tpu.experimental.initialize_tpu_system(tpu)

            strategy = tf.distribute.experimental.TPUStrategy(tpu)

            print("TPU initialized")

        except _:

            print("failed to initialize TPU")

    else:

        DEVICE = "GPU"



if DEVICE != "TPU":

    print("Using default strategy for CPU and single GPU")

    strategy = tf.distribute.get_strategy()



if DEVICE == "GPU":

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    



AUTO = tf.data.experimental.AUTOTUNE

REPLICAS = strategy.num_replicas_in_sync

#REPLICAS = 8

print(f'REPLICAS: {REPLICAS}')
#get GCS path for flower classification data set

from kaggle_datasets import KaggleDatasets

GCS_DS_PATH = KaggleDatasets().get_gcs_path('tpu-getting-started')
#can use this path in Google Colabs if you want to host notebook else where

print(GCS_DS_PATH)
#for reproducibility

SEED = 34     #my favorite number



#define image size we will use

#IMAGE_SIZE = [192, 192]               #if you aren't using TPU

#IMAGE_SIZE = [331, 331]               #middle ground

IMAGE_SIZE = [512, 512]               #if you are using TPU



#how many training samples we want going to TPUs 

BATCH_SIZE = 16 * strategy.num_replicas_in_sync 



#define aug batch size

AUG_BATCH = BATCH_SIZE



#how many folds we will use to train our model on

FOLDS = 3



#how many TTA steps to apply

TTA = 5



#list other options we have for image sizes

GCS_PATH_SELECT = {

    192: GCS_DS_PATH + '/tfrecords-jpeg-192x192',

    224: GCS_DS_PATH + '/tfrecords-jpeg-224x224',

    331: GCS_DS_PATH + '/tfrecords-jpeg-331x331',

    512: GCS_DS_PATH + '/tfrecords-jpeg-512x512'

}



#choose 512 image size for best performance

GCS_PATH = GCS_PATH_SELECT[IMAGE_SIZE[0]]
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

    

    #returns a dataset of (image, label) pairs

    return image, label



def read_unlabeled_tfrecord(example, return_image_name):

    UNLABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "id": tf.io.FixedLenFeature([], tf.string),  # [] means single entry

    }

    

    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    idnum = example['id']

    #returns a dataset of image(s)

    return image, idnum if return_image_name else 0



#some simply image augmentation we can perform with tf.image

def data_augment(image, label):

    

    #random augmentations

    image = tf.image.random_flip_left_right(image)

    #image = tf.image.random_flip_up_down(image)

    #image = tf.image.random_hue(image, 0.01)

    #image = tf.image.random_saturation(image, 0.7, 1.3)

    #image = tf.image.random_contrast(image, 0.8, 1.2)

    #image = tf.image.random_brightness(image, 0.1)

    

    #fixed augmentations

    #image = tf.image.adjust_saturation(image, max_delta = .2)

    #image = tf.image.central_crop(image, central_fraction = 0.5)

    return image, label  



def count_data_items(filenames):

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) 

         for filename in filenames]

    return np.sum(n)
#define pre fetching strategy

AUTO = tf.data.experimental.AUTOTUNE



#use tf.io.gfile.glob to find our training and test files from GCS bucket

TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/train/*.tfrec') + tf.io.gfile.glob(GCS_PATH + '/val/*.tfrec')

TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/test/*.tfrec')

    

#show item counts

NUM_TRAINING_IMAGES = int( count_data_items(TRAINING_FILENAMES) * (FOLDS-1.)/FOLDS )

NUM_VALIDATION_IMAGES = int( count_data_items(TRAINING_FILENAMES) * (1./FOLDS) )

NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE



print('Dataset: {} training images, {} validation images, {} unlabeled test images'.format(NUM_TRAINING_IMAGES, NUM_VALIDATION_IMAGES, NUM_TEST_IMAGES))
def get_train_ds(files, tta_aug = False, cutmix_aug = False, shuffle = True, 

                 repeat = True, labeled=True, return_image_names = True):

    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO)

    ds = ds.cache()

    

    if repeat:

        ds = ds.repeat()

    

    if shuffle: 

        ds = ds.shuffle(1024*8)

        opt = tf.data.Options()

        opt.experimental_deterministic = False

        ds = ds.with_options(opt)



    if labeled: 

        ds = ds.map(read_labeled_tfrecord, num_parallel_calls=AUTO)

    else:

        ds = ds.map(lambda example: read_unlabeled_tfrecord(example, return_image_names), 

                    num_parallel_calls=AUTO) 

    

    if tta_aug:

        ds = ds.map(data_augment, num_parallel_calls = AUTO)

        ds = ds.map(transform, num_parallel_calls=AUTO)

    

    if cutmix_aug: 

        #need to batch to use CutMix/mixup

        ds = ds.batch(AUG_BATCH)

        ds = ds.map(mixup_and_cutmix, num_parallel_calls=AUTO) # note we put AFTER batching

        

        #now unbatch and shuffle before re-batching

        ds = ds.unbatch()

        #ds = ds.shuffle(2048)

    

    #prefetch next batch while training

    ds = ds.batch(BATCH_SIZE)

    ds = ds.prefetch(AUTO)

    

    return ds





def get_val_ds(files, shuffle = True, labeled=True, return_image_names=False):

    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO)

    ds = ds.cache()  

    

    if shuffle: 

        ds = ds.shuffle(1024*8)

        opt = tf.data.Options()

        opt.experimental_deterministic = False

        ds = ds.with_options(opt)



    if labeled: 

        ds = ds.map(read_labeled_tfrecord, num_parallel_calls=AUTO)

    else:

        ds = ds.map(lambda example: read_unlabeled_tfrecord(example, return_image_names), 

                    num_parallel_calls=AUTO)

        

    

    #prefetch next batch while training

    ds = ds.batch(BATCH_SIZE)

    

    #we must one hot encode if we use CutMix or mixup

    ds = ds.map(onehot, num_parallel_calls=AUTO)

    ds = ds.prefetch(AUTO)

    

    return ds

#define flower classes for labeling purposes

classes = ['pink primrose',    'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea',     'wild geranium',     'tiger lily',           'moon orchid',              'bird of paradise', 'monkshood',        'globe thistle',         # 00 - 09

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
#numpy and matplotlib defaults

np.set_printoptions(threshold=15, linewidth=80)



def batch_to_numpy_images_and_labels(data):

    images, labels = data

    numpy_images = images.numpy()

    numpy_labels = labels.numpy()

    #binary strings are image IDs

    if numpy_labels.dtype == object:

        numpy_labels = [None for _ in enumerate(numpy_images)]

    #If no labels, only image IDs, return None for labels (this is the case for test data)

    return numpy_images, numpy_labels



def title_from_label_and_target(label, correct_label):

    if correct_label is None:

        return classes[label], True

    correct = (label == correct_label)

    return "{} [{}{}{}]".format(classes[label], 'OK' if correct else 'NO', u"\u2192" if not correct else '',

                                classes[correct_label] if not correct else ''), correct



def display_one_flower(image, title, subplot, red=False, titlesize=16):

    plt.subplot(*subplot)

    plt.axis('off')

    plt.imshow(image)

    if len(title) > 0:

        plt.title(title, fontsize=int(titlesize) if not red else int(titlesize/1.2), color='red' if red else 'black', fontdict={'verticalalignment':'center'}, pad=int(titlesize/1.5))

    return (subplot[0], subplot[1], subplot[2]+1)

    

def display_batch_of_images(databatch, predictions=None):

    # data

    images, labels = batch_to_numpy_images_and_labels(databatch)

    if labels is None:

        labels = [None for _ in enumerate(images)]

        

    #auto-squaring: this will drop data that does not fit into square or square-ish rectangle

    rows = int(math.sqrt(len(images)))

    cols = len(images)//rows

        

    #size and spacing

    FIGSIZE = 13.0

    SPACING = 0.1

    subplot=(rows,cols,1)

    if rows < cols:

        plt.figure(figsize=(FIGSIZE,FIGSIZE/cols*rows))

    else:

        plt.figure(figsize=(FIGSIZE/rows*cols,FIGSIZE))

    

    #display

    for i, (image, label) in enumerate(zip(images[:rows*cols], labels[:rows*cols])):

        title = '' if label is None else classes[label]

        correct = True

        if predictions is not None:

            title, correct = title_from_label_and_target(predictions[i], label)

        dynamic_titlesize = FIGSIZE*SPACING/max(rows,cols)*40+3 # magic formula tested to work from 1x1 to 10x10 images

        subplot = display_one_flower(image, title, subplot, not correct, titlesize=dynamic_titlesize)

    

    #get optimal spacing

    plt.tight_layout()

    if label is None and predictions is None:

        plt.subplots_adjust(wspace=0, hspace=0)

    else:

        plt.subplots_adjust(wspace=SPACING, hspace=SPACING)

    plt.show()
#first look at training dataset

training_dataset = get_train_ds(TRAINING_FILENAMES, cutmix_aug = False, tta_aug = False, labeled = True,

                                        shuffle = True, repeat = True)

training_dataset = training_dataset.unbatch().batch(20)

train_batch = iter(training_dataset)
#first look at test dataset

test_dataset = get_train_ds(TEST_FILENAMES, labeled = False, shuffle = True, repeat = False)

test_dataset = test_dataset.unbatch().batch(20)

test_batch = iter(test_dataset)
#view batch of flowers from train

display_batch_of_images(next(train_batch))

#you can run this cell again and it will load a new batch
#view batch of flowers from test

display_batch_of_images(next(test_batch))

#you can run this cell again and it will load a new batch
def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):

    # returns 3x3 transformmatrix which transforms indicies

    

    rotation = math.pi * rotation / 180.

    shear = math.pi * shear / 180.

    

    c1 = tf.math.cos(rotation)

    s1 = tf.math.sin(rotation)

    one = tf.constant([1],dtype='float32')

    zero = tf.constant([0],dtype='float32')

    rotation_matrix = tf.reshape( tf.concat([c1,s1,zero, -s1,c1,zero, zero,zero,one],axis=0),[3,3] )

        

    c2 = tf.math.cos(shear)

    s2 = tf.math.sin(shear)

    shear_matrix = tf.reshape( tf.concat([one,s2,zero, zero,c2,zero, zero,zero,one],axis=0),[3,3] )    

    

    zoom_matrix = tf.reshape( tf.concat([one/height_zoom,zero,zero, zero,one/width_zoom,zero, zero,zero,one],axis=0),[3,3] )

    

    shift_matrix = tf.reshape( tf.concat([one,zero,height_shift, zero,one,width_shift, zero,zero,one],axis=0),[3,3] )

    

    return K.dot(K.dot(rotation_matrix, shear_matrix), K.dot(zoom_matrix, shift_matrix))
def transform(image,label):

    DIM = IMAGE_SIZE[0]

    XDIM = DIM%2 #fix for size 331

    

    rot = 15. * tf.random.normal([1],dtype='float32')

    shr = 5. * tf.random.normal([1],dtype='float32') 

    h_zoom = 1.0 + tf.random.normal([1],dtype='float32')/10.

    w_zoom = 1.0 + tf.random.normal([1],dtype='float32')/10.

    h_shift = 16. * tf.random.normal([1],dtype='float32') 

    w_shift = 16. * tf.random.normal([1],dtype='float32') 



    m = get_mat(rot,shr,h_zoom,w_zoom,h_shift,w_shift) 



    x = tf.repeat( tf.range(DIM//2,-DIM//2,-1), DIM )

    y = tf.tile( tf.range(-DIM//2,DIM//2),[DIM] )

    z = tf.ones([DIM*DIM],dtype='int32')

    idx = tf.stack( [x,y,z] )

    

    idx2 = K.dot(m,tf.cast(idx,dtype='float32'))

    idx2 = K.cast(idx2,dtype='int32')

    idx2 = K.clip(idx2,-DIM//2+XDIM+1,DIM//2)

         

    idx3 = tf.stack( [DIM//2-idx2[0,], DIM//2-1+idx2[1,]] )

    d = tf.gather_nd(image,tf.transpose(idx3))

        

    return tf.reshape(d,[DIM,DIM,3]),label
#need to one hot encode images so we can blend their labels like above

def onehot(image,label):

    CLASSES = len(classes)

    return image,tf.one_hot(label,CLASSES)
def mixup(image, label, PROBABILITY = 1.0):

    DIM = IMAGE_SIZE[0]

    CLASSES = 104

    

    imgs = []; labs = []

    for j in range(AUG_BATCH):



        P = tf.cast( tf.random.uniform([],0,1)<=PROBABILITY, tf.float32)



        k = tf.cast( tf.random.uniform([],0,AUG_BATCH),tf.int32)

        a = tf.random.uniform([],0,1)*P # this is beta dist with alpha=1.0



        img1 = image[j,]

        img2 = image[k,]

        imgs.append((1-a)*img1 + a*img2)



        if len(label.shape)==1:

            lab1 = tf.one_hot(label[j],CLASSES)

            lab2 = tf.one_hot(label[k],CLASSES)

        else:

            lab1 = label[j,]

            lab2 = label[k,]

        labs.append((1-a)*lab1 + a*lab2)



    image2 = tf.reshape(tf.stack(imgs),(AUG_BATCH,DIM,DIM,3))

    label2 = tf.reshape(tf.stack(labs),(AUG_BATCH,CLASSES))

    return image2,label2
#first look at test dataset

test_dataset = get_train_ds(TEST_FILENAMES, cutmix_aug = False, tta_aug = False, labeled = False,

                                        shuffle = True, repeat = False)

test_dataset = test_dataset.unbatch().batch(20)

test_batch = iter(test_dataset)
def cutmix(image, label, PROBABILITY = 1.0):

    # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]

    # output - a batch of images with cutmix applied

    DIM = IMAGE_SIZE[0]

    CLASSES = 104

    

    imgs = []; labs = []

    for j in range(AUG_BATCH):



        P = tf.cast( tf.random.uniform([],0,1)<=PROBABILITY, tf.int32)



        k = tf.cast( tf.random.uniform([],0,AUG_BATCH),tf.int32)



        x = tf.cast( tf.random.uniform([],0,DIM),tf.int32)

        y = tf.cast( tf.random.uniform([],0,DIM),tf.int32)

        b = tf.random.uniform([],0,1) # this is beta dist with alpha=1.0

        WIDTH = tf.cast( DIM * tf.math.sqrt(1-b),tf.int32) * P

        ya = tf.math.maximum(0,y-WIDTH//2)

        yb = tf.math.minimum(DIM,y+WIDTH//2)

        xa = tf.math.maximum(0,x-WIDTH//2)

        xb = tf.math.minimum(DIM,x+WIDTH//2)



        one = image[j,ya:yb,0:xa,:]

        two = image[k,ya:yb,xa:xb,:]

        three = image[j,ya:yb,xb:DIM,:]

        middle = tf.concat([one,two,three],axis=1)

        img = tf.concat([image[j,0:ya,:,:],middle,image[j,yb:DIM,:,:]],axis=0)

        imgs.append(img)



        a = tf.cast(WIDTH*WIDTH/DIM/DIM,tf.float32)

        if len(label.shape)==1:

            lab1 = tf.one_hot(label[j],CLASSES)

            lab2 = tf.one_hot(label[k],CLASSES)

        else:

            lab1 = label[j,]

            lab2 = label[k,]

        labs.append((1-a)*lab1 + a*lab2)

        

    image2 = tf.reshape(tf.stack(imgs),(AUG_BATCH,DIM,DIM,3))

    label2 = tf.reshape(tf.stack(labs),(AUG_BATCH,CLASSES))

    return image2,label2
#create function to apply both cutmix and mixup

def mixup_and_cutmix(image,label):

    CLASSES = len(classes)

    DIM = IMAGE_SIZE[0]

    

    #define how often we want to do activate cutmix or mixup

    SWITCH = 1/2

    

    #define how often we want cutmix or mixup to activate when switch is active

    CUTMIX_PROB = 2/3

    MIXUP_PROB = 2/3

    

    #apply cutmix and mixup

    image2, label2 = cutmix(image, label, CUTMIX_PROB)

    image3, label3 = mixup(image, label, MIXUP_PROB)

    imgs = []; labs = []

    

    for j in range(BATCH_SIZE):

        P = tf.cast( tf.random.uniform([],0,1)<=SWITCH, tf.float32)

        imgs.append(P*image2[j,]+(1-P)*image3[j,])

        labs.append(P*label2[j,]+(1-P)*label3[j,])

        

    #must explicitly reshape so TPU complier knows output shape

    image4 = tf.reshape(tf.stack(imgs),(BATCH_SIZE,DIM,DIM,3))

    label4 = tf.reshape(tf.stack(labs),(BATCH_SIZE,CLASSES))

    return image4,label4
row = 6; col = 4;

row = min(row,AUG_BATCH//col)

all_elements = get_train_ds(TRAINING_FILENAMES, cutmix_aug = True, tta_aug = False, labeled = True,

                                        shuffle = True, repeat = True)



for (img,label) in all_elements:

    plt.figure(figsize=(15,int(15*row/col)))

    for j in range(row*col):

        plt.subplot(row,col,j+1)

        plt.axis('off')

        plt.imshow(img[j,])

    plt.show()

    break
#define epoch parameters

EPOCHS = 20                

STEPS_PER_EPOCH = count_data_items(TRAINING_FILENAMES) // BATCH_SIZE



#define learning rate parameters

LR_START = 0.00001

LR_MAX = 0.00005 * strategy.num_replicas_in_sync

LR_MIN = 0.00001

LR_RAMPUP_EPOCHS = 5

LR_SUSTAIN_EPOCHS = 0

LR_DECAY = .8



#define ramp up and decay

def lr_schedule(epoch):

    if epoch < LR_RAMPUP_EPOCHS:

        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START

    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:

        lr = LR_MAX

    else:

        lr = (LR_MAX - LR_MIN) * LR_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN

    return lr

    

lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose = True)



#visualize learning rate schedule

rng = [i for i in range(EPOCHS)]

y = [lr_schedule(x) for x in rng]

plt.plot(rng, y)

print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))
#import DenseNet201, Xception, InceptionV3, and InceptionResNetV2

from tensorflow.keras.applications import DenseNet201, Xception, InceptionV3, InceptionResNetV2



#requirements to use EfficientNet(s)

!pip install -q efficientnet

import efficientnet.tfkeras as efn



#helper function to create our model

def get_DenseNet201():

    CLASSES = len(classes)

    with strategy.scope():

        dnet = DenseNet201(

            input_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3),

            weights = 'imagenet',

            include_top = False

        )

        #make trainable so we can fine-tune

        dnet.trainable = True

        model = tf.keras.Sequential([

            dnet,

            tf.keras.layers.GlobalAveragePooling2D(),

            #tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Dense(CLASSES, activation = 'softmax',dtype = 'float32')

        ])

    model.compile(

        optimizer='adam',

        loss = 'categorical_crossentropy', 

        metrics=['categorical_accuracy']

    )

    return model



#create Xception model

def get_Xception():

    CLASSES = len(classes)

    with strategy.scope():

        xception = Xception(

            input_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3),

            weights = 'imagenet',

            include_top = False

        )

        #make trainable so we can fine-tune

        xception.trainable = True

        model = tf.keras.Sequential([

            xception,

            tf.keras.layers.GlobalAveragePooling2D(),

            #tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Dense(CLASSES, activation = 'softmax',dtype = 'float32')

        ])

    model.compile(

        optimizer='adam',

        loss = 'categorical_crossentropy',

        metrics=['categorical_accuracy']

    )

    return model



#create Inception model

def get_InceptionV3():

    CLASSES = len(classes)

    with strategy.scope():

        inception = InceptionV3(

            input_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3),

            weights = 'imagenet',

            include_top = False

        )

        #make trainable so we can fine-tune

        inception.trainable = True

        model = tf.keras.Sequential([

            inception,

            tf.keras.layers.GlobalAveragePooling2D(),

            #tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Dense(CLASSES, activation = 'softmax',dtype = 'float32')

        ])

    model.compile(

        optimizer='adam',

        loss = 'categorical_crossentropy',

        metrics=['categorical_accuracy']

    )

    return model



#create EfficientNetB4 model

def get_EfficientNetB4():

    CLASSES = len(classes)

    with strategy.scope():

        efficient = efn.EfficientNetB4(

            input_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3),

            weights = 'noisy-student', #or imagenet

            include_top = False

        )

        #make trainable so we can fine-tune

        efficient.trainable = True

        model = tf.keras.Sequential([

            efficient,

            tf.keras.layers.GlobalAveragePooling2D(),

            #tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Dense(CLASSES, activation = 'softmax',dtype = 'float32')

        ])

    model.compile(

        optimizer='adam',

        loss = 'categorical_crossentropy',

        metrics=['categorical_accuracy']

    )

    return model



#create EfficientNetB5 model

def get_EfficientNetB5():

    CLASSES = len(classes)

    with strategy.scope():

        efficient = efn.EfficientNetB5(

            input_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3),

            weights = 'noisy-student', #or imagenet

            include_top = False

        )

        #make trainable so we can fine-tune

        efficient.trainable = True

        model = tf.keras.Sequential([

            efficient,

            tf.keras.layers.GlobalAveragePooling2D(),

            #tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Dense(CLASSES, activation = 'softmax',dtype = 'float32')

        ])

    model.compile(

        optimizer='adam',

        loss = 'categorical_crossentropy',

        metrics=['categorical_accuracy']

    )

    return model





#create EfficientNetB6 model

def get_EfficientNetB6():

    CLASSES = len(classes)

    with strategy.scope():

        efficient = efn.EfficientNetB6(

            input_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3),

            weights = 'imagenet', #or imagenet

            include_top = False

        )

        #make trainable so we can fine-tune

        efficient.trainable = True

        model = tf.keras.Sequential([

            efficient,

            tf.keras.layers.GlobalAveragePooling2D(),

            #tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Dense(CLASSES, activation = 'softmax',dtype = 'float32')

        ])

    model.compile(

        optimizer='adam',

        loss = 'categorical_crossentropy',

        metrics=['categorical_accuracy']

    )

    return model



#create EfficientNetB7 model

def get_EfficientNetB7():

    CLASSES = len(classes)

    with strategy.scope():

        efficient = efn.EfficientNetB7(

            input_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3),

            weights = 'noisy-student', #or imagenet

            include_top = False

        )

        #make trainable so we can fine-tune

        efficient.trainable = True

        model = tf.keras.Sequential([

            efficient,

            tf.keras.layers.GlobalAveragePooling2D(),

            #tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Dense(CLASSES, activation = 'softmax',dtype = 'float32')

        ])

    model.compile(

        optimizer='adam',

        loss = 'categorical_crossentropy',

        metrics=['categorical_accuracy']

    )

    return model



#create InceptionResNet model

def get_InceptionResNetV2():

    CLASSES = len(classes)

    with strategy.scope():

        inception_res = InceptionResNetV2(

            input_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3),

            weights = 'imagenet',

            include_top = False

        )

        #make trainable so we can fine-tune

        inception_res.trainable = True

        model = tf.keras.Sequential([

            inception_res,

            tf.keras.layers.GlobalAveragePooling2D(),

            #tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Dense(CLASSES, activation = 'softmax',dtype = 'float32')

        ])

    model.compile(

        optimizer='adam',

        loss = 'categorical_crossentropy',

        metrics=['categorical_accuracy']

    )

    return model
from sklearn.model_selection import KFold

#train and cross validate in folds





histories = []

models = []

#early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 3)

kfold = KFold(FOLDS, shuffle = True, random_state = SEED)



#break into different fodls

for f, (train_index, val_index) in enumerate(kfold.split(TRAINING_FILENAMES)):

#to clear the TPU memory each fold

    #tf.tpu.experimental.initialize_tpu_system(tpu)

    print(); print('-'*25)

    print(f"Training fold {f + 1} with EfficientNetB6")

    print('-'*25)

    print('Getting training data...')

    print('')

    #get files for training and validation   

    train_ds = get_train_ds(list(pd.DataFrame({'TRAINING_FILENAMES': TRAINING_FILENAMES}).loc[train_index]['TRAINING_FILENAMES']),

                                    cutmix_aug = True, tta_aug = False, labeled = True, shuffle = True, repeat = True)

    

    print('Geting validation data...')

    print('')

    val_ds = get_val_ds(list(pd.DataFrame({'TRAINING_FILENAMES': TRAINING_FILENAMES}).loc[val_index]['TRAINING_FILENAMES']),

                            labeled = True, shuffle = True, return_image_names = False)

              

    #train and cross validate

    model = get_EfficientNetB6()

    history = model.fit(train_ds, 

                        steps_per_epoch = STEPS_PER_EPOCH,

                        epochs = EPOCHS,

                        callbacks = [lr_callback], #,early_stopping]

                        validation_data = val_ds,

                        verbose = 2)

    models.append(model)

    histories.append(history)
#define function to visualize learning curves

def plot_learning_curves(histories): 

    fig, ax = plt.subplots(1, 2, figsize = (20, 10))

    

    #plot accuracies

    for i in range(0, 3):

        ax[0].plot(histories[i].history['categorical_accuracy'], color = 'C0')

        ax[0].plot(histories[i].history['val_categorical_accuracy'], color = 'C1')



    #plot losses

    for i in range(0, 3):

        ax[1].plot(histories[i].history['loss'], color = 'C0')

        ax[1].plot(histories[i].history['val_loss'], color = 'C1')



    #fix legend

    ax[0].legend(['train', 'validation'], loc = 'upper left')

    ax[1].legend(['train', 'validation'], loc = 'upper right')

    

    #set master titles

    fig.suptitle("Model Performance", fontsize=14)

    

    #label axis

    for i in range(0,2):

        ax[0].set_ylabel('Accuracy')

        ax[0].set_xlabel('Epoch')

        ax[1].set_ylabel('Loss')

        ax[1].set_xlabel('Epoch')



    return plt.show()
#look at our learning curves to check bias/variance trade off

plot_learning_curves(histories)
def get_test_dataset(filenames, shuffle = False, repeat = True, labeled = False, tta_aug = True, return_image_names = True):



    ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)

    ds = ds.cache()

    

    if repeat:

        ds = ds.repeat()

    

    if shuffle: 

        ds = ds.shuffle(1024*8)

        opt = tf.data.Options()

        opt.experimental_deterministic = False

        ds = ds.with_options(opt)



    if labeled: 

        ds = ds.map(read_labeled_tfrecord, num_parallel_calls=AUTO)

    else:

        ds = ds.map(lambda example: read_unlabeled_tfrecord(example, return_image_names), 

                    num_parallel_calls=AUTO)



    if tta_aug:

        ds = ds.map(data_augment, num_parallel_calls = AUTO)

        ds = ds.map(transform, num_parallel_calls=AUTO)

    

    ds = ds.batch(BATCH_SIZE)

    ds = ds.prefetch(AUTO)



    return ds
def average_tta_preds(tta_preds):

    average_preds = np.zeros((ct_test, 104))

    for fold in range(FOLDS):

        average_preds += tta_preds[(fold)*ct_test:(fold + 1)*ct_test] / FOLDS

    return average_preds
#since we are splitting the dataset and iterating separately on images and ids, order matters.

test_ds = get_test_dataset(TEST_FILENAMES, tta_aug = True, labeled = False,

                           repeat = True, return_image_names = True)

test_images_ds = test_ds.map(lambda image, idnum: image)

    

#set up TTA

ct_test = count_data_items(TEST_FILENAMES)

STEPS = TTA * ct_test/BATCH_SIZE



#predict  

print('Getting TTA predictions...')

pred0 = models[0].predict(test_images_ds,steps = STEPS,verbose = 2)[:TTA*ct_test,] 

pred1 = models[1].predict(test_images_ds,steps = STEPS,verbose = 2)[:TTA*ct_test,] 

pred2 = models[2].predict(test_images_ds,steps = STEPS,verbose = 2)[:TTA*ct_test,] 

print('')



#get averages of each augmentation

print('Averaging TTA predictions...')

average_pred0 = average_tta_preds(pred0)

average_pred1 = average_tta_preds(pred1)

average_pred2 = average_tta_preds(pred1)

print('')



#merge together and average

print('Merging predictions from models...')

final_preds = np.zeros((ct_test, 104))

final_preds += average_pred0

final_preds += average_pred1

final_preds += average_pred2



#and take argmax to get in label form

final_preds = np.argmax(final_preds, axis = 1)



#lastly, get test ids for submission

test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()

test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U')

print('Done')
#create submission dataframe

submission = pd.DataFrame()

submission['id'] = test_ids

submission['label'] = final_preds

print(submission.shape)

submission.head(10)
#submit to disk

submission.to_csv('submission.csv', index = False)

print('Submission saved')