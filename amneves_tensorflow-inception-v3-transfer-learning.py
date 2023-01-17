# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

# pepi

import re

import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import tensorflow as tf

import seaborn as sns

import random

import math

import PIL

import tensorflow as tf, tensorflow.keras.backend as K

import tensorflow as tf

from tensorflow.keras.applications.inception_v3 import InceptionV3

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, LeakyReLU, Dropout, Input

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import SGD, RMSprop, Adam

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from tensorflow.data.experimental import AUTOTUNE

from tensorflow.keras.metrics import AUC, Accuracy

from tensorflow.keras.losses import BinaryCrossentropy

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import pydicom 

from tqdm import tqdm

import missingno as msno 

from sklearn.metrics import roc_curve, roc_auc_score, auc

import shutil

from sklearn.model_selection import train_test_split

from functools import partial

from kaggle_datasets import KaggleDatasets

from sklearn.utils import class_weight



base_dir = '/kaggle/input/siim-isic-melanoma-classification/'

external_base_dir = '/kaggle/input/melanoma-external-malignant-256/'

external_tfrec_base_dir = '../input/melanoma-256x256'



sns.set(style="darkgrid")

random.seed(42)

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
DEVICE = 'TPU'
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

    



AUTO     = tf.data.experimental.AUTOTUNE

REPLICAS = strategy.num_replicas_in_sync

print(f'REPLICAS: {REPLICAS}')
IMG_HEIGHT = 256

IMG_WIDTH = 256

N_CHANNELS = 3

epochs = 20

BATCH_SIZE = 16 * REPLICAS

IMAGE_SIZE = [IMG_HEIGHT, IMG_WIDTH]

IMAGE_RESIZE = [IMG_HEIGHT, IMG_WIDTH]

input_shape = (IMG_HEIGHT, IMG_WIDTH, N_CHANNELS)

BALANCE_DATA = True
if DEVICE == 'TPU':

    files_dir = KaggleDatasets().get_gcs_path('melanoma-256x256')

else:

    files_dir = external_tfrec_base_dir
TRAINING_FILENAMES, VALID_FILENAMES = train_test_split(

    tf.io.gfile.glob(files_dir + '/train*.tfrec'),

    test_size=0.1, random_state=42

)

TEST_FILENAMES = tf.io.gfile.glob(files_dir + '/test*.tfrec')

print('Train TFRecord Files:', len(TRAINING_FILENAMES))

print('Validation TFRecord Files:', len(VALID_FILENAMES))

print('Test TFRecord Files:', len(TEST_FILENAMES))



submission_example = pd.read_csv(base_dir + 'sample_submission.csv')
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

    

    rot = 90. * tf.random.normal([1],dtype='float32')

    shr = 2. * tf.random.normal([1],dtype='float32') 

    h_zoom = 1.0 + tf.random.normal([1],dtype='float32')/8.

    w_zoom = 1.0 + tf.random.normal([1],dtype='float32')/8.

    h_shift = 8. * tf.random.normal([1],dtype='float32') 

    w_shift = 8. * tf.random.normal([1],dtype='float32') 

  

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
aug_data = True



def decode_image(image):

    image = tf.image.decode_jpeg(image, channels=3)

    image = tf.cast(image, tf.float32) / 255.0        

    image = tf.reshape(image, [IMAGE_SIZE[0],IMAGE_SIZE[1], 3])

    return image



def read_tfrecord(example, labeled):

    tfrecord_format = {

        "image": tf.io.FixedLenFeature([], tf.string),

        "target": tf.io.FixedLenFeature([], tf.int64)

    } if labeled else {

        "image": tf.io.FixedLenFeature([], tf.string),

        "image_name": tf.io.FixedLenFeature([], tf.string)

    }

    example = tf.io.parse_single_example(example, tfrecord_format)

    image = decode_image(example['image'])

    if labeled:

        label = tf.cast(example['target'], tf.int32)

        return image, label

    idnum = example['image_name']

    return image, idnum



def load_dataset(filenames, labeled=True, ordered=False):

    ignore_order = tf.data.Options()

    if not ordered:

        ignore_order.experimental_deterministic = False # disable order, increase speed

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files

    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order

    dataset = dataset.map(partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTO)

    return dataset



def augmentation_pipeline(image, label):

    """

    add augmentation functions here

    """

    if aug_data:

        image,_ = transform(image,label)

        image = tf.image.random_flip_left_right(image)

    return image, label



def get_training_dataset():

    dataset = load_dataset(TRAINING_FILENAMES, labeled=True)

    dataset = dataset.map(augmentation_pipeline, num_parallel_calls=AUTO)

    dataset = dataset.repeat()

    dataset = dataset.shuffle(2048)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO)

    return dataset



def get_validation_dataset(ordered=False):

    dataset = load_dataset(VALID_FILENAMES, labeled=True, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.cache()

    dataset = dataset.prefetch(AUTO)

    return dataset



def get_test_dataset(ordered=False):

    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO)

    return dataset



def count_data_items(filenames):

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]

    return np.sum(n)
train_dataset = get_training_dataset()

valid_dataset = get_validation_dataset()
NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)

NUM_VALIDATION_IMAGES = count_data_items(VALID_FILENAMES)

NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

VALID_STEPS = NUM_VALIDATION_IMAGES // BATCH_SIZE 

print("Num of steps per epoch:", STEPS_PER_EPOCH)

print("Num of steps per validation", VALID_STEPS)

print(

    'Dataset: {} training images, {} validation images, {} unlabeled test images'.format(

        NUM_TRAINING_IMAGES, NUM_VALIDATION_IMAGES, NUM_TEST_IMAGES

    )

)
def show_dataset(thumb_size, cols, rows, ds):

    mosaic = PIL.Image.new(mode='RGB', size=(thumb_size*cols + (cols-1), 

                                             thumb_size*rows + (rows-1)))

   

    for idx, data in enumerate(iter(ds)):

        img, target_or_imgid = data

        ix  = idx % cols

        iy  = idx // cols

        img = np.clip(img.numpy() * 255, 0, 255).astype(np.uint8)

        img = PIL.Image.fromarray(img)

        img = img.resize((thumb_size, thumb_size), resample=PIL.Image.BILINEAR)

        mosaic.paste(img, (ix*thumb_size + ix, 

                           iy*thumb_size + iy))



    display(mosaic)

    

ds = train_dataset.unbatch().take(12*5)  

ds_val = valid_dataset.unbatch().take(12*5)  

show_dataset(64, 12, 5, ds)

show_dataset(64, 12, 5, ds_val)
def plot_roc(y_true, y_score):

    """

    """

    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel())

    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    

    plt.figure()

    lw = 2

    plt.plot(fpr[2], tpr[2], color='darkorange',

             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic example')

    plt.legend(loc="lower right")

    plt.show()

    

def plot_train_hist(hist):

    # Plot training & validation accuracy values

    plt.plot(hist.history['auc'])

    plt.plot(hist.history['val_auc'])

    plt.title('Model accuracy')

    plt.ylabel('Accuracy')

    plt.xlabel('Epoch')

    plt.legend(['Train', 'Test'], loc='upper left')

    plt.show()



    # Plot training & validation loss values

    plt.plot(hist.history['loss'])

    plt.plot(hist.history['val_loss'])

    plt.title('Model loss')

    plt.ylabel('Loss')

    plt.xlabel('Epoch')

    plt.legend(['Train', 'Test'], loc='upper left')

    plt.show()
train_df = pd.read_csv(base_dir + 'train.csv')

y_train = train_df['target']



class_weights = class_weight.compute_class_weight('balanced',

                                                 classes=np.unique(y_train),

                                                 y=y_train)





class_weights = {0: class_weights[0],1: class_weights[1]}

if not BALANCE_DATA:

    class_weights = {0: 1,1: 2}

print(class_weights)
learning_rate_reduction =ReduceLROnPlateau( 

    monitor='loss',    # Quantity to be monitored.

    factor=0.25,       # Factor by which the learning rate will be reduced. new_lr = lr * factor

    patience=2,        # The number of epochs with no improvement after which learning rate will be reduced.

    verbose=1,         # 0: quiet - 1: update messages.

    mode="auto",       # {auto, min, max}. In min mode, lr will be reduced when the quantity monitored has stopped decreasing; 

                       # in the max mode it will be reduced when the quantity monitored has stopped increasing; 

                       # in auto mode, the direction is automatically inferred from the name of the monitored quantity.

    min_delta=0.0001,  # threshold for measuring the new optimum, to only focus on significant changes.

    cooldown=0,        # number of epochs to wait before resuming normal operation after learning rate (lr) has been reduced.

    min_lr=0.00001     # lower bound on the learning rate.

    )



es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=300, restore_best_weights=True)



def get_lr_callback():

    lr_start   = 0.000005

    lr_max     = 0.00000125 * REPLICAS

    lr_min     = 0.000001

    lr_ramp_ep = 5

    lr_sus_ep  = 0

    lr_decay   = 0.8

   

    def lrfn(epoch):

        if epoch < lr_ramp_ep:

            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start

            

        elif epoch < lr_ramp_ep + lr_sus_ep:

            lr = lr_max

            

        else:

            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min

            

        return lr



    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)

    return lr_callback
optimizer_train = RMSprop(learning_rate=0.00025,

    rho=0.9,

    momentum=0.1,

    epsilon=1e-07,

    centered=True,

    name='RMSprop')



optimizer_fine = SGD(lr=0.000005)
# create the base pre-trained model

base_model = InceptionV3(weights='imagenet', include_top=False,

                         input_shape=(IMG_HEIGHT, IMG_WIDTH, N_CHANNELS))



# add a global spatial average pooling layer

x = base_model.output

x = GlobalAveragePooling2D()(x)



# sigmoid for binary classification with one neuron

predictions = Dense(1, activation='sigmoid')(x)



# this is the model we will train

model = Model(inputs=base_model.input, outputs=predictions)



# first: train only the top layers (which were randomly initialized)

# i.e. freeze all convolutional InceptionV3 layers

for layer in base_model.layers:

    layer.trainable = False



# compile the model

# (should be done *after* setting layers to non-trainable)

model.compile(optimizer=optimizer_train, loss=BinaryCrossentropy(label_smoothing=0.05),

              metrics=['AUC'])



# train the model on the new data for a few epochs

hist_train = model.fit(train_dataset,

                 validation_data=valid_dataset,

                 steps_per_epoch=STEPS_PER_EPOCH,

                 validation_steps=VALID_STEPS,

                 epochs=epochs,callbacks=[learning_rate_reduction, es])



#we chose to train the top 2 inception blocks, i.e. we will freeze

# the first 249 layers and unfreeze the rest:

for layer in model.layers[:249]:

    layer.trainable = False

for layer in model.layers[249:]:

    layer.trainable = True



# we need to recompile the model for these modifications to take effect

# we use SGD with a low learning rate



model.compile(optimizer=optimizer_fine,

              loss=BinaryCrossentropy(label_smoothing=0.05),

             metrics=['AUC'])



# we train our model again (

# this time fine-tuning the top 2 inception blocks

# alongside the top Dense layers

hist_fine = model.fit(train_dataset,

                 validation_data=valid_dataset,

                 steps_per_epoch=STEPS_PER_EPOCH,

                 validation_steps=VALID_STEPS,

                 epochs=epochs, callbacks=[get_lr_callback(), es])

plot_train_hist(hist_train)
plot_train_hist(hist_fine)
test_ds = get_test_dataset(ordered=True)

test_images_ds = test_ds.map(lambda image, idnum: image)
probabilities = model.predict(test_images_ds)
print('Generating submission.csv file...')

test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()

test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch
pred_df = pd.DataFrame({'image_name': test_ids, 'target': np.concatenate(probabilities)})

pred_df.head()
del submission_example['target']

submission_example = submission_example.merge(pred_df, on='image_name')

#sub.to_csv('submission_label_smoothing.csv', index=False)

submission_example.to_csv('submission_b5.csv', index=False)

submission_example.head()