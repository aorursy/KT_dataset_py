!pip install -q efficientnet

!pip install tensorflow-addons
import random, re, math

import glob

import os

import numpy as np, pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

import tensorflow as tf, tensorflow.keras.backend as K

from kaggle_datasets import KaggleDatasets

print('Tensorflow version ' + tf.__version__)

from sklearn.model_selection import KFold

import efficientnet.tfkeras as efn

import tensorflow_addons as tfa

from tensorflow.python.ops import math_ops

from tensorflow.keras.applications import DenseNet201

import types
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
LOAD_PRETRAIN = True

MODEL_NAME = 'efficientnetb3'

MODEL_PATH = '../input/model25/model24.h5'



FOLDS = 4

SEED = 777

TRAIN_FOLD = 0



LR = 0.001



SHUFFLE = True



IMAGE_SIZE = [512, 512]

EPOCHS = 8

BATCH_SIZE = 48 * strategy.num_replicas_in_sync



AUTO = tf.data.experimental.AUTOTUNE
from tqdm.notebook import tqdm 

import glob

import os

CLASSES = ['cover', 'jmipod', 'juniward', 'uerd' ]      

training_filenames = []



for i in tqdm(CLASSES):

    

    records = glob.glob(f'/kaggle/input/alaska2-ds-0512-{i}/*.tfrec')

    TF_REC_DS_PATH = KaggleDatasets().get_gcs_path(f'alaska2-ds-0512-{i}')

    records = [os.path.join(TF_REC_DS_PATH, record.split('/')[-1]) for record in records]

    training_filenames += records

    

TRAINING_FILENAMES = training_filenames
def sequentialSample(elem):

    return int(elem.split('/')[-1][-9:-6])

TRAINING_FILENAMES.sort(key=sequentialSample)
callbacks = [

    tf.keras.callbacks.ReduceLROnPlateau(

        monitor='val_loss', 

        factor=0.25, 

        patience=2, 

        verbose=1, 

        mode='auto',

        min_lr=0.00001

    ),

    tf.keras.callbacks.ModelCheckpoint(

        'model_best.h5', 

        monitor='val_loss', 

        verbose=0, 

        save_best_only=True,

        save_weights_only=False

    ),

]
## https://www.kaggle.com/cdeotte/cutmix-and-mixup-on-gpu-tpu



def decode_image(image_data):

    image = tf.image.decode_jpeg(image_data, channels=3)

    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range

    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU

    return image



def read_labeled_tfrecord(example):

    LABELED_TFREC_FORMAT = {

        'dct1': tf.io.FixedLenFeature([], tf.string),

        'dct2': tf.io.FixedLenFeature([], tf.string),

        'image': tf.io.FixedLenFeature([], tf.string),

        'mask': tf.io.FixedLenFeature([], tf.string),

        'label': tf.io.FixedLenFeature([], tf.int64),

        'q': tf.io.FixedLenFeature([], tf.int64),

    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    label = tf.cast(example['label'], tf.int32)

    return image, label # returns a dataset of (image, label) pairs



def load_dataset(filenames, labeled = True, ordered = False):

    # Read from TFRecords. For optimal performance, reading from multiple files at once and

    # Diregarding data order. Order does not matter since we will be shuffling the data anyway

    

    ignore_order = tf.data.Options()

    if not ordered:

        ignore_order.experimental_deterministic = False # disable order, increase speed

        

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads = AUTO) # automatically interleaves reads from multiple files

    dataset = dataset.with_options(ignore_order) # use data as soon as it streams in, rather than in its original order

    dataset = dataset.map(read_labeled_tfrecord, num_parallel_calls = AUTO) # returns a dataset of (image, label) pairs if labeled = True or (image, id) pair if labeld = False

    return dataset



def data_augment(image, label):

    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),

    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part

    # of the TPU while the TPU itself is computing gradients.

    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_flip_up_down(image)

#     if np.random.random() < 0.5:

#         image = tf.keras.preprocessing.image.random_rotation(image, 30)

    return image, label   



def get_training_dataset(dataset):

    if SHUFFLE:

        dataset = dataset.shuffle(2048)

    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)

    dataset = dataset.repeat() # the training dataset must repeat for several epochs

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.map(onehot, num_parallel_calls=AUTO)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def get_validation_dataset(dataset, do_onehot=True):

#     dataset = dataset.repeat()

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.map(onehot, num_parallel_calls=AUTO) # we must use one hot like augmented train data

#     dataset = dataset.cache()

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def onehot(image,label):

    return image,tf.one_hot(label,4)
NUM_TRAINING_IMAGES = 75000*4

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE



print('Dataset: {} training images'.format(NUM_TRAINING_IMAGES))
def fix_auc(binary = False):

    # only tested with default parameters for the AUC class

    # do not run this again in the same session 

    

    def update_state(self, y_true, y_pred, sample_weight=None):

        # for sparse categorical accuracy 

        y_true = 1 - y_true[:, 0]

        y_true = tf.cast(y_true != 0, tf.int64)

        y_pred = 1 - y_pred[:, 0]

        return self._update_state(y_true, y_pred, sample_weight)



    def result(self):



        normalization = 1.4



        recall = math_ops.div_no_nan(self.true_positives,

                                     self.true_positives + self.false_negatives)



        fp_rate = math_ops.div_no_nan(self.false_positives,

                                    self.false_positives + self.true_negatives)

        x = fp_rate

        y = recall



        heights = (y[:self.num_thresholds - 1] + y[1:]) / 2.



        regular_auc = math_ops.reduce_sum(math_ops.multiply(x[:self.num_thresholds - 1] - x[1:], heights), name=self.name)



        under40_auc = math_ops.reduce_sum(math_ops.multiply(x[:self.num_thresholds - 1] - x[1:], tf.clip_by_value (heights, 0, 0.4)), name=self.name)



        return (regular_auc + under40_auc) / 1.4

    

    if not binary: 

        if not hasattr(tf.keras.metrics.AUC, '_update_state'):

            tf.keras.metrics.AUC._update_state = tf.keras.metrics.AUC.update_state

        tf.keras.metrics.AUC.update_state = update_state

    

    tf.keras.metrics.AUC.result = result

    

    return tf.keras.metrics.AUC
from tensorflow.keras.applications import DenseNet201

lode_pretrain = LOAD_PRETRAIN

def get_model(name='efficientnetb3'):

    tf.keras.metrics.AUC = fix_auc(binary = False)

    lr = tf.keras.experimental.CosineDecayRestarts(

        initial_learning_rate=0.0005, first_decay_steps=300, t_mul=2.0, m_mul=1.0, alpha=0.0,

        name=None

    )

    with strategy.scope():

        if not lode_pretrain:

            if name.lower() == 'densenet201':

                rnet = DenseNet201(

                    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),

                    weights='imagenet',

                    include_top=False

                )

                # trainable rnet

                rnet.trainable = True

                model = tf.keras.Sequential([

                    rnet,

                    tf.keras.layers.GlobalAveragePooling2D(),

                    tf.keras.layers.Dense(len(CLASSES), activation='softmax',dtype='float32')

                ])

            elif name.lower() == 'efficientnetb7':

                enb7 = efn.EfficientNetB7(weights='noisy-student', include_top=False, input_shape=[512, 512, 3])

                enb7.trainable = True 

                model = tf.keras.Sequential([

                    enb7,

                    tf.keras.layers.GlobalAveragePooling2D(),

                    tf.keras.layers.Dense(len(CLASSES), activation='softmax',dtype='float32')

                ])

            elif name.lower() == 'efficientnetb3':

                enb3 = efn.EfficientNetB3(include_top=False, input_shape=[512, 512, 3])

                enb3.trainable = True 

                model = tf.keras.Sequential([

                    enb3,

                    tf.keras.layers.GlobalAveragePooling2D(),

                    tf.keras.layers.Dense(len(CLASSES), activation='softmax',dtype='float32')

                ])



            model.compile(

                optimizer='adam',#tfa.optimizers.AdamW(learning_rate = 0.001, weight_decay=1e-5),

                loss = tf.keras.losses.CategoricalCrossentropy(),

                metrics=[tf.keras.metrics.AUC()]

            )

        else:

            model = tf.keras.models.load_model(MODEL_PATH)

            model.compile(

                optimizer=tfa.optimizers.AdamW(learning_rate = LR, weight_decay=1e-5),

                loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),

                metrics=[tf.keras.metrics.AUC()]

            )

    return model
def train_cross_validate(folds = 5):

    kfold = KFold(folds, shuffle = True, random_state = SEED)

    for f, (trn_ind, val_ind) in enumerate(kfold.split(TRAINING_FILENAMES)):

        if f != TRAIN_FOLD:

            continue 

        print(); print('#'*25)

        print('### FOLD',f+1)

        print('#'*25)

        train_dataset = load_dataset(list(pd.DataFrame({'TRAINING_FILENAMES': TRAINING_FILENAMES}).loc[trn_ind]['TRAINING_FILENAMES']), labeled = True)

        val_dataset = load_dataset(list(pd.DataFrame({'TRAINING_FILENAMES': TRAINING_FILENAMES}).loc[val_ind]['TRAINING_FILENAMES']), labeled = True)

        model = get_model(MODEL_NAME)

        model.summary()

        history = model.fit(

            get_training_dataset(train_dataset), 

            steps_per_epoch = int(np.floor((FOLDS-1)*STEPS_PER_EPOCH/FOLDS)),

            epochs = EPOCHS,

            callbacks = callbacks,

            validation_data = get_validation_dataset(val_dataset),

            verbose=1

        )

    return history, model

    

# run train and predict

history, model = train_cross_validate(folds = FOLDS)
model.save("model_last.h5")
def display_training_curves(training, validation, title, subplot):

    """

    Source: https://www.kaggle.com/mgornergoogle/getting-started-with-100-flowers-on-tpu

    """

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



display_training_curves(

    history.history['loss'], 

    history.history['val_loss'], 

    'loss', 211)

display_training_curves(

    history.history['auc'], 

    history.history['val_auc'], 

    'auc', 212)