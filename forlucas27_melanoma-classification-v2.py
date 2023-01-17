# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math, re, os

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf
from kaggle_datasets import KaggleDatasets

AUTO = tf.data.experimental.AUTOTUNE
# Detect TPU, return appropriate distribution strategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver() 
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy() 

print("REPLICAS: ", strategy.num_replicas_in_sync)
GCS_DS_PATH = KaggleDatasets().get_gcs_path('siim-isic-melanoma-classification')

IMAGE_SIZE = [1024, 1024]                   
EPOCHS = 12
BATCH_SIZE = 16 * strategy.num_replicas_in_sync

GCS_PATH = GCS_DS_PATH + '/tfrecords'

# %% [code]
TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/*train*.tfrec')[0:11]
VALIDATION_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/*train*.tfrec')[12:16]
TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/*test*.tfrec') 
print(TRAINING_FILENAMES[0])
#for example in tf.compat.v1.python_io.tf_record_iterator(TRAINING_FILENAMES[0]):
#    print(tf.train.Example.FromString(example))


def data_augment(image, label):
    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),
    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part
    # of the TPU while the TPU itself is computing gradients.
    image = tf.image.random_flip_left_right(image)
    #image = tf.image.random_saturation(image, 0, 2)
    return image, label  

def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU
    return image

def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "image_name": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element
        #"patient_id": tf.io.FixedLenFeature([], tf.int64),
        #"sex": tf.io.FixedLenFeature([], tf.int64),
        #"age_approx": tf.io.FixedLenFeature([], tf.int64),
        #"anatom_site_general_challenge": tf.io.FixedLenFeature([], tf.int64),
        #"diagnosis": tf.io.FixedLenFeature([], tf.int64),
        "target": tf.io.FixedLenFeature([], tf.int64)
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    label = tf.cast(example['target'], tf.int32)
    print(image,label)
    return image, label # returns a dataset of (image, label) pairs


def read_unlabeled_tfrecord(example):
    UNLABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "image_name": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element
        #"target": tf.io.FixedLenFeature([], tf.int64)
        # class is missing, this competitions's challenge is to predict flower classes for the test dataset
    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    idnum = example['image_name']
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

print(get_training_dataset())
ds_train = get_training_dataset()
ds_valid = get_validation_dataset()
ds_test = get_test_dataset()

print("Training:", ds_train)
print ("Validation:", ds_valid)
print("Test:", ds_test)
!pip install -q efficientnet
import efficientnet.tfkeras as efn

def get_model():
    model_input = tf.keras.Input(shape=(1024, 1024, 3), name='imgIn')

    dummy = tf.keras.layers.Lambda(lambda x:x)(model_input)
    
    outputs = []    
    for i in range(7):
        constructor = getattr(efn, f'EfficientNetB{i}')
        
        x = constructor(include_top=False, weights='imagenet', 
                        input_shape=(1024, 1024, 3), 
                        pooling='avg')(dummy)
        
        x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        outputs.append(x)
        
    model = tf.keras.Model(model_input, outputs, name='aNetwork')
    model.summary()
    return model

def compile_new_model():    
    with strategy.scope():
        model = get_model()
     
        losses = [tf.keras.losses.BinaryCrossentropy(label_smoothing = 0.05)
                  for i in range(7)]
        
        model.compile(
            optimizer = 'adam',
            loss      = losses,
            metrics   = [tf.keras.metrics.AUC(name='auc')])
        
    return model
model        = compile_new_model()
model.summary()
##from keras.layers.advanced_activations import LeakyReLU
##and then change you model from
##model.add(Activation("relu")
##to
##model.add(LeakyReLU(alpha=0.3))

#with strategy.scope():
#    pretrained_model = tf.keras.applications.EfficientNetB7(
#        weights='imagenet',
#        include_top=False ,
#        input_shape=[*IMAGE_SIZE, 3]
#    )
#    pretrained_model.trainable = False
#    
#    model = tf.keras.Sequential([
#        # To a base pretrained on ImageNet to extract features from images...
#        pretrained_model,
#        # ... attach a new head to act as a classifier.
#        tf.keras.layers.GlobalAveragePooling2D(),
#        tf.keras.layers.Dense( 100, activation='sigmoid')
#        #input_dim=X_train.shape[1]
#    ])
#    model.compile(
#        optimizer='adam',
#        loss = 'sparse_categorical_crossentropy',
#        metrics=['sparse_categorical_accuracy'],
#    )
    
    
#model.summary()
## Define the batch size. This will be 16 with TPU off and 128 (=16*8) with TPU on
#BATCH_SIZE = 16 * strategy.num_replicas_in_sync
#NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)

## Define training epochs
#EPOCHS = 5 # 20 is better
#STEPS_PER_EPOCH = int(NUM_TRAINING_IMAGES // BATCH_SIZE)
#print(STEPS_PER_EPOCH)
#print(ds_train)
#history = model.fit(
#    ds_train,
#    validation_data=ds_valid,
#    epochs=EPOCHS,
#    steps_per_epoch=STEPS_PER_EPOCH,
#)
STEPS_PER_EPOCH = int(NUM_TRAINING_IMAGES // BATCH_SIZE)

model        = compile_new_model(CFG)
history      = model.fit(ds_train, 
                         verbose          = 1,
                         steps_per_epoch  = STEPS_PER_EPOCH, 
                         epochs           = 5) #12
                         #callbacks        = [get_lr_callback()])
from matplotlib import pyplot as plt
display_training_curves(
    history.history['loss'],
    history.history['val_loss'],
    'loss',
    211,
)
display_training_curves(
    history.history['sparse_categorical_accuracy'],
    history.history['val_sparse_categorical_accuracy'],
    'accuracy',
    212,
)
test_ds = get_test_dataset(ordered=True)

print('Computing predictions...')
test_images_ds = test_ds.map(lambda image, idnum: image)
probabilities = model.predict(test_images_ds)
predictions = np.argmax(probabilities, axis=-1)
print(predictions)
print('Generating submission.csv file...')

NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)

# Get image ids from test set and convert to unicode
test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()
test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U')

# Write the submission file
np.savetxt(
    'submission.csv',
    np.rec.fromarrays([test_ids, predictions]),
    fmt=['%s', '%d'],
    delimiter=',',
    header='image_name,target',
    comments='',
)

# Look at the first few predictions
!head submission.csv
