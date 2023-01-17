from datetime import datetime
start_time=datetime.now()
import math, re, os
import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from kaggle_datasets import KaggleDatasets
print("Tensorflow version " + tf.__version__)
AUTO = tf.data.experimental.AUTOTUNE
GCS_DS_PATH = KaggleDatasets().get_gcs_path() 
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
BATCH_SIZE = 16*strategy.num_replicas_in_sync

IMAGE_SIZE = [1024, 1024]

EPOCHS = 15


#train tfrec files
TRAINING_FILENAMES=tf.io.gfile.glob(GCS_DS_PATH+'/tfrecords/train*')

#Splitting the training data
TRAINING_FILENAMES,VALIDATION_FILENAMES=train_test_split(TRAINING_FILENAMES,test_size=0.2,random_state=7)

#test tfrec files
TEST_FILENAMES=tf.io.gfile.glob(GCS_DS_PATH+ '/tfrecords/test*.tfrec')

print(len(TRAINING_FILENAMES)  , len(VALIDATION_FILENAMES))
def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.bfloat16) / 255.0  # convert image to floats in [0, 1] range
    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU
    return image

def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "target": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    label = tf.cast(example['target'], tf.int32)
    return image, label # returns a dataset of (image, label) pairs

def read_unlabeled_tfrecord(example):
    UNLABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "image_name": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element
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

def data_augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_hue(image,0,3)
    #image = tf.image.random_crop(image ,[224,224,3])
    return image, label  

def get_training_dataset():
    dataset = load_dataset(TRAINING_FILENAMES, labeled=True)
    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_validation_dataset():
    dataset = load_dataset(VALIDATION_FILENAMES, labeled=True)
    #dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    #dataset = dataset.repeat() # the training dataset must repeat for several epochs
    #dataset = dataset.shuffle(128)
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

NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
NUM_VALIDATION_IMAGES= count_data_items(VALIDATION_FILENAMES)
NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
VALIDATION_STEPS= NUM_VALIDATION_IMAGES // BATCH_SIZE
print('Dataset: {} training images,{} unlabeled test images'.format(NUM_TRAINING_IMAGES,NUM_TEST_IMAGES))
print("STEPS_PER_EPOCH are {}".format(STEPS_PER_EPOCH))
print("VALIDATION_STEPS are {}".format(VALIDATION_STEPS))
#training preprocessed dataset 
train_ds=get_training_dataset()
train_ds.element_spec
valid_ds=get_validation_dataset()
valid_ds.element_spec
!pip install -q efficientnet
from efficientnet.tfkeras import EfficientNetB7
from tensorflow.keras import *
from tensorflow.keras.layers import *
# Learning rate schedule for TPU, GPU and CPU.
# Using an LR ramp up because fine-tuning a pre-trained model.
# Starting with a high LR would break the pre-trained weights.

LR_START = 0.000001
LR_MAX = 0.00005 * 16 #SETTING THE TPU CORES FOR EVERY DEVICE TO WORK ON
LR_MIN = 0.00001
LR_RAMPUP_EPOCHS = 2
LR_SUSTAIN_EPOCHS = 4
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

es=tf.keras.callbacks.EarlyStopping(monitor='val_loss',mode='min',patience=4,verbose=1)
def create_model():
    base_model=EfficientNetB7(include_top=False,input_shape=(*IMAGE_SIZE,3),weights='imagenet')
    base_model.trainable=False
    inputs=Input(shape=(*IMAGE_SIZE,3))
    X=base_model(inputs,training=False)
    X=GlobalAveragePooling2D()(X)
    X=Dense(1024,activation='relu')(X)
    X=BatchNormalization()(X)
    X=Dropout(0.3)(X)
    outputs=Dense(1,activation='sigmoid')(X)
    return Model(inputs,outputs)
with strategy.scope():
    model=create_model()
    
    model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=[tf.keras.metrics.AUC()])
    
    model.summary()
model.fit(train_ds,
          epochs=EPOCHS,
          steps_per_epoch=STEPS_PER_EPOCH,
          validation_data=valid_ds,
          validation_steps=VALIDATION_STEPS,
          callbacks=[es,lr_callback]
         )
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
    model.history.history['loss'],
    model.history.history['val_loss'],
    'loss',211
)
display_training_curves(
    model.history.history['auc'],
    model.history.history['val_auc'],
    'auc',212
)
#testing dataset
test_ds=get_test_dataset(ordered=True)

test_images_ids=test_ds.map(lambda image ,idnum : image) #Retriving the image data from the test dataset

predictions = model.predict(test_images_ids) #predictions on images

test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch() #Retriving the Image name i.e., IDNUM

test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') #Getting the image names for the csv file

prediction_df=pd.DataFrame({'image_name':test_ids ,'target':np.concatenate(predictions)})

prediction_df.to_csv('submission.csv',index=False) #generating the submission.csv file
print("Time taken")
print(datetime.now()-start_time)