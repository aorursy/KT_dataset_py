import re

import numpy as np

import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt

from kaggle_datasets import KaggleDatasets

print(tf.__version__)

AUTO = tf.data.experimental.AUTOTUNE

GCS_PATH=KaggleDatasets().get_gcs_path('512x512-melanoma-tfrecords-70k-images')
try:

    # TPU detection. No parameters necessary if TPU_NAME environment variable is set.

    # On Kaggle this is always the case.

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    # default distribution strategy in Tensorflow. Works on CPU and single GPU.

    strategy = tf.distribute.get_strategy()



print("REPLICAS: ", strategy.num_replicas_in_sync)
BATCH_SIZE = 16 * strategy.num_replicas_in_sync

            

IMAGE_SIZE = [512 , 512]
TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH+'/train*')



TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH+'/test*')
def decode_augument_image(image_data):

    image = tf.image.decode_jpeg(image_data, channels=3)

    image = tf.cast(image, tf.bfloat16) / 255.0  # convert image to floats in [0, 1] range

    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU

    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_flip_up_down(image)

    return image



def decode_image(image_data):

    image = tf.image.decode_jpeg(image_data, channels=3)

    image = tf.cast(image, tf.bfloat16) / 255.0  # convert image to floats in [0, 1] range

    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU

    return image



def read_labeled_tfrecord(example):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), 

        "age_approx": tf.io.FixedLenFeature([], tf.int64),  

        "sex": tf.io.FixedLenFeature([], tf.int64), 

        "anatom_site_general_challenge" : tf.io.FixedLenFeature([] , tf.int64),

        "target": tf.io.FixedLenFeature([], tf.int64),  

    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_augument_image(example['image'])

    age = tf.cast(example['age_approx'], tf.bfloat16)

    sex = tf.cast(example['sex'], tf.bfloat16)

    asg = tf.cast(example['anatom_site_general_challenge'] , tf.bfloat16)

    target = tf.cast(example['target'], tf.int32)

    return image,target



def read_unlabeled_tfrecord(example):

    UNLABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "age_approx": tf.io.FixedLenFeature([], tf.int64),  

        "sex": tf.io.FixedLenFeature([], tf.int64), 

        "anatom_site_general_challenge" : tf.io.FixedLenFeature([] , tf.int64),

        "image_name": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element

        # class is missing, this competitions's challenge is to predict flower classes for the test dataset

    }

    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    age = tf.cast(example['age_approx'], tf.bfloat16)

    sex = tf.cast(example['sex'], tf.bfloat16)

    asg = tf.cast(example['anatom_site_general_challenge'] , tf.bfloat16)

    idnum = example['image_name']

    return image,idnum # returns a dataset of image(s)



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

    dataset = load_dataset(TRAINING_FILENAMES,labeled=True)

    dataset = dataset.repeat() # the training dataset must repeat for several epochs

    dataset = dataset.shuffle(2048)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def get_test_dataset(ordered=True):

    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def count_data_items(filenames):

    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]

    return np.sum(n)



NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)

NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

#VALIDATION_STEPS = NUM_VALID_IMAGES // BATCH_SIZE

print('Dataset: {} training images ,{} unlabeled test images'.format(NUM_TRAINING_IMAGES,NUM_TEST_IMAGES))

print("STEPS_PER_EPOCH are {}".format(STEPS_PER_EPOCH))

#print("validation Steps are {}".format(VALIDATION_STEPS))
train_ds=get_training_dataset()
!pip install -q efficientnet
from tensorflow.keras import *

from tensorflow.keras.layers import *

from efficientnet.tfkeras import *
def create_model():

    base_model=EfficientNetB7(include_top=False,input_shape=(*IMAGE_SIZE,3))

    base_model.trainable=False

    inp1=Input(shape=(*IMAGE_SIZE,3))

    #inp2=Input(shape=(3,))

    X=base_model(inp1,training=False)

    X=GlobalAveragePooling2D()(X)

    '''Z=Dense(256,activation='relu')(inp2)

    Z=BatchNormalization()(Z)

    Z=Dropout(0.4)(Z)

    Z=Dense(256,activation='relu')(Z)

    Z=BatchNormalization()(Z)

    Z=Dropout(0.4)(Z)

    X=Concatenate()([X,Z])'''

    X=Dense(512,activation='relu')(X)

    X=BatchNormalization()(X)

    X=Dropout(0.2)(X)

    X=Dense(1024,activation='relu')(X)

    X=BatchNormalization()(X)

    X=Dropout(0.4)(X)

    Y=Dense(1,activation='sigmoid')(X)

    return Model(inputs=inp1,outputs=Y)
def create_model2():

    base_model=EfficientNetB7(include_top=False,input_shape=(*IMAGE_SIZE,3))

    base_model.trainable=False

    inp1=Input(shape=(*IMAGE_SIZE,3))

    #inp2=Input(shape=(3,))

    X=base_model(inp1,training=False)

    X=GlobalAveragePooling2D()(X)

    '''Z=Dense(256,activation='relu')(inp2)

    Z=BatchNormalization()(Z)

    Z=Dropout(0.4)(Z)

    Z=Dense(256,activation='relu')(Z)

    Z=BatchNormalization()(Z)

    Z=Dropout(0.4)(Z)

    X=Concatenate()([X,Z])'''

    X=Dense(256,activation='relu')(X)

    X=Dropout(0.4)(X)

    X=BatchNormalization()(X)

    X=Dense(1024,activation='relu')(X)

    X=BatchNormalization()(X)

    X=Dropout(0.4)(X)

    Y=Dense(1,activation='sigmoid')(X)

    return Model(inputs=inp1,outputs=Y)
with strategy.scope():

    model = create_model()

    

    model.compile(optimizer='rmsprop',

                  loss='binary_crossentropy',

                  metrics=[tf.keras.metrics.BinaryCrossentropy(),'accuracy'])

    

    model.summary()
tf.keras.utils.plot_model(model,show_shapes=True)
# Learning rate schedule for TPU, GPU and CPU.

# Using an LR ramp up because fine-tuning a pre-trained model.

# Starting with a high LR would break the pre-trained weights.



LR_START = 0.000001

LR_MAX = 0.00005 * strategy.num_replicas_in_sync

LR_MIN = 0.00001

LR_RAMPUP_EPOCHS = 4

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



es=tf.keras.callbacks.EarlyStopping(monitor='loss',mode='min',patience=4,verbose=1)



reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2,patience=4,mode='min', min_lr=0.001)
model.fit(train_ds,

          epochs=20,

          steps_per_epoch=STEPS_PER_EPOCH,

          callbacks=[es,lr_callback,reduce_lr]

         )
"""def display_training_curves(training, validation, title, subplot):

    #Source: https://www.kaggle.com/mgornergoogle/getting-started-with-100-flowers-on-tpu

    

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

    ax.legend(['train', 'valid.'])"""
import h5py



model_json = model.to_json()

with open("model.json", "w") as json_file:

    json_file.write(model_json)



model.save_weights("mo.h5")

print("Saved model to disk")
def predictions():

    test_ds=get_test_dataset(ordered=True)



    test_ds_features=test_ds.map(lambda img,idnum: img) #Getting the features of the Test_ds



    preds=model.predict(test_ds_features) #predicting with the model



    test_ids_ds = test_ds.map(lambda img,imname: imname).unbatch()



    test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U')



    prediction_df=pd.DataFrame({'image_name':test_ids ,'target':np.concatenate(preds)}) #writing to Dataframs



    prediction_df.to_csv("submission.csv",index=False) #Generating CSV file

    

predictions()
"""with strategy.scope():

    model2=create_model2()

    

    model2.compile(optimizer='adam',

                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),

                  metrics=[tf.keras.metrics.BinaryCrossentropy(),'accuracy'])

    model2.summary()"""
tf.keras.utils.plot_model(model2,show_shapes=True)
model2.fit(get_training_dataset(),

           epochs=10,

           steps_per_epoch=STEPS_PER_EPOCH//2,

           callbacks=[es,lr_callback,reduce_lr]

          )
def predictions():

    test_ds=get_test_dataset(ordered=True)



    test_ds_features=test_ds.map(lambda img,idnum: img) #Getting the features of the Test_ds



    preds=model2.predict(test_ds_features) #predicting with the model



    test_ids_ds = test_ds.map(lambda img,imname: imname).unbatch()



    test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U')



    prediction_df=pd.DataFrame({'image_name':test_ids ,'target':np.concatenate(preds)}) #writing to Dataframs



    prediction_df.to_csv("submission(1).csv",index=False) #Generating CSV file

    

predictions()

#not well