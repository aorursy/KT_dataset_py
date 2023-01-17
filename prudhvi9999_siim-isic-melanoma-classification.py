from datetime import datetime
start_time=datetime.now()
import math, re, os
import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
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
BATCH_SIZE=8*strategy.num_replicas_in_sync

IMAGE_SIZE = 1024

EPOCHS = 5


#train tfrec files
TRAINING_FILENAMES=['gs://kds-f941eda182a90288c86bd91ec807ec5f6c58e5c4e0fbf9ad2114f762/tfrecords/train00-2071.tfrec',
 'gs://kds-f941eda182a90288c86bd91ec807ec5f6c58e5c4e0fbf9ad2114f762/tfrecords/train01-2071.tfrec',
 'gs://kds-f941eda182a90288c86bd91ec807ec5f6c58e5c4e0fbf9ad2114f762/tfrecords/train02-2071.tfrec',
 'gs://kds-f941eda182a90288c86bd91ec807ec5f6c58e5c4e0fbf9ad2114f762/tfrecords/train03-2071.tfrec',
 'gs://kds-f941eda182a90288c86bd91ec807ec5f6c58e5c4e0fbf9ad2114f762/tfrecords/train04-2071.tfrec',
 'gs://kds-f941eda182a90288c86bd91ec807ec5f6c58e5c4e0fbf9ad2114f762/tfrecords/train05-2071.tfrec',
 'gs://kds-f941eda182a90288c86bd91ec807ec5f6c58e5c4e0fbf9ad2114f762/tfrecords/train06-2071.tfrec',
 'gs://kds-f941eda182a90288c86bd91ec807ec5f6c58e5c4e0fbf9ad2114f762/tfrecords/train07-2071.tfrec',
 'gs://kds-f941eda182a90288c86bd91ec807ec5f6c58e5c4e0fbf9ad2114f762/tfrecords/train08-2071.tfrec',
 'gs://kds-f941eda182a90288c86bd91ec807ec5f6c58e5c4e0fbf9ad2114f762/tfrecords/train09-2071.tfrec',
 'gs://kds-f941eda182a90288c86bd91ec807ec5f6c58e5c4e0fbf9ad2114f762/tfrecords/train10-2071.tfrec',
 'gs://kds-f941eda182a90288c86bd91ec807ec5f6c58e5c4e0fbf9ad2114f762/tfrecords/train11-2071.tfrec',
 'gs://kds-f941eda182a90288c86bd91ec807ec5f6c58e5c4e0fbf9ad2114f762/tfrecords/train12-2071.tfrec',
 'gs://kds-f941eda182a90288c86bd91ec807ec5f6c58e5c4e0fbf9ad2114f762/tfrecords/train13-2071.tfrec',
 'gs://kds-f941eda182a90288c86bd91ec807ec5f6c58e5c4e0fbf9ad2114f762/tfrecords/train14-2071.tfrec',
]

VALID=tf.io.gfile.glob('gs://kds-f941eda182a90288c86bd91ec807ec5f6c58e5c4e0fbf9ad2114f762/tfrecords/train14-2071.tfrec')

#test tfrec files
TEST_FILENAMES=tf.io.gfile.glob(GCS_DS_PATH+ '/tfrecords/test*.tfrec')

def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.reshape(image, [IMAGE_SIZE,IMAGE_SIZE, 3]) # explicit size needed for TPU
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
    image = tf.image.random_hue(image,0.05)
    image = tf.image.random_saturation(image, 0, 0.05)
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
    dataset = load_dataset(VALID, labeled=True)
    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(128)
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
NUM_VALIDATION_IMAGES= count_data_items(VALID)
NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
VALIDATION_STEPS= NUM_VALIDATION_IMAGES // BATCH_SIZE
print('Dataset: {} training images, {} validation images {} unlabeled test images'.format(NUM_TRAINING_IMAGES,NUM_VALIDATION_IMAGES,NUM_TEST_IMAGES))
print("STEPS_PER_EPOCH are {}".format(STEPS_PER_EPOCH))
print("VALIDATION_STEPS are {}".format(VALIDATION_STEPS))
#training preprocessed dataset 
train_ds=get_training_dataset()
valid_ds=get_validation_dataset()
for img,label in valid_ds.take(1):
    print(img.numpy())
    print(label.numpy())
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D,Dense,Flatten
def scheduler(epoch):
  if epoch < 2:
    return 0.001
  else:
    return 0.001 * tf.math.exp(0.1 * (10 - epoch))
LR=tf.keras.callbacks.LearningRateScheduler(scheduler)
with strategy.scope():
    model=Sequential([
        InceptionResNetV2(include_top=False,input_shape=(IMAGE_SIZE,IMAGE_SIZE,3)),
        GlobalAveragePooling2D(),
        Flatten(),
        Dense(1024,activation='relu'),
        Dense(1,activation='sigmoid')
    ])
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    
    model.summary()
model.fit(train_ds,
          epochs=EPOCHS,
          steps_per_epoch=STEPS_PER_EPOCH,
          validation_data=valid_ds,
          validation_steps=VALIDATION_STEPS,
          callbacks=[LR]
         )
import matplotlib.pyplot as plt
loss=model.history.history.get('loss')
acc=model.history.history.get('accuracy')
val_loss=model.history.history.get('val_loss')
val_acc=model.history.history.get('val_accuracy')
plt.plot(loss,acc)
plt.show()
plt.plot(val_acc)
plt.show()
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