import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.ndimage

from os import listdir, mkdir
import os
import time

import tensorflow as tf

from kaggle_datasets import KaggleDatasets
from kaggle_secrets import UserSecretsClient

# Import Keras Libraries

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from tensorflow.keras import layers

print(tf.__version__)
# If you are using TPUs, execute this cell and skip the next cell

# Use the cluster resolver to communicate with the TPU

tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)

strategy = tf.distribute.experimental.TPUStrategy(tpu)
# If you are using a GPU, remove the comment and execute this cell as opposed to the previous cell
#strategy = tf.distribute.MirroredStrategy()
strategy.num_replicas_in_sync
# Set your own project id here
YOUR_PROJECT_ID = 'your_project_ID_here'

from google.cloud import bigquery
bigquery_client = bigquery.Client(project=YOUR_PROJECT_ID)
from google.cloud import storage
storage_client = storage.Client(project=YOUR_PROJECT_ID)
def create_bucket(dataset_name):
    """Creates a new bucket. https://cloud.google.com/storage/docs/ """
    bucket = storage_client.create_bucket(dataset_name)
    print('Bucket {} created'.format(bucket.name))

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket. https://cloud.google.com/storage/docs/ """
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print('File {} uploaded to {}.'.format(
        source_file_name,
        destination_blob_name))
    
def list_blobs(bucket_name):
    """Lists all the blobs in the bucket. https://cloud.google.com/storage/docs/"""
    blobs = storage_client.list_blobs(bucket_name)
    for blob in blobs:
        print(blob.name)
        
def get_blob_names(bucket_name):
    """Lists all the blobs in the bucket. https://cloud.google.com/storage/docs/"""
    blobs = storage_client.list_blobs(bucket_name)
    return blobs
        
def download_to_kaggle(bucket_name,destination_directory,file_name):
    """Takes the data from your GCS Bucket and puts it into the working directory of your Kaggle notebook"""
    os.makedirs(destination_directory, exist_ok = True)
    full_file_path = os.path.join(destination_directory, file_name)
    blobs = storage_client.list_blobs(bucket_name)
    for blob in blobs:
        blob.download_to_filename(full_file_path)
# Define the TFExample Data type for training models
# Our TFRecord format will include the CT Image and metadata of the image, including the prediction label (is PE present)

# Utilities serialize data into a TFRecord
def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# Create a dictionary describing the features.
image_feature_description = {
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
    'study_id': tf.io.FixedLenFeature([], tf.string),
    'img_name': tf.io.FixedLenFeature([], tf.string),
    'pred_label': tf.io.FixedLenFeature([], tf.int64)
}

PE_WINDOW_LEVEL = 100
PE_WINDOW_WIDTH = 700

def _parse_image_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
    single_example = tf.io.parse_single_example(example_proto, image_feature_description)
    img_height = single_example['height']
    img_width = single_example['width']
    img_bytes = tf.io.decode_raw(single_example['image_raw'],out_type='float64')
    resized_image = tf.reshape(img_bytes, (img_height,img_width))
    windowed_image = CT_window(resized_image, PE_WINDOW_LEVEL,PE_WINDOW_WIDTH )
    sample_image = tf.reshape(windowed_image, (img_height,img_width,1))
    mtd = dict()
    mtd['width'] = single_example['width']
    mtd['height'] = single_example['height']
    #mtd['study_id'] = tf.io.decode_base64(single_example['study_id'])
    #mtd['img_name'] = tf.io.decode_base64(single_example['img_name'])
    mtd['pred_label'] = single_example['pred_label']
    struct = {
    'img': sample_image,
    'img_mtd': mtd
    } 
    return struct


def read_tf_dataset(storage_file_path):
    encoded_image_dataset = tf.data.TFRecordDataset(storage_file_path, compression_type="GZIP")
    record_structs = encoded_image_dataset.map(_parse_image_function)
    return record_structs

def CT_window(img, WL=50, WW=350):
    upper, lower = WL+WW//2, WL-WW//2
    X = tf.clip_by_value(img, lower, upper)
    X = X - tf.math.reduce_min(X)
    X = X / tf.math.reduce_max(X)
    return X
## GLOBALS AND CONSTANTS
BUFFER_SIZE=120
GLOBAL_BATCH_SIZE=1024
NUM_REPLICAS=strategy.num_replicas_in_sync
BATCH_SIZE_PER_REPLICA = GLOBAL_BATCH_SIZE // NUM_REPLICAS

EPOCHS=3
LEARNING_RATE=0.00005
BETA_1=0.1

# provide a file name where checkpoints will be stored.
experiment_number = '2'
# setup a bucket for saving checkpoints
your_bucket_name = 'your_bucket_name_here'
checkpoint_path = 'gs://'+your_bucket_name+'/training_checkpoints/exp' + experiment_number
checkpoint_prefix  = checkpoint_path + '/_ckpt'

from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
user_credential = user_secrets.get_gcloud_credential()
user_secrets.set_tensorflow_credential(user_credential)
#from kaggle_datasets import KaggleDatasets
GCS_PATH = KaggleDatasets().get_gcs_path('rsna-pe-window-mixed')
GCS_PATH
# get a list of directories
# strip the first 5 chars, that is the "gs://" prefix
bucket_name = GCS_PATH[5:]
bucket_name

filenames = []
blobs = storage_client.list_blobs(bucket_name)
for blob in blobs:
    filenames.append('gs://{}/{}'.format(bucket_name,blob.name))
filenames.__len__()
filenames[0]
g_dataset = read_tf_dataset(filenames)
subset = g_dataset.take(1)
test_image = []
for struct in subset.as_numpy_iterator():
    #struct = g_dataset.get_next()
    img_mtd = struct["img_mtd"]
    img_bytes = struct["img"]
    test_image = img_bytes.reshape(512,512)
    #print("img_name = {}, pred_label = {}, image_shape = {}".format(img_mtd["img_name"], img_mtd["pred_label"], img_bytes.shape))
    fig, ax = plt.subplots(1,2,figsize=(20,3))
    ax[0].set_title("PE Specific CT-scan")
    ax[0].imshow(test_image, cmap="bone")
    ax[1].set_title("Pixelarray distribution");
    sns.distplot(test_image.flatten(), ax=ax[1]);
    #for img_data in g_dataset["mtd"]:
    #    print("img_name = {}, pred_label = {}".format(img_data["img_name"], img_data["pred_label"]))
## Define the Discriminator Network - This example is for a 512 x 512 image. 

def make_discriminator_model():
    model = tf.keras.Sequential()
    #model.add(layers.Conv2D(64, (5, 5), strides=(2), padding='same',input_shape=[512, 512, 1]))
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[512, 512, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Conv2D(512, (5, 5), strides=(2,2 ), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    #model.add(layers.Dense(1))
    model.add(layers.Dense(1, activation='relu'))

    return model
test_image.shape
test_img = test_image.reshape(1,512,512,1)
discriminator = make_discriminator_model()
# provide the image we just generated, and get the decisio score. It should be near zero, since we provided a noie image
decision = discriminator(test_img)
print (decision)
# define loss function
with strategy.scope():
    loss_object = tf.keras.losses.BinaryCrossentropy(
    from_logits=True,
    reduction=tf.keras.losses.Reduction.NONE)

    def compute_loss(labels, predictions):
        per_example_loss = loss_object(labels, predictions)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)
# define loss metrics

with strategy.scope():
    test_loss = tf.keras.metrics.Mean(name='test_loss')

    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
# create global variables in the strategy scope
with strategy.scope():
    discriminator = make_discriminator_model()
    discriminator_optimizer = tf.keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=BETA_1, amsgrad=False)
    
with strategy.scope():
        
    def restore_models( gan_checkpoint, checkpoint_directory ):
        status = gan_checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))
        return status

# Define Train Step
    @tf.function
    def train_step(tf_records):
        with tf.GradientTape() as disc_tape:
            images = tf_records["img"]
            reshaped_images = tf.reshape(images,(BATCH_SIZE_PER_REPLICA,512,512,1))
            labels = tf_records["img_mtd"]["pred_label"]
            labels = tf.dtypes.cast(labels, tf.float32)
            labels = tf.reshape(labels, (BATCH_SIZE_PER_REPLICA,1))
            model_output = discriminator(reshaped_images, training=True)
            loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)
            per_example_loss = loss_object( labels, model_output)
            disc_loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)
           
        step_loss = 0.1 * disc_loss  
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
       
        return step_loss
GLOBAL_BATCH_SIZE
BATCH_SIZE_PER_REPLICA
with strategy.scope():
  # `run` replicates the provided computation and runs it
  # with the distributed input. Note the use of the ReduceOp.SUM in the strategy.reduce operation
    @tf.function
    def distributed_train_step(dataset_inputs):
        per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
        #test_loss = 0.5
        reduced_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,axis=None)
        return reduced_loss
    
    #tf.function
    def distributed_test_step(dataset_inputs):
        return strategy.run(test_step, args=(dataset_inputs,))

    
def train_loop( num_epochs, input_dataset ):
        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0.0
            start_time = time.time()
           
            # if you are testing, use a subset
            #subset = input_dataset.take(32)
            #train_dataset = subset.batch(GLOBAL_BATCH_SIZE, drop_remainder=True).cache()
            
            # when you are ready to run an epoch on the whole 10,000 images comment the line below. 
            train_dataset = input_dataset.batch(GLOBAL_BATCH_SIZE, drop_remainder=True).cache()
            
            # Distribute the dataset among the several cores
            dist_train_dataset = strategy.experimental_distribute_dataset(train_dataset)
            for x in dist_train_dataset:
                total_loss += distributed_train_step(x)
                # print a * for every step of GLOBAL_BATH_SIZE images
                print("*",end='')
                num_batches += 1
            train_loss = total_loss / num_batches
            end_time = time.time()
            elapsed_time = end_time - start_time 
            img_per_second = num_batches * GLOBAL_BATCH_SIZE /elapsed_time
            # print how many images/sec we processed
            print("training speed = {} images per second".format(img_per_second))
            
        return elapsed_time, train_loss
# local function to drive training
def train( num_epochs, status_interval, check_option):

    num_processed = 0
    input_dataset = read_tf_dataset(filenames)
    while num_processed < num_epochs:
        print("Training Epoch #{}".format(num_processed))
        epoch_time, train_loss = train_loop(status_interval, input_dataset)
        template = ("Epoch {}, Loss: {}, elapsed time in epoch: {}")
        print (template.format(num_processed, train_loss, epoch_time))
        num_processed += status_interval
        
        if check_option == 1:
            template = ("saving checkpoint: Epoch {}, Loss: {}, elapsed time in last epoch: {}")
            print (template.format(num_processed, train_loss, epoch_time))
            gan_checkpoint.save(gan_checkpoint_prefix)    
train(3,1,0)