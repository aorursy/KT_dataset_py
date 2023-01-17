from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
user_credential = user_secrets.get_gcloud_credential()
user_secrets.set_tensorflow_credential(user_credential)
# Set your own project id here
PROJECT_ID = 'marine-copilot-286613'
from google.cloud import storage
storage_client = storage.Client(project=PROJECT_ID)

for bucket in storage_client.list_buckets():
    print(bucket)
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from skimage.color import rgb2lab, lab2rgb
from skimage.io import imread, imshow
import keras.backend as K

from kaggle_datasets import KaggleDatasets
BATCH_SIZE = 16
NUM_EPOCHS = 100
IMAGE_SIZE = (256, 256)
DEVIE = 'TPU'
GCS_PATH        = 'gs://image-colorization-tf-records/records'
train_files      = np.sort(np.array(tf.io.gfile.glob(GCS_PATH + '/train*.tfrecord')))
validation_files = np.sort(np.array(tf.io.gfile.glob(GCS_PATH + '/validation*.tfrecord')))
print("Tensorflow version " + tf.__version__)
AUTO = tf.data.experimental.AUTOTUNE

### Loading data

try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()
    
REPLICAS = strategy.num_replicas_in_sync
print("REPLICAS: ", REPLICAS)
!cp -r ../input/image-colorization-code/model.py ../input/image-colorization-code/freeze.py ../input/image-colorization-code/utils .
delta = 1
def l_delta_loss(y_true, y_pred):
    smaller = K.cast(K.abs(y_true - y_pred) < delta, tf.float32)
    bigger = 1 - smaller
    loss = K.sum(smaller * K.square(y_true - y_pred)) / 2 + delta * K.sum(bigger * (K.abs(y_true - y_pred) - (delta / 2)))
    return loss
def PSNR(y_true, y_pred):
    return tf.image.psnr(a=y_true, b=y_pred, max_val=2)
from utils.generate_local_hints_tf import LocalHintsGenerator
from utils.rgb_to_lab_tf import rgb_to_lab

feature_description = {
    'image_bytes': tf.io.FixedLenFeature([], tf.string)
}

def _parse_function(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description), tf.zeros([256, 256, 2])

generator = LocalHintsGenerator(256, 256)

def prepare_image(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [256, 256])
    img = tf.cast(img, tf.float32)
    img = img / 255.0

    img = rgb_to_lab(img)

    l = img[:, :, 0:1] / 100
    ab = img[:, :, 1:3] / 128

    local_hints = generator.generate_local_hints(ab)
#     local_hints = generator.generate_local_hints(ab], tf.double)
    
    return {'input_2': l, 'input_1': local_hints}, ab
def get_dataset_folds(files, repeat=False):
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO)
  
    ds = ds.cache()
    
    if repeat:
        ds = ds.repeat()
#     ds = ds.shuffle(1024*8)
#     opt = tf.data.Options()
#     opt.experimental_deterministic = False
#     ds = ds.with_options(opt)
    
    ds = ds.map(lambda raw_record: _parse_function(raw_record), num_parallel_calls=AUTO)
    
    ds = ds.map(lambda input_data, ground_truth: prepare_image(input_data['image_bytes']), num_parallel_calls=AUTO)
    
    ds = ds.batch(BATCH_SIZE * REPLICAS)
    ds = ds.prefetch(AUTO)
    return ds
train_dataset = get_dataset_folds(train_files, repeat=True)
validation_dataset = get_dataset_folds(validation_files, repeat=True)
print(train_dataset.unbatch())


for x in train_dataset.unbatch().take(7):
    print(x[0]['input_2'].shape, x[0]['input_1'].shape, x[1].shape)
loaded_model = tf.keras.models.load_model('../input/image-colorization-weights/model.10-15983.52.h5', custom_objects={
    'l_delta_loss': l_delta_loss,
    'huber_loss': tf.compat.v1.losses.huber_loss,
    'PSNR': PSNR})
predictions = loaded_model.predict(validation_dataset, steps=1)
from utils.rgb_to_lab_tf import lab_to_rgb

count = 20
index = 45
for x in validation_dataset.unbatch().skip(index).take(count):
    l = x[0]['input_2']
    
    
    grayscale = np.dstack((l * 100, np.zeros_like(predictions[index])))
    concat_image = np.dstack((l * 100, predictions[index] * 128))
    real_image = np.dstack((l * 100, x[1] * 128))
    index = index + 1
    
    f = plt.figure(figsize=(12, 10))
    
    f.add_subplot(1, 3, 1)
    plt.title('Output')
    plt.imshow(lab_to_rgb(grayscale))

    f.add_subplot(1, 3, 2)
    plt.title('Output')
    plt.imshow(lab_to_rgb(concat_image))

    f.add_subplot(1, 3, 3)
    plt.title('Real')
    plt.imshow(lab_to_rgb(real_image))
