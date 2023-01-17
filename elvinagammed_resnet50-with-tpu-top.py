# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas  as pd
train = pd.read_csv("../input/landmark-recognition-2020/train.csv")
train['filename'] = train.id.str[0] + "/" + train.id.str[1] + "/" + train.id.str[2] + "/" + train.id + ".jpg"
train["label"] = train['landmark_id'].astype(str)
sub = pd.read_csv("../input/landmark-recognition-2020/sample_submission.csv")
sub['filename'] = sub.id.str[0] + "/" + sub.id.str[1] + "/" + sub.id.str[2] + "/" + sub.id + "jpg"
y = train['landmark_id'].values
y
num_classes = np.max(y)
num_classes
from collections import Counter
count = Counter(y).most_common(1000)
count
k_labels = [c[0] for c in count]
train_keep = train[train['landmark_id'].isin(k_labels)]
train_keep
val_rate = 0.25
batch_size = 32
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(validation_split=val_rate)

dir = "../input/landmark-recognition-2020/train/"
train_gen = datagen.flow_from_dataframe(train_keep, directory=dir, x_col="filename", y_col="label", weight_col=None, 
                                        target_size=(256, 256), color_mode="rgb", classes=None, class_mode="categorical",
                                       batch_size=batch_size, shuffle=True, subset="training", interpolation="nearest",
                                       validate_filenames=False)
val_gen = datagen.flow_from_dataframe(train_keep, directory=dir, x_col="filename", y_col="label", weight_col=None,
                                     target_size=(256, 256), color_mode="rgb",classes=None, class_mode="categorical", 
                                     batch_size=batch_size, shuffle=True, subset="validation",interpolation="nearest", 
                                     validate_filenames=False)
from keras.applications import MobileNetV2
from keras.utils import to_categorical
from keras.layers import Dense
from keras import Model
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
import tensorflow as tf
learning_rate_reduction = ReduceLROnPlateau(monitor = 'categorical_accuracy', patience = 3, verbose = 1, 
                                           factor = 0.2, min_lr = 0.00001)

optimizer = Adam(lr = .0001, beta_1 = .9, beta_2 = .999, epsilon = None, decay = .0, amsgrad = False)

# from keras.preprocessing.image import ImageDataGenerator
# from keras.applications import MobileNetV2
# from keras.utils import to_categorical
# from keras.layers import Dense
# from keras import Model
# from keras.callbacks import ModelCheckpoint
# from keras.models import load_model
# from tensorflow.keras.applications.xception import Xception
# import tensorflow as tf
# import tensorflow.keras.layers as L
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
with strategy.scope():
    pretrained_model = tf.keras.applications.ResNet50V2(
    weights='imagenet',
    include_top=False ,
    input_shape=(256,256,3)
    )
    pretrained_model.trainable = False
    
    model = tf.keras.Sequential([
        # To a base pretrained on ImageNet to extract features from images...
        pretrained_model,
        # ... attach a new head to act as a classifier.
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1000, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss = 'categorical_crossentropy',
        metrics=['categorical_accuracy'],
    )

from kaggle_datasets import KaggleDatasets
import math, re, os, random
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

print("Tensorflow version " + tf.__version__)
GCS_DS_PATH = KaggleDatasets().get_gcs_path() # you can list the bucket with "!gsutil ls $GCS_DS_PATH"
IMAGE_SIZE = [256, 256] # at this size, a GPU will run out of memory. Use the TPU
EPOCHS = 5
# BATCH_SIZE = 16 * strategy.num_replicas_in_sync
BATCH_SIZE = 16 * 8

SEED = 42
NUM_TRAINING_IMAGES = 188811
NUM_TEST_IMAGES = 62937
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
def random_blockout(img, sl = 0.1, sh = 0.2, rl = 0.4):
    p = random.random()
    if p >= 0.25:
        w, h, c = IMAGE_SIZE[0], IMAGE_SIZE[1], 3
        origin_area = tf.cast(h * w, tf.float32)

        e_size_l = tf.cast(tf.round(tf.sqrt(origin_area * sl * rl)), tf.int32)
        e_size_h = tf.cast(tf.round(tf.sqrt(origin_area * sh / rl)), tf.int32)

        e_height_h = tf.minimum(e_size_h, h)
        e_width_h = tf.minimum(e_size_h, w)

        erase_height = tf.random.uniform(shape = [], minval = e_size_l, maxval = e_height_h, dtype = tf.int32)
        erase_width = tf.random.uniform(shape = [], minval = e_size_l, maxval = e_width_h, dtype = tf.int32)

        erase_area = tf.zeros(shape = [erase_height, erase_width, c])
        erase_area = tf.cast(erase_area, tf.uint8)

        pad_h = h - erase_height
        pad_top = tf.random.uniform(shape = [], minval = 0, maxval = pad_h, dtype = tf.int32)
        pad_bottom = pad_h - pad_top

        pad_w = w - erase_width
        pad_left = tf.random.uniform(shape = [], minval = 0, maxval = pad_w, dtype = tf.int32)
        pad_right = pad_w - pad_left

        erase_mask = tf.pad([erase_area], [[0,0],[pad_top, pad_bottom], [pad_left, pad_right], [0,0]], constant_values=1)
        erase_mask = tf.squeeze(erase_mask, axis = 0)
        erased_img = tf.multiply(tf.cast(img,tf.float32), tf.cast(erase_mask, tf.float32))

        return tf.cast(erased_img, img.dtype)
    else:
        return tf.cast(img, img.dtype)
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

    dataset = tf.data.TFRecordDataset(filenames) # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord)
    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
    return dataset

def data_augment(image, label):
    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),
    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part
    # of the TPU while the TPU itself is computing gradients.

    image = tf.image.random_flip_left_right(image, seed=SEED)
    image = random_blockout(image)
    return image, label

def get_training_dataset():
    dataset = load_dataset(tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords-jpeg-224x224/train/*.tfrec'), labeled=True)
    dataset = dataset.map(data_augment)
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset

def get_validation_dataset():
    dataset = load_dataset(tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords-jpeg-224x224/val/*.tfrec'), labeled=True, ordered=False)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    return dataset

def get_test_dataset(ordered=False):
    dataset = load_dataset(tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords-jpeg-224x224/test/*.tfrec'), labeled=False, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset
import tensorflow as tf 
import tensorflow.keras as tfk
import numpy as np 
import glob
import pandas as pd 
from skimage import io, transform
from tqdm.notebook import tqdm

tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
strategy = tf.distribute.experimental.TPUStrategy(tpu)

learning_rate = 0.01        

with strategy.scope():
    
    model = tfk.models.Sequential()
    
    pt = tfk.applications.resnet.ResNet50 

    
    ptmod = pt(include_top=False
                , weights='imagenet'
                , input_tensor=None
                , input_shape=(256, 256, 3)
                , pooling = 'avg')


    model.add(ptmod)
#     model.add(tfk.layers.Dropout(rate = 0.5))
#     model.add(GlobalAveragePooling2D())
    model.add(tfk.layers.Dense(1000))
    model.add(tfk.layers.Activation('softmax'))
    
    optimizer = tfk.optimizers.Adam(learning_rate = learning_rate)


    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    model.summary()
# training_dataset = get_training_dataset()
# validation_dataset = get_validation_dataset()

# with strategy.scope():
#     pretrained_model = tf.keras.applications.ResNet50V2(
#     weights='imagenet',
#     include_top=False ,
#     input_shape=[*IMAGE_SIZE, 3]
#     )
#     pretrained_model.trainable = False
    
#     model = tf.keras.Sequential([
#         # To a base pretrained on ImageNet to extract features from images...
#         pretrained_model,
#         # ... attach a new head to act as a classifier.
#         tf.keras.layers.GlobalAveragePooling2D(),
#         tf.keras.layers.Dense(1000, activation='softmax')
#     ])
#     model.compile(
#         optimizer='adam',
#         loss = 'sparse_categorical_crossentropy',
#         metrics=['sparse_categorical_accuracy'],
#     )

model.summary()
historical = model.fit_generator(train_gen, 
          steps_per_epoch=STEPS_PER_EPOCH, 
          epochs = EPOCHS,
          callbacks = [learning_rate_reduction],
          validation_data=val_gen)

sub = pd.read_csv("/kaggle/input/landmark-recognition-2020/sample_submission.csv")
sub["filename"] = sub.id.str[0]+"/"+sub.id.str[1]+"/"+sub.id.str[2]+"/"+sub.id+".jpg"
sub
test_gen = ImageDataGenerator().flow_from_dataframe(
    sub,
    directory="/kaggle/input/landmark-recognition-2020/test/",
    x_col="filename",
    y_col=None,
    weight_col=None,
    target_size=(256, 256),
    color_mode="rgb",
    classes=None,
    class_mode=None,
    batch_size=1,
    shuffle=True,
    subset=None,
    interpolation="nearest",
    validate_filenames=False)
y_pred_one_hot = model.predict_generator(test_gen, verbose=1, steps=len(sub))
y_pred = np.argmax(y_pred_one_hot, axis=-1)
y_prob = np.max(y_pred_one_hot, axis=-1)
print(y_pred.shape, y_prob.shape)
y_uniq = np.unique(train_keep.landmark_id.values)

y_pred = [y_uniq[Y] for Y in y_pred]
for i in range(len(sub)):
    sub.loc[i, "landmarks"] = str(y_pred[i])+" "+str(y_prob[i])
sub = sub.drop(columns="filename")
sub.to_csv("submission.csv", index=False)
sub
