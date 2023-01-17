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
!pip install -q efficientnet
# Importing Necessary Libraries

%matplotlib inline

import tensorflow as tf

import plotly.express as px

import matplotlib.pyplot as plt

import seaborn as sns

from tqdm.notebook import tqdm

from kaggle_datasets import KaggleDatasets

from collections import Counter

import efficientnet.tfkeras as efn

import re

from tensorflow.keras import layers as L

import sklearn



sns.set_style("dark")

sns.set(rc={'figure.figsize':(12,8)})
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



print("REPLICAS: ", strategy.num_replicas_in_sync)
# Reading the dataset

dataset = pd.read_csv("../input/siim-isic-melanoma-classification/train.csv")

dataset
dataset.head()
dataset.isnull().sum()
dataset.describe()
dataset.info()
dataset.nunique()
f, axes = plt.subplots(2, 2, figsize=(12,12))

f.tight_layout() 

plt.subplots_adjust(left=0.01, wspace=0.6, hspace=0.4)

sns.countplot(y="anatom_site_general_challenge", data=dataset,  ax=axes[0][1])

sns.countplot(y="diagnosis", data=dataset,  ax=axes[0][0])

sns.countplot(x='sex', data=dataset, ax=axes[1][0])

sns.countplot("benign_malignant", data=dataset,  ax=axes[1][1])
sns.distplot(dataset['age_approx'])
sns.countplot("target", data=dataset)
dataset['target'].value_counts(normalize=True) * 10
# Showing a sample image

image = plt.imread('/kaggle/input/siim-isic-melanoma-classification/jpeg/train/ISIC_5766923.jpg')

plt.imshow(image)
w = 10

h = 10

fig = plt.figure(figsize=(15, 15))

columns = 4

rows = 4



# ax enables access to manipulate each of subplots

ax = []



for i in range(columns*rows):

    img = plt.imread('/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'+dataset['image_name'][i]+'.jpg')

    # create subplot and append to ax

    ax.append( fig.add_subplot(rows, columns, i+1) )

    # Hide grid lines

    ax[-1].grid(False)



    # Hide axes ticks

    ax[-1].set_xticks([])

    ax[-1].set_yticks([])

    ax[-1].set_title(dataset['benign_malignant'][i])  # set title

    plt.imshow(img)







plt.show()  # finally, render the plot
w = 10

h = 10

fig = plt.figure(figsize=(15, 15))

columns = 4

rows = 4



# ax enables access to manipulate each of subplots

ax = []



for i in range(columns*rows):

    img = plt.imread('/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'+dataset.loc[dataset['target'] == 1]['image_name'].values[i]+'.jpg')

    # create subplot and append to ax

    ax.append( fig.add_subplot(rows, columns, i+1) )

    # Hide grid lines

    ax[-1].grid(False)



    # Hide axes ticks

    ax[-1].set_xticks([])

    ax[-1].set_yticks([])

    ax[-1].set_title(dataset.loc[dataset['target'] == 1]['benign_malignant'].values[i])  # set title

    plt.imshow(img)







plt.show()  # finally, render the plot
dataset.isnull().sum()
dataset.loc[dataset.isnull().any(axis=1)]
dataset = dataset.dropna(axis=0)

dataset.isnull().sum()
cleaned_dataset = dataset.copy()

cleaned_dataset
cleaned_dataset.sex = cleaned_dataset.sex.replace({'male':0, 'female':1})

cleaned_dataset = cleaned_dataset.join(pd.get_dummies(cleaned_dataset.anatom_site_general_challenge))

cleaned_dataset = cleaned_dataset.join(pd.get_dummies(cleaned_dataset.diagnosis))
pd.options.display.max_rows = 999

cleaned_dataset = cleaned_dataset.reset_index()

cleaned_dataset.head(35)
# For tf.dataset

AUTO = tf.data.experimental.AUTOTUNE



# Data access

GCS_PATH = KaggleDatasets().get_gcs_path('siim-isic-melanoma-classification')
TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/tfrecords/train*.tfrec')

TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/tfrecords/test*.tfrec')



CLASSES = [0,1]   

IMAGE_SIZE = [1024, 1024]

BATCH_SIZE = 8 * strategy.num_replicas_in_sync
def decode_image(image_data):

    image = tf.image.decode_jpeg(image_data, channels=3)

    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range

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

    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),

    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part

    # of the TPU while the TPU itself is computing gradients.

    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_brightness(image, 0.1)

    image = tf.image.random_flip_up_down(image)

    #image = tf.image.random_saturation(image, 0, 2)

    return image, label   



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



NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)

NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

print('Dataset: {} training images and {} unlabeled test images'.format(NUM_TRAINING_IMAGES,NUM_TEST_IMAGES))
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

EPOCHS = 5
def build_lrfn(lr_start=0.00001, lr_max=0.0001, 

               lr_min=0.000001, lr_rampup_epochs=20, 

               lr_sustain_epochs=0, lr_exp_decay=.8):

    lr_max = lr_max * strategy.num_replicas_in_sync



    def lrfn(epoch):

        if epoch < lr_rampup_epochs:

            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start

        elif epoch < lr_rampup_epochs + lr_sustain_epochs:

            lr = lr_max

        else:

            lr = (lr_max - lr_min) * lr_exp_decay**(epoch - lr_rampup_epochs - lr_sustain_epochs) + lr_min

        return lr

    

    return lrfn
with strategy.scope():

    efficientnetb5_model = tf.keras.Sequential([

        efn.EfficientNetB5(

            input_shape=(*IMAGE_SIZE, 3),

            #weights='imagenet',

            weights='imagenet',

            include_top=False

        ),

        L.GlobalAveragePooling2D(),

        L.Dense(1024, activation = 'relu'), 

        L.Dropout(0.3), 

        L.Dense(512, activation= 'relu'), 

        L.Dropout(0.2), 

        L.Dense(256, activation='relu'), 

        L.Dropout(0.2), 

        L.Dense(128, activation='relu'), 

        L.Dropout(0.1), 

        L.Dense(1, activation='sigmoid')

    ])
from tensorflow.keras import backend as K



# Compatible with tensorflow backend



def focal_loss(gamma=2., alpha=.25):

	def focal_loss_fixed(y_true, y_pred):

		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))

		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

	return focal_loss_fixed
efficientnetb5_model.compile(

    optimizer='adam',

    loss = focal_loss(gamma=2., alpha=.25),

    #loss = tf.keras.losses.BinaryCrossentropy(label_smoothing = 0.1),

    metrics=['binary_crossentropy', 'accuracy']

)

efficientnetb5_model.summary()
lrfn = build_lrfn()

lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)
#model.load_weights('../input/melenoma/model_weights.h5')
history = efficientnetb5_model.fit(

    get_training_dataset(), 

    epochs=EPOCHS, 

    steps_per_epoch=STEPS_PER_EPOCH,

    callbacks=[lr_schedule],

    class_weight = {0:0.50899675,1: 28.28782609}

)
# summarize history for accuracy

plt.plot(history.history['loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
# summarize history for loss

plt.plot(history.history['binary_crossentropy'])

plt.title('model crossentropy')

plt.ylabel('crossentropy')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
efficientnetb5_model.save('complete_data_efficient_model.h5')
efficientnetb5_model.save_weights('complete_data_efficient_weights.h5')
test_ds = get_test_dataset(ordered=True)

test_images_ds = test_ds.map(lambda image, idnum: image)
probabilities = efficientnetb5_model.predict(test_images_ds)
print('Generating submission.csv file...')

test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()

test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch
pred_df = pd.DataFrame({'image_name': test_ids, 'target': np.concatenate(probabilities)})

pred_df.head()
sub = pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv")

sub
del sub['target']

sub = sub.merge(pred_df, on='image_name')

#sub.to_csv('submission_label_smoothing.csv', index=False)

sub.to_csv('complete_data.csv', index=False)

sub.head()
import PIL

from PIL import Image
idx = 459

print(cleaned_dataset['image_name'][idx])

img = Image.open('/kaggle/input/siim-isic-melanoma-classification/jpeg/train/' + cleaned_dataset['image_name'][idx]+'.jpg')
img = img.resize((1024, 1024), PIL.Image.ANTIALIAS)
img = np.array(img)

img = img/255.0
img = img[np.newaxis, ...]

img.shape
efficientnetb5_model.predict(img)