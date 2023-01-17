import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

plt.style.use('ggplot')

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import tensorflow as tf

from tqdm.notebook import tqdm

from kaggle_datasets import KaggleDatasets

from collections import Counter

from tensorflow.keras import layers as L

import sklearn
!pip install -q efficientnet
train = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')

test = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv')

sample_submission = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')
train.head()
test.head()
#Plotting overall number of benign and malignant tumors

plt.style.use('fivethirtyeight')

sns.countplot(x='target', data=train, color='red')

print('Benign: {}%'.format(round(train.target.value_counts()[0]/len(train)*100.0,2)))

print('Malignant: {}%'.format(round(train.target.value_counts()[1]/len(train)*100.0,2)))
#Plotting distribution of ages by benign vs malignant tumors

sns.kdeplot(train.loc[train['target'] == 0, 'age_approx'], label = 'Benign',shade=True, color='purple')

sns.kdeplot(train.loc[train['target'] == 1, 'age_approx'], label = 'Malignant',shade=True, color='yellow')

plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages');
#Plotting distribution of ages by male vs female persons

sns.kdeplot(train.loc[train['sex'] == 'male', 'age_approx'], label = 'Male',shade=True, color='black')

sns.kdeplot(train.loc[train['sex'] == 'female', 'age_approx'], label = 'Female',shade=True, color='cyan')

plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages');
#Plotting number of tumors by affected body parts

fig= plt.figure(figsize=(10,6))

ax = sns.countplot(x="anatom_site_general_challenge", data=train, hue='sex', palette='seismic',

                  order=train['anatom_site_general_challenge'].value_counts().index)

plt.title("Affected Body Parts")
#Plotting a random image

img = plt.imread('/kaggle/input/siim-isic-melanoma-classification/jpeg/train/ISIC_5766923.jpg')

plt.imshow(img)
#Plotting benign images

w = 10

h = 10

fig = plt.figure(figsize=(15, 15))

columns = 4

rows = 4



#ax enables access to manipulate each of subplots

ax = []



for i in range(columns*rows):

    img = plt.imread('/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'+train['image_name'][i]+'.jpg')

    #creating a subplot and appending it to ax

    ax.append( fig.add_subplot(rows, columns, i+1) )

    #hiding grid lines

    ax[-1].grid(False)



    #hiding axes ticks

    ax[-1].set_xticks([])

    ax[-1].set_yticks([])

    ax[-1].set_title(train['benign_malignant'][i])

    plt.imshow(img)
#Plotting malignant images

w = 10

h = 10

fig = plt.figure(figsize=(15, 15))

columns = 4

rows = 4



ax = []



for i in range(columns*rows):

    img = plt.imread('/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'+train.loc[train['target'] == 1]['image_name'].values[i]+'.jpg')

    ax.append( fig.add_subplot(rows, columns, i+1) )

    ax[-1].grid(False)

    ax[-1].set_xticks([])

    ax[-1].set_yticks([])

    ax[-1].set_title(train.loc[train['target'] == 1]['benign_malignant'].values[i])

    plt.imshow(img)
#Checking for missing values in the train set

total = train.isnull().sum().sort_values(ascending=False)

percent = train.isnull().mean().sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head()
#Checking for missing values in the test set

total = test.isnull().sum().sort_values(ascending=False)

percent = test.isnull().mean().sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head()
#Imputing missing values

train['sex'] = train['sex'].fillna('male')

train['age_approx'] = train['age_approx'].fillna(train['age_approx'].mean())

train['anatom_site_general_challenge'] = train['anatom_site_general_challenge'].fillna('head/neck')

test['anatom_site_general_challenge'] = test['anatom_site_general_challenge'].fillna('head/neck')
#Transforming the values into categorical

from sklearn.preprocessing import LabelEncoder

enc1 = LabelEncoder()

enc2 = LabelEncoder()



train['sex'] = enc1.fit_transform(train['sex'])

train['anatom_site_general_challenge'] = enc2.fit_transform(train['anatom_site_general_challenge'])

test['anatom_site_general_challenge'] = enc2.fit_transform(test['anatom_site_general_challenge'])
#Detecting TPU

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
#For tf.dataset

AUTO = tf.data.experimental.AUTOTUNE



#Accessing the data

GCS_PATH = KaggleDatasets().get_gcs_path('siim-isic-melanoma-classification')
TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/tfrecords/train*.tfrec')

TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/tfrecords/test*.tfrec')



CLASSES = [0,1]   

IMAGE_SIZE = [1024, 1024]

BATCH_SIZE = 8 * strategy.num_replicas_in_sync
import re

def decode_image(image_data):

    image = tf.image.decode_jpeg(image_data, channels=3)

    image = tf.cast(image, tf.float32) / 255.0  #converting image to floats in [0, 1] range

    image = tf.reshape(image, [*IMAGE_SIZE, 3]) #explicit size needed for TPU

    return image



def read_labeled_tfrecord(example):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), #tf.string means bytestring

       

        "target": tf.io.FixedLenFeature([], tf.int64),  #shape [] means single element

    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    label = tf.cast(example['target'], tf.int32)

    

    return image, label #returns a dataset of (image, label) pairs



def read_unlabeled_tfrecord(example):

    UNLABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string),

        "image_name": tf.io.FixedLenFeature([], tf.string),

    }

    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    idnum = example['image_name']

    return image, idnum #returns a dataset of image(s)



def load_dataset(filenames, labeled=True, ordered=False):

    #Reading from TFRecords. For optimal performance, reading from multiple files at once and

    #disregarding data order. Order does not matter since we will be shuffling the data anyway.



    ignore_order = tf.data.Options()

    if not ordered:

        ignore_order.experimental_deterministic = False #disabling order, increasing speed

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) #automatically interleaving reads from multiple files

    dataset = dataset.with_options(ignore_order) #using data as soon as it streams in, rather than in its original order

    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)

    #returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False

    return dataset



def data_augment(image, label):

    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_brightness(image, 0.1)

    image = tf.image.random_flip_up_down(image)

    #image = tf.image.random_saturation(image, 0, 2)

    return image, label   

def get_training_dataset():

    dataset = load_dataset(TRAINING_FILENAMES, labeled=True)

    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)

    dataset = dataset.repeat() #the training dataset must repeat for several epochs

    dataset = dataset.shuffle(2048)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) #prefetching next batch while training (autotune prefetch buffer size)

    return dataset



def get_validation_dataset(ordered=False):

    dataset = load_dataset(VALIDATION_FILENAMES, labeled=True, ordered=ordered)

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



NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)

NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

print('Dataset: {} training images and {} unlabeled test images'.format(NUM_TRAINING_IMAGES,NUM_TEST_IMAGES))
#Defining the parameters

EPOCHS = 4
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
from tensorflow.keras.layers import Dense

import efficientnet.tfkeras as efn

#Defining the model

with strategy.scope():

    efficientnetb7_model = tf.keras.Sequential([

        efn.EfficientNetB7(

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



def focal_loss(gamma=2., alpha=.25):

	def focal_loss_fixed(y_true, y_pred):

		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))

		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

	return focal_loss_fixed
#Compiling the model

efficientnetb7_model.compile(

    optimizer='Adam',

    loss = focal_loss(gamma=2., alpha=.25),

    #loss = tf.keras.losses.BinaryCrossentropy(label_smoothing = 0.1),

    metrics=['binary_crossentropy', 'accuracy']

)

efficientnetb7_model.summary()
lrfn = build_lrfn()

lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)
#model.load_weights('../input/melanoma/model_weights.h5')
#Training the model

history = efficientnetb7_model.fit(

    get_training_dataset(), 

    epochs=EPOCHS, 

    steps_per_epoch=STEPS_PER_EPOCH,

    callbacks=[lr_schedule],

    class_weight = {0:0.50899675,1: 28.28782609}

)
hist = pd.DataFrame(history.history)

hist['epoch'] = history.epoch

hist.tail()
#Visualizing model loss

plt.plot(history.history['loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')
#Visualizing model crossentropy

plt.plot(history.history['binary_crossentropy'])

plt.title('model crossentropy')

plt.ylabel('crossentropy')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')
efficientnetb7_model.save('complete_data_efficient_model.h5')
efficientnetb7_model.save_weights('complete_data_efficient_weights.h5')
test_ds = get_test_dataset(ordered=True)

test_images_ds = test_ds.map(lambda image, idnum: image)
probabilities = efficientnetb7_model.predict(test_images_ds)
print('Generating submission.csv file...')

test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()

test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') #all in one batch
pred_df = pd.DataFrame({'image_name': test_ids, 'target': np.concatenate(probabilities)})

pred_df.head()
sub = pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv")

sub
del sub['target']

sub = sub.merge(pred_df, on='image_name')

#sub.to_csv('submission_label_smoothing.csv', index=False)

sub.to_csv('complete_data.csv', index=False)

sub.head()