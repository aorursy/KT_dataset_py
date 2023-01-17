# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow as tf

from kaggle_datasets import KaggleDatasets

import os

import glob

import re

%matplotlib inline

from sklearn.metrics import confusion_matrix,f1_score,precision_score,recall_score

# from sklearn.utils import class_weight

# import efficientnet.tfkeras as efn

# import keras.backend as K



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
print('The Number of Training TF Record Files:{}'.format(len(os.listdir("../input/tpu-getting-started/tfrecords-jpeg-224x224/train"))))

print('The Number of Validation TF Record Files:{}'.format(len(os.listdir("../input/tpu-getting-started/tfrecords-jpeg-224x224/val"))))

print('The Number of Testing TF Record Files:{}'.format(len(os.listdir("../input/tpu-getting-started/tfrecords-jpeg-224x224/test"))))
# Getting Count of Total Number of Train,Validation and Test Files

def count_tfrecord_examples(

        tfrecords_dir: str,

) -> int:

    """

    Counts the total number of examples in a collection of TFRecord files.



    :param tfrecords_dir: directory that is assumed to contain only TFRecord files

    :return: the total number of examples in the collection of TFRecord files

        found in the specified directory

    """



    count = 0

    for file_name in os.listdir(tfrecords_dir):

        tfrecord_path = os.path.join(tfrecords_dir, file_name)

        count += sum(1 for _ in tf.data.TFRecordDataset(tfrecord_path))



    return count
TRAIN_FILES_COUNT=count_tfrecord_examples("../input/tpu-getting-started/tfrecords-jpeg-224x224/train")

VALIDATION_FILES_COUNT=count_tfrecord_examples("../input/tpu-getting-started/tfrecords-jpeg-224x224/val")

TEST_FILES_COUNT=count_tfrecord_examples("../input/tpu-getting-started/tfrecords-jpeg-224x224/test")



print('Number of Training Image Files: {}'.format(TRAIN_FILES_COUNT))

print('Number of Validation Image Files: {}'.format(VALIDATION_FILES_COUNT))

print('Number of Testing Image Files: {}'.format(TEST_FILES_COUNT))
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
GCS_DS_PATH = KaggleDatasets().get_gcs_path()
EPOCHS = 50

BATCH_SIZE = 16 * strategy.num_replicas_in_sync

IMAGE_SIZE=[224, 224]

NUM_TRAINING_IMAGES = 12753

NUM_TEST_IMAGES = 7382

NUM_VALIDATION_IMAGES=3712

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

AUTO = tf.data.experimental.AUTOTUNE
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



    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files

    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order

    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)

    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False

    return dataset



def data_augment(image, label):

    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_saturation(image, 0, 2)

    return image, label



def get_training_dataset():

    dataset = load_dataset(tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords-jpeg-224x224/train/*.tfrec'), labeled=True)

#     dataset=dataset.map(augment_data())

    dataset = dataset.repeat() # the training dataset must repeat for several epochs

    dataset = dataset.shuffle(2048)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def get_validation_dataset(ordered=False):

    dataset = load_dataset(tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords-jpeg-224x224/val/*.tfrec'), labeled=True, ordered=False)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.cache()

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def get_test_dataset(ordered=False):

    dataset = load_dataset(tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords-jpeg-224x224/test/*.tfrec'), labeled=False, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



training_dataset = get_training_dataset()

validation_dataset = get_validation_dataset()
# Visualizing Some Random Training

def plot_images(dataset,batches):

    training_dataset = dataset.unbatch().batch(batches)

    training_batch = iter(training_dataset)

    images,labels=next(training_batch)

    

    for i in range(len(images)):

        plt.subplot(5,5,i+1)

        plt.imshow(images[i])

    plt.show()
# Training Images

plt.figure(figsize=(12,12))

plot_images(training_dataset,20)
# Number of  Classes

CLASSES = [

    'pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea', 

    'wild geranium', 'tiger lily', 'moon orchid', 'bird of paradise', 'monkshood', 

    'globe thistle', 'snapdragon', "colt's foot", 'king protea', 'spear thistle', 

    'yellow iris', 'globe-flower', 'purple coneflower', 'peruvian lily', 

    'balloon flower', 'giant white arum lily', 'fire lily', 'pincushion flower', 

    'fritillary', 'red ginger', 'grape hyacinth', 'corn poppy', 

    'prince of wales feathers', 'stemless gentian', 'artichoke', 'sweet william', 

    'carnation', 'garden phlox', 'love in the mist', 'cosmos',  'alpine sea holly', 

    'ruby-lipped cattleya', 'cape flower', 'great masterwort',  'siam tulip', 

    'lenten rose', 'barberton daisy', 'daffodil',  'sword lily', 'poinsettia', 

    'bolero deep blue',  'wallflower', 'marigold', 'buttercup', 'daisy', 

    'common dandelion', 'petunia', 'wild pansy', 'primula',  'sunflower', 

    'lilac hibiscus', 'bishop of llandaff', 'gaura',  'geranium', 'orange dahlia', 

    'pink-yellow dahlia', 'cautleya spicata',  'japanese anemone', 'black-eyed susan', 

    'silverbush', 'californian poppy',  'osteospermum', 'spring crocus', 'iris', 

    'windflower',  'tree poppy', 'gazania', 'azalea', 'water lily',  'rose', 

    'thorn apple', 'morning glory', 'passion flower',  'lotus', 'toad lily', 

    'anthurium', 'frangipani',  'clematis', 'hibiscus', 'columbine', 'desert-rose', 

    'tree mallow', 'magnolia', 'cyclamen ', 'watercress',  'canna lily', 

    'hippeastrum ', 'bee balm', 'pink quill',  'foxglove', 'bougainvillea', 

    'camellia', 'mallow',  'mexican petunia',  'bromelia', 'blanket flower', 

    'trumpet creeper',  'blackberry lily', 'common tulip', 'wild rose']
# Training and Validation Distribution

y_train = next(iter(training_dataset.unbatch().map(lambda image, label: label).batch(NUM_TRAINING_IMAGES))).numpy()

y_val = next(iter(validation_dataset.unbatch().map(lambda image, label: label).batch(NUM_VALIDATION_IMAGES))).numpy()

train_agg = np.asarray([[label, (y_train == index).sum()] for index, label in enumerate(CLASSES)])

val_agg = np.asarray([[label, (y_val == index).sum()] for index, label in enumerate(CLASSES)])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 64))

ax1 = sns.barplot(x=train_agg[...,1], y=train_agg[...,0], order=CLASSES, ax=ax1)

ax1.set_title('Train', fontsize=30)

ax1.tick_params(labelsize=16)

ax2 = sns.barplot(x=val_agg[...,1], y=val_agg[...,0], order=CLASSES, ax=ax2)

ax2.set_title('Validation', fontsize=30)

ax2.tick_params(labelsize=16)

plt.show()
# Defining Callbacks

save_best=tf.keras.callbacks.ModelCheckpoint(filepath='best_model.h5',monitor='val_loss',save_best_only=True)

reduce_lr=tf.keras.callbacks.ReduceLROnPlateau(patience=10,monitor='val_accuracy',factor=0.6,min_lr=0.0000001)

# early_stopping=tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=10)

my_callbacks=[save_best,reduce_lr]
# Building the Model Under Strategy Scope

with strategy.scope():

    base_model=tf.keras.applications.DenseNet201(weights='imagenet',include_top=False,input_shape=(224,224,3))

    for layer in base_model.layers:

        layer.trainable=False

    x=tf.keras.layers.GlobalAveragePooling2D()(base_model.output)

    x=tf.keras.layers.Dropout(0.5)(x)

    out=tf.keras.layers.Dense(104,activation='softmax')(x)

    model=tf.keras.models.Model(inputs=base_model.input,outputs=out)



# Compiling the Model

# opt=tf.keras.optimizers.Adam(0.001)

model.compile(optimizer="adam",loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])



# Fitting the Model

history=model.fit(training_dataset,steps_per_epoch=NUM_TRAINING_IMAGES//BATCH_SIZE,validation_data=validation_dataset,epochs=EPOCHS,callbacks=my_callbacks)
# Saving the Model

model.save("model.h5")
# Visualizing Training and Validation Loss

print(history.history.keys())

#  "Accuracy"

plt.plot(history.history['sparse_categorical_accuracy'])

plt.plot(history.history['val_sparse_categorical_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()

# "Loss"

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
def display_confusion_matrix(cmat, score, precision, recall):

    plt.figure(figsize=(15,15))

    ax = plt.gca()

    ax.matshow(cmat, cmap='Reds')

    ax.set_xticks(range(len(CLASSES)))

    ax.set_xticklabels(CLASSES, fontdict={'fontsize': 7})

    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")

    ax.set_yticks(range(len(CLASSES)))

    ax.set_yticklabels(CLASSES, fontdict={'fontsize': 7})

    plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    titlestring = ""

    if score is not None:

        titlestring += 'f1 = {:.3f} '.format(score)

    if precision is not None:

        titlestring += '\nprecision = {:.3f} '.format(precision)

    if recall is not None:

        titlestring += '\nrecall = {:.3f} '.format(recall)

    if len(titlestring) > 0:

        ax.text(101, 1, titlestring, fontdict={'fontsize': 18, 'horizontalalignment':'right', 'verticalalignment':'top', 'color':'#804040'})

    plt.show()

    
# Plotting Confusion Matrix F1 Score

cmdataset = get_validation_dataset(ordered=True)

images_ds = cmdataset.map(lambda image, label: image)

labels_ds = cmdataset.map(lambda image, label: label).unbatch()



cm_correct_labels = next(iter(labels_ds.batch(NUM_VALIDATION_IMAGES))).numpy()

cm_probabilities = model.predict(images_ds)

cm_predictions = np.argmax(cm_probabilities, axis=-1)



labels = range(len(CLASSES))

cmat = confusion_matrix(

    cm_correct_labels,

    cm_predictions,

    labels=labels,

)

cmat = (cmat.T / cmat.sum(axis=1)).T # normalize
score = f1_score(

    cm_correct_labels,

    cm_predictions,

    labels=labels,

    average='macro',

)

precision = precision_score(

    cm_correct_labels,

    cm_predictions,

    labels=labels,

    average='macro',

)

recall = recall_score(

    cm_correct_labels,

    cm_predictions,

    labels=labels,

    average='macro',

)

display_confusion_matrix(cmat, score, precision, recall)