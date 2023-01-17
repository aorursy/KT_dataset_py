import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import tensorflow  as tf

import pathlib

import math

from matplotlib import pyplot as plt

print(tf.__version__)

!pip install -q efficientnet

import efficientnet.tfkeras as efn
#Get gs bucket path of public Kaggle dataset

#TFRecord files should be stored in gs buckets to train on TPU

from kaggle_datasets import KaggleDatasets

path=KaggleDatasets().get_gcs_path()

train_filenames=(tf.io.gfile.glob(path+'/data/train_records/'+'*.tfrecord'))

test_filenames=(tf.io.gfile.glob(path+'/data/test_records/'+'*.tfrecord'))

validation_filenames=path+'/data/validation_record.tfrecord'
#Size of images in TFRecord files

IMAGE_SIZE=[256,256]

csv=pd.read_csv('/kaggle/input/data/labels.csv')

CLASSES=list(csv['Class'])

NO_OF_CLASSES=len(CLASSES)
# numpy and matplotlib defaults

np.set_printoptions(threshold=15, linewidth=80)



def batch_to_numpy_images_and_labels(data):

    images, labels = data

    numpy_images = images.numpy()

    numpy_labels = labels.numpy()

    if numpy_labels.dtype == object: # binary string in this case, these are image ID strings

        numpy_labels = [None for _ in enumerate(numpy_images)]

    # If no labels, only image IDs, return None for labels (this is the case for test data)

    return numpy_images, numpy_labels



def title_from_label_and_target(label, correct_label):

    if correct_label is None:

        return CLASSES[label], True

    correct = (label == correct_label)

    return "{} [{}{}{}]".format(CLASSES[label], 'OK' if correct else 'NO', u"\u2192" if not correct else '',

                                CLASSES[correct_label] if not correct else ''), correct



def display_one_flower(image, title, subplot, red=False, titlesize=16):

    plt.subplot(*subplot)

    plt.axis('off')

    plt.imshow(image)

    if len(title) > 0:

        plt.title(title, fontsize=int(titlesize) if not red else int(titlesize/1.2), color='red' if red else 'black', fontdict={'verticalalignment':'center'}, pad=int(titlesize/1.5))

    return (subplot[0], subplot[1], subplot[2]+1)

    

def display_batch_of_images(databatch, predictions=None):

    """This will work with:

    display_batch_of_images(images)

    display_batch_of_images(images, predictions)

    display_batch_of_images((images, labels))

    display_batch_of_images((images, labels), predictions)

    """

    # data

    images, labels = batch_to_numpy_images_and_labels(databatch)

    if labels is None:

        labels = [None for _ in enumerate(images)]

        

    # auto-squaring: this will drop data that does not fit into square or square-ish rectangle

    rows = int(math.sqrt(len(images)))

    cols = len(images)//rows

        

    # size and spacing

    FIGSIZE = 13.0

    SPACING = 0.1

    subplot=(rows,cols,1)

    if rows < cols:

        plt.figure(figsize=(FIGSIZE,FIGSIZE/cols*rows))

    else:

        plt.figure(figsize=(FIGSIZE/rows*cols,FIGSIZE))

    

    # display

    for i, (image, label) in enumerate(zip(images[:rows*cols], labels[:rows*cols])):

        title = '' if label is None else CLASSES[label]

        correct = True

        if predictions is not None:

            title, correct = title_from_label_and_target(predictions[i], label)

        dynamic_titlesize = FIGSIZE*SPACING/max(rows,cols)*40+3 # magic formula tested to work from 1x1 to 10x10 images

        subplot = display_one_flower(image, title, subplot, not correct, titlesize=dynamic_titlesize)

    

    #layout

    plt.tight_layout()

    if label is None and predictions is None:

        plt.subplots_adjust(wspace=0, hspace=0)

    else:

        plt.subplots_adjust(wspace=SPACING, hspace=SPACING)

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
#Create tf.data.Dataset from TFRecord files

AUTO = tf.data.experimental.AUTOTUNE

ignore_order = tf.data.Options()

ignore_order.experimental_deterministic = False

def decode_image(image_data):

    image = tf.image.decode_jpeg(image_data, channels=3)

    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range

    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU

    return image



def read_tfrecord(example):

    features={

            "image": tf.io.FixedLenFeature([], tf.string),

            "label": tf.io.FixedLenFeature([], tf.int64),}

    example = tf.io.parse_single_example(example, features)

    image = decode_image(example['image'])

    label = tf.cast(example['label'], tf.int32)

    return image, label



def load_dataset(filenames,shuffle,BATCH_SIZE,repeat=False):

    # read from TFRecords. For optimal performance, read from multiple

    # TFRecord files at once and set the option experimental_deterministic = False

    # to allow order-altering optimizations.

    dataset = tf.data.TFRecordDataset(filenames,num_parallel_reads=AUTO)

    dataset = dataset.with_options(ignore_order)

    dataset = dataset.map(read_tfrecord)

    dataset = dataset.prefetch(AUTO)

    dataset = dataset.shuffle(shuffle)

    if(repeat):

        dataset = dataset.repeat()

    dataset = dataset.batch(BATCH_SIZE)

    return dataset
#Display some images from train dataset

d=load_dataset(train_filenames,5000,30)

train_batch=iter(d)

display_batch_of_images(next(train_batch))
#Get TPU strategy

try:

  tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection

  print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])

  if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

except ValueError:

    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.
#Define model

with strategy.scope():

    enb = efn.EfficientNetB7(weights='noisy-student', include_top=False, input_shape=[*IMAGE_SIZE, 3])

    enb.trainable = True # Full Training

    

    model = tf.keras.Sequential([

        enb,

        tf.keras.layers.MaxPool2D(),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(NO_OF_CLASSES, activation='softmax')

    ])

        

model.compile(

    optimizer = tf.keras.optimizers.Adam(),

    loss = 'sparse_categorical_crossentropy',

    metrics=['sparse_categorical_accuracy']

)

model.summary()
#Create train and validation datasets

BATCH_SIZE =16 * strategy.num_replicas_in_sync

shuffle=10000

dataset=load_dataset(train_filenames,shuffle,BATCH_SIZE,True)

validation_dataset=load_dataset(validation_filenames,int(shuffle*0.5),BATCH_SIZE)
#Train the model

history=model.fit(dataset,steps_per_epoch=150,epochs=3,validation_data=validation_dataset)
#Plot the loss details

import matplotlib.pyplot as plt

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])
#Plot the accuracy details

plt.plot(history.history['sparse_categorical_accuracy'])

plt.plot(history.history['val_sparse_categorical_accuracy'])
#Display predicted images in test set

valid=load_dataset(test_filenames,2000,50)

valid=next(iter(valid))

result=model.predict(valid)

display_batch_of_images(valid,predictions=tf.argmax(result,axis=1))
#Save the model

model.save('/kaggle/working/classifier.h5')