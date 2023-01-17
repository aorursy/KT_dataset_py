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
import tensorflow as tf

from kaggle_datasets import KaggleDatasets

import numpy as np

%matplotlib inline 

from matplotlib import pyplot as plt



print("Tensorflow version " + tf.__version__)
#Detect my accelerator

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
#Get my data path

GCS_DS_PATH = KaggleDatasets().get_gcs_path() # you can list the bucket with "!gsutil ls $GCS_DS_PATH"

print(GCS_DS_PATH)
#Set some parameters

IMAGE_SIZE = [192, 192] # at this size, a GPU will run out of memory. Use the TPU

EPOCHS = 25

BATCH_SIZE = 16 * strategy.num_replicas_in_sync



NUM_TRAINING_IMAGES = 12753

NUM_TEST_IMAGES = 7382

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

AUTO = tf.data.experimental.AUTOTUNE
def augmentation_flip(image,label):

    #for image,label in dataset:

    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_flip_up_down(image)

    return image, label

def augmentation_rotate(image,label):

    #for image,label in dataset:

    image = tf.image.rot90(image)

    return image, label

def augmentation_color(image,label):

    #for image,label in dataset:

    image = tf.image.random_hue(image, 0.08)

    image = tf.image.random_saturation(image, 0.6, 1.6)

    image = tf.image.random_brightness(image, 0.05)

    image = tf.image.random_contrast(image, 0.7, 1.3)

    return image, label

    
#Load my data

#This data is loaded from Kaggle and automatically sharded to maximize parallelization.

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

    print(image)

    return image, label # returns a dataset of (image, label) pairs



def read_labeled_tfrecord_For_Vis(example):

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



def get_training_dataset():

    dataset = load_dataset(tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords-jpeg-192x192/train/*.tfrec'), labeled=True)

    dataset = dataset.repeat() # the training dataset must repeat for several epochs

    dataset = dataset.shuffle(2048)

    #dataset = dataset.map(augmentation_flip)

    #dataset = dataset.map(augmentation_rotate)

    #dataset = dataset.map(augmentation_color)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def get_validation_dataset():

    dataset = load_dataset(tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords-jpeg-192x192/val/*.tfrec'), labeled=True, ordered=False)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.cache()

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def get_test_dataset(ordered=False):

    dataset = load_dataset(tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords-jpeg-192x192/test/*.tfrec'), labeled=False, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset
training_dataset = get_training_dataset()

validation_dataset = get_validation_dataset()

test_ds = get_test_dataset(ordered=True)
#Check shape of data

for i,j in training_dataset.take(3):

    print(i[0].numpy().shape, j.numpy().shape)
print("Training images: {}".format(NUM_TRAINING_IMAGES))

print("Testing images: {}".format(NUM_TEST_IMAGES))
#Examine the shape of train data

for image, label in training_dataset.take(1):

    image = image.numpy()

    label = label.numpy()

    print(len(image))

    print(image.shape)

    print(label)
#Examine training/validation data

fig=plt.figure()

fig.set_figheight(15)

fig.set_figwidth(15)

columns = 4

rows = 4

for image, label in training_dataset.take(1):

    image = image.numpy()

    label = label.numpy()

    num = int(len(image))

    for k in range(1,rows*columns):

        #plt.subplot(ct,1,1)

        #print(i.numpy().shape)

        fig.add_subplot(rows,columns,(k))

        plt.imshow(image[k])

        plt.xlabel("Flower Label: {}".format(label[k]))

        

        

    
#Examine testing data

fig=plt.figure()

fig.set_figheight(15)

fig.set_figwidth(15)

columns = 4

rows = 4

for image, ID in test_ds.take(1):

    image = image.numpy()

    ID = ID.numpy()

    num = int(len(image))

    for k in range(1,rows*columns):

        #plt.subplot(ct,1,1)

        #print(i.numpy().shape)

        fig.add_subplot(rows,columns,(k))

        plt.imshow(image[k])

        plt.xlabel("Flower ID: {}".format(ID[k]))
!pip install -q efficientnet
#Build a model on TPU (or GPU, or CPU...) with Tensorflow 2.1!

#Inception_Resnet_V2

#from keras_applications.efficientnet import EfficientNetB0



#from tensorflow.keras.applications.vgg19 import VGG19

from tensorflow.keras.applications.inception_v3 import InceptionV3

import efficientnet.tfkeras as efficientnet

with strategy.scope():    

    #pretrained_model = tf.keras.applications.InceptionResNetV2(include_top=False,weights="imagenet",input_shape=[*IMAGE_SIZE, 3])

    #pretrained_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])

    #pretrained_model = InceptionV3(weights='imagenet', include_top=False,input_shape=[*IMAGE_SIZE, 3])

    pretrained_model = tf.keras.applications.DenseNet201(weights='imagenet', include_top=False,input_shape=[*IMAGE_SIZE, 3])

    #pretrained_model = efficientnet.EfficientNetB0(include_top=False,weights="noisy-student",input_shape=[*IMAGE_SIZE, 3])

    #base_model = VGG19(weights='imagenet')

    #pretrained_model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)

    #pretrained_model = tf.keras.applications.VGG19(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])

    #pretrained_model = EfficientNetB0(include_top=False,weights="imagenet",input_shape=(192, 192, 3),backend=keras.backend,layers=keras.layers,models=keras.models,utils=keras.utils)

    pretrained_model.trainable = False # tramsfer learning

    

    model = tf.keras.Sequential([

        pretrained_model,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(104, activation='softmax')

    ])



#lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(

    #initial_learning_rate=1e-2,

    #decay_steps=10000,

    #decay_rate=0.9) 

#optimizer = tf.keras.optimizers.Adam(lr=0.0001)

        

model.compile(

    optimizer= 'Adam',

    loss = 'sparse_categorical_crossentropy',

    metrics=['sparse_categorical_accuracy']

)



model.summary()
historical = model.fit(training_dataset, 

          steps_per_epoch=STEPS_PER_EPOCH, 

          epochs=EPOCHS, 

          validation_data=validation_dataset)
# list all data in history

print(historical.history.keys())

# summarize history for accuracy

plt.plot(historical.history['sparse_categorical_accuracy'])

plt.plot(historical.history['val_sparse_categorical_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(historical.history['loss'])

plt.plot(historical.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
test_ds = get_test_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and ids, order matters.



print('Computing predictions...')

test_images_ds = test_ds.map(lambda image, idnum: image)

probabilities = model.predict(test_images_ds)

predictions = np.argmax(probabilities, axis=-1)

print(predictions)



print('Generating submission.csv file...')

test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()

test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch
np.savetxt('submission.csv', np.rec.fromarrays([test_ids, predictions]), fmt=['%s', '%d'], delimiter=',', header='id,label', comments='')