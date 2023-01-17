# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install -q efficientnet

import tensorflow as tf

import efficientnet.tfkeras as efn

import matplotlib.pyplot as plt

import glob

from kaggle_datasets import KaggleDatasets

keras = tf.keras

layers = keras.layers

EPOCH = 50

DATASET_SIZE = 16465

TRAIN_SIZE = 12753

VAL_SIZE = 3712

TEST_SIZE = 7382

AUTO = tf.data.experimental.AUTOTUNE

ignore_order = tf.data.Options()

ignore_order.experimental_deterministic = False

try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.

    print('Running on TPU',tpu.master())

except ValueError:

    tpu = None

    

if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.



BATCH_SIZE = 16 * strategy.num_replicas_in_sync

print(f'tensorflow version : {tf.__version__}')

print("REPLICAS: ", strategy.num_replicas_in_sync)
dataset_filenames = tf.io.gfile.glob(r'gs://kds-b2e6cdbc4af76dcf0363776c09c12fe46872cab211d1de9f60ec7aec/tfrecords-jpeg-512x512/*a*/*.tfrec')

train_filenames = tf.io.gfile.glob(r'gs://kds-b2e6cdbc4af76dcf0363776c09c12fe46872cab211d1de9f60ec7aec/tfrecords-jpeg-512x512/train/*.tfrec')

val_filenames = tf.io.gfile.glob(r'gs://kds-b2e6cdbc4af76dcf0363776c09c12fe46872cab211d1de9f60ec7aec/tfrecords-jpeg-512x512/val/*.tfrec')

test_filenames = tf.io.gfile.glob(r'gs://kds-b2e6cdbc4af76dcf0363776c09c12fe46872cab211d1de9f60ec7aec/tfrecords-jpeg-512x512/test/*.tfrec')

#count_dataset = np.sum([int(path.split('-')[-1].split('.')[0]) for path in dataset_filenames])

#count_train = np.sum([int(path.split('-')[-1].split('.')[0]) for path in train_filenames])

#count_val = np.sum([int(path.split('-')[-1].split('.')[0]) for path in val_filenames])

#print(count_dataset)

#print(count_train)

#print(count_val)

dataset = tf.data.TFRecordDataset(dataset_filenames)

dataset = dataset.with_options(ignore_order)

dataset_train = tf.data.TFRecordDataset(train_filenames)

dataset_train = dataset_train.with_options(ignore_order)

dataset_val = tf.data.TFRecordDataset(val_filenames)

dataset_val = dataset_val.with_options(ignore_order)

dataset_test = tf.data.TFRecordDataset(test_filenames)

dataset_test = dataset_test.with_options(ignore_order)
feature_description = {

    'class':tf.io.FixedLenFeature([],tf.int64),

    'image':tf.io.FixedLenFeature([],tf.string)

}

test_feature_description = {

    'id':tf.io.FixedLenFeature([],tf.string),

    'image':tf.io.FixedLenFeature([],tf.string)

}

def dataset_decode(data):

    decode_data = tf.io.parse_single_example(data,feature_description)

    label = decode_data['class']

    image = tf.image.decode_jpeg(decode_data['image'],channels=3)

    image = tf.reshape(image,[512,512,3])

    image = tf.image.random_flip_left_right(image)

    image = tf.cast(image,tf.float32)

    image = (image - 127.5) / 127.5

    return image,label

def test_dataset_decode(data):

    decode_data = tf.io.parse_single_example(data,test_feature_description)

    ID = decode_data['id']

    image = tf.image.decode_jpeg(decode_data['image'],channels=3)

    image = tf.reshape(image,[512,512,3])

    image = tf.cast(image,tf.float32)

    image = (image - 127.5) / 127.5

    return ID,image
dataset = dataset.map(dataset_decode)

dataset_train = dataset_train.map(dataset_decode)

dataset_val = dataset_val.map(dataset_decode)

dataset_test = dataset_test.map(test_dataset_decode)

dataset = dataset.shuffle(DATASET_SIZE).repeat().batch(BATCH_SIZE).prefetch(AUTO)

dataset_train = dataset_train.shuffle(DATASET_SIZE).repeat().batch(BATCH_SIZE).prefetch(AUTO)

dataset_val = dataset_val.batch(BATCH_SIZE).prefetch(AUTO)

dataset_test = dataset_test.batch(BATCH_SIZE).prefetch(AUTO)

print(dataset)

print(dataset_train)

print(dataset_val)

print(dataset_test)
optimizer = keras.optimizers.Adam()

loss = keras.losses.SparseCategoricalCrossentropy()

metrics = keras.metrics.SparseCategoricalAccuracy()

lr_callback = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=3,min_lr=1e-6)

with strategy.scope():

    enet = efn.EfficientNetB7(input_shape=(512,512,3),weights='imagenet',include_top=False)

    #base_network = keras.applications.InceptionResNetV2(include_top=False,input_shape=[512,512,3])

    network = keras.Sequential()

    network.add(enet)

    network.add(layers.MaxPooling2D())

    network.add(layers.Conv2D(2560,3,padding='same'))

    network.add(layers.BatchNormalization())

    network.add(layers.ReLU())

    network.add(layers.GlobalAveragePooling2D())

    network.add(layers.Dense(1024))

    network.add(layers.BatchNormalization())

    network.add(layers.LeakyReLU())

    network.add(layers.Dense(512))

    network.add(layers.BatchNormalization())

    network.add(layers.LeakyReLU())

    network.add(layers.Dense(104,activation='softmax'))

    network.compile(optimizer=optimizer,loss=loss,metrics=[metrics])

network.summary()

network.fit(dataset_train,

            epochs=EPOCH,

            steps_per_epoch=TRAIN_SIZE//BATCH_SIZE,

            validation_data=dataset_val,

            validation_steps=VAL_SIZE//BATCH_SIZE,

            callbacks=[lr_callback])

network.save(r'./Xception.h5')
predict_csv = []

for ID,image in dataset_test:

    prediction = network.predict(image)

    ID = [item.numpy().decode('utf-8') for item in ID]

    label = tf.argmax(prediction,axis=1).numpy()

    for i in range(len(ID)):

        predict_csv.append([ID[i],label[i]])

    print('*',end='')
csv_name = ['id','label']

csv_data = pd.DataFrame(columns=csv_name,data=predict_csv)

csv_data.to_csv(r'submission.csv',index=False)
print('DONE')