import numpy as np

import pandas as pd

import tensorflow as tf

import glob,os

import pickle
print(f'tensorflow version : {tf.__version__}')

keras = tf.keras

layers = keras.layers

TRAIN_PATH = glob.glob(r'/kaggle/input/cifar10-python/cifar-10-batches-py/data*')

TEST_PATH = [r'/kaggle/input/cifar10-python/cifar-10-batches-py/test_batch']

BATCH_META = r'/kaggle/input/cifar10-python/cifar-10-batches-py/batches.meta'

BATCH_SIZE = 100

TRAIN_SIZE = 50000

TEST_SIZE = 10000

EPOCH = 35

AUTOTUNE = tf.data.experimental.AUTOTUNE
def load_data(path_lib,dataset_size):

    image = []

    label = []

    for path in path_lib:

        with open(path,'rb') as file:

            dataset = pickle.load(file,encoding='latin1')

            x = dataset['data']

            y = dataset['labels']

            image.append(x)

            label.append(y)

    image = np.concatenate(image,axis=0)

    label = np.concatenate(label,axis=0)

    image = np.reshape(image,[dataset_size,3,32,32])

    image = np.moveaxis(image,1,2)

    image = np.moveaxis(image,2,3)

    label = np.array(label)

    return image,label



def load_meta(path):

    with open(path,'rb') as file:

        dictionary = pickle.load(file,encoding='latin1')

    label_to_name = dict((index,name) for index,name in enumerate(dictionary['label_names']))

    return label_to_name
dictionary = load_meta(BATCH_META)

test_image,test_label = load_data(TEST_PATH,TEST_SIZE)

train_image,train_label = load_data(TRAIN_PATH,TRAIN_SIZE)

train_dataset = tf.data.Dataset.from_tensor_slices((train_image,train_label))

test_dataset = tf.data.Dataset.from_tensor_slices((test_image,test_label))



train_dataset = train_dataset.shuffle(TRAIN_SIZE).repeat().batch(BATCH_SIZE).prefetch(AUTOTUNE)

test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)
optimizer = keras.optimizers.Adam(1e-3)

loss = keras.losses.SparseCategoricalCrossentropy()

metrics = keras.metrics.SparseCategoricalAccuracy()

learning_rate_callback = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',

                                                           factor=0.5,

                                                           patience=5,

                                                           min_lr=1e-5)

base_network = keras.applications.DenseNet201(weights='imagenet',include_top=False,input_shape=[128,128,3])

#base_network = efn.EfficientNetB7(include_top=False,input_shape=(128,128,3),weights='imagenet')

network = keras.Sequential([

    layers.UpSampling2D(size=[2,2],input_shape=[32,32,3]),

    layers.UpSampling2D(size=[2,2]),

    base_network,

    layers.GlobalAveragePooling2D(),

    layers.Dense(2048),

    layers.BatchNormalization(),

    layers.ReLU(),

    layers.Dense(512),

    layers.BatchNormalization(),

    layers.ReLU(),

    layers.Dense(10,activation='softmax')

])

network.summary()
network.compile(optimizer=optimizer,

                loss=loss,

                metrics=[metrics])

network.fit(train_dataset,

            epochs=EPOCH,

            steps_per_epoch=TRAIN_SIZE//BATCH_SIZE,

            validation_data=test_dataset,

            validation_steps=TEST_SIZE//BATCH_SIZE,

            callbacks=[learning_rate_callback])

network.save(r'Network.h5')
print('Done')