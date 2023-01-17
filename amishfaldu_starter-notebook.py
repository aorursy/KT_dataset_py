from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

user_credential = user_secrets.get_gcloud_credential()

user_secrets.set_tensorflow_credential(user_credential)
from kaggle_datasets import KaggleDatasets

GCS_DS_PATH = KaggleDatasets().get_gcs_path('cifar10dataset')
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (Dense, Conv2D, Flatten, Dropout, BatchNormalization, Activation, SeparableConv2D)

from tensorflow.keras.optimizers import Adam, SGD

import matplotlib.pyplot as plt
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
# The training and validation tfrecord have same feature set 

# i.e., image in bytes and corresponding labels in int64 format



PATH = GCS_DS_PATH

AUTO = tf.data.experimental.AUTOTUNE

BATCH_SIZE = 128 * 4

CATEGORIES = ['airplane','automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']



train_features = {

    'image':tf.io.FixedLenFeature([], dtype=tf.string),

    'label':tf.io.FixedLenFeature([], dtype=tf.int64),

}



# The test tfrecord features same as above with change instead of label we have id of that image.

test_features = {

    'image':tf.io.FixedLenFeature([], dtype=tf.string),

    'id':tf.io.FixedLenFeature([], dtype=tf.int64),

}





def load_train_example(example):

    example = tf.io.parse_single_example(example, train_features)

    image = tf.io.parse_tensor(example['image'], out_type=tf.float32)[:32, :32, :3] # image is a serialized tensor.

    image = tf.reshape(tf.clip_by_value(image, 0, 1), shape=(32, 32, 3)) # As it is serialized we need to reshape.

    return image, tf.one_hot(example['label'], depth=10) # labels needed to be one-hot encoded.



def load_test_example(example):

    example = tf.io.parse_single_example(example, test_features)

    image = tf.io.parse_tensor(example['image'], out_type=tf.float32)[:32, :32, :3]

    image = tf.reshape(tf.clip_by_value(image, 0, 1), shape=(32, 32, 3))

    return image, example['id']

    

train_data = tf.data.TFRecordDataset(PATH + "/preprocessed_train_data.tfrecord",

                                     compression_type='GZIP')

val_data = tf.data.TFRecordDataset(PATH + "/preprocessed_val_data.tfrecord",

                                   compression_type='GZIP')



train_data = train_data.map(load_train_example, num_parallel_calls=AUTO)

# Shuffle the train data and batch it. Also don't forget to prefetch to avoid bottleneck performance.

train_data = train_data.shuffle(buffer_size=100000, seed=1).batch(BATCH_SIZE).prefetch(AUTO)



val_data = val_data.map(load_train_example, num_parallel_calls=AUTO).cache().shuffle(buffer_size=100000, seed=2)

val_data = val_data.batch(BATCH_SIZE).prefetch(AUTO)



test_data = tf.data.TFRecordDataset(PATH + '/test_data.tfrecord')

test_data =test_data.map(load_test_example, num_parallel_calls=AUTO)

test_data = test_data.batch(BATCH_SIZE).prefetch(AUTO)
def plot_images(data, labels, grid, has_categories=True):

    _, axes = plt.subplots(grid[0], grid[1], figsize=(20,15), gridspec_kw={'hspace':.01, 'wspace':.01})

    if len(labels.shape) > 1:

        print("Converting One-Hot encoded labels to Categorical labels")

        labels = tf.argmax(labels, axis=1)

    for img, label, ax in zip(data, labels, axes.ravel()):

        ax.imshow(img)

        ax.axis('off')

        if has_categories:

            ax.set_title(CATEGORIES[int(label)])

        else:

            ax.set_title(int(label))
for i in train_data:

    sample1 = i

    break
for i in val_data:

    sample2 = i

    break
for i in test_data:

    sample3 = i

    break
# 32 images = 4 rows * 8 columns

plot_images(data=sample1[0], labels=sample1[1], grid=[4, 8])
# 32 images = 4 rows * 8 columns

plot_images(data=sample2[0], labels=sample2[1], grid=[4, 8])
# 32 images = 4 rows * 8 columns

plot_images(data=sample3[0], labels=sample3[1], grid=[4, 8], has_categories=False)
from tqdm import tqdm



data = None

for i in tqdm(val_data):

    data = pd.concat([data, pd.DataFrame(np.argmax(i[1].numpy(), axis=1))], axis=0)

    

print(data[0].value_counts())

print()



data = None

for i in tqdm(train_data):

    data = pd.concat([data, pd.DataFrame(np.argmax(i[1].numpy(), axis=1))], axis=0)

    

print(data[0].value_counts())



del data
def get_model():

    with strategy.scope():

        ki = 'glorot_normal'

        activation = tf.nn.leaky_relu



        inputs = tf.keras.Input(shape=(32, 32, 3))

        x = Conv2D(filters=16, kernel_size=(1,1), strides=(1,1), padding='same', 

                activation=None, kernel_initializer=ki)(inputs)

        x = BatchNormalization()(x)

        x = Activation(activation)(x)

        x = Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding='same', 

                activation=None, kernel_initializer=ki)(x)

        x = BatchNormalization()(x)

        x1 = Activation(activation)(x)







        x = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), padding='same', 

                activation=None, kernel_initializer=ki)(x1)

        x = BatchNormalization()(x)

        x = Activation(activation)(x)

        x = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same', 

                activation=None, kernel_initializer=ki)(x)

        x = BatchNormalization()(x)







        # skip connection - 1

        x1 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same', 

                activation=None, kernel_initializer=ki)(x1)

        x1 = BatchNormalization()(x1)

        x1 = Activation(activation)(x + x1)







        x = Conv2D(filters=64, kernel_size=(1,1), strides=(1,1), padding='same', 

                activation=None, kernel_initializer=ki)(x1)

        x = BatchNormalization()(x)

        x = Activation(activation)(x)

        x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', 

                activation=None, kernel_initializer=ki)(x)

        x = BatchNormalization()(x)

        

        

        

        # Recalibrate Filters - 1

        aux = SeparableConv2D(filters=32, kernel_size=(32,32), strides=(1,1), padding='valid', 

                activation=None, kernel_initializer=ki)(x)

        shape = tf.shape(aux) + (0, 0, 0, 32)

        aux = BatchNormalization()(aux)

        aux = Activation(activation)(aux)

        aux = Flatten()(aux)

        aux = Dense(4, activation=None)(aux)

        aux = Activation(activation)(aux)

        aux = Dense(64, activation='softmax')(aux)

        x *= tf.reshape(aux, shape=shape)







        # skip connection - 2

        x1 = Conv2D(filters=64, kernel_size=(1,1), strides=(1,1), padding='same', 

                activation=None, kernel_initializer=ki)(x1)

        x1 = BatchNormalization()(x1)

        x1 = Activation(activation)(x + x1)







        x = Conv2D(filters=128, kernel_size=(1,1), strides=(1,1), padding='same', 

                activation=None, kernel_initializer=ki)(x1)

        x = BatchNormalization()(x)

        x = Activation(activation)(x)

        x = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', 

                activation=None, kernel_initializer=ki)(x)

        x = BatchNormalization()(x)

        

        

                        

        # Recalibrate Filters - 2

        aux = SeparableConv2D(filters=64, kernel_size=(32,32), strides=(1,1), padding='valid', 

                activation=None, kernel_initializer=ki)(x)

        shape = tf.shape(aux) + (0, 0, 0, 64)

        aux = BatchNormalization()(aux)

        aux = Activation(activation)(aux)

        aux = Flatten()(aux)

        aux = Dense(8, activation=None)(aux)

        aux = Activation(activation)(aux)

        aux = Dense(128, activation='softmax')(aux)

        x *= tf.reshape(aux, shape=shape)







        # skip connection - 3

        x1 = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', 

                activation=None, kernel_initializer=ki)(x1)

        x1 = BatchNormalization()(x1)

        x1 = Activation(activation)(x + x1)







        x = Conv2D(filters=64, kernel_size=(1,1), strides=(1,1), padding='same', 

                activation=None, kernel_initializer=ki)(x1)

        x = BatchNormalization()(x)

        x = Activation(activation)(x)         

        x = Conv2D(filters=64, kernel_size=(32,32), strides=(1,1), padding='valid', 

                activation=None, kernel_initializer=ki)(x)

        x = BatchNormalization()(x)

        x = Activation(activation)(x)







        x = Flatten()(x)    

        output = Dense(10, activation='softmax')(x)



        model = tf.keras.models.Model(inputs=[inputs], outputs=[output])



        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), 

                    optimizer=tf.keras.optimizers.Adam(learning_rate=.005), 

                    metrics=['accuracy'])



    return model



model = get_model()
model.summary()
tf.keras.utils.plot_model(model)
# Model Checkpoint to store only model which has highest validation accuracy

model_cb = tf.keras.callbacks.ModelCheckpoint('cifar10_model1.h5', save_best_only=True, verbose=1)

# Learning Rate Scheduler to update learning rate.



# I have set patience to 2 which means that it will wait for 2 consecutive epochs to improve validation loss.

# if it fails then learning rate is reduced by factor of .6.

lr_cb = tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=.6, min_delta=.005, verbose=1)



history = model.fit(train_data, validation_data = val_data, epochs=35, 

                    callbacks=[model_cb, lr_cb]) 
y_test = model.predict(test_data, verbose=1)

y_test = tf.argmax(y_test, axis=1)
for i in test_data:

    sample = i

    break
plot_images(sample[0], y_test[:BATCH_SIZE], grid=(4,8))