!pip install efficientnet
import numpy as np

import pandas as pd



import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, LeakyReLU, BatchNormalization, Dropout

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.losses import CategoricalCrossentropy

from tensorflow.keras.callbacks import ReduceLROnPlateau,ModelCheckpoint

import tensorflow.keras.backend as K

import math

from tqdm import tqdm

from sklearn.model_selection import train_test_split

import efficientnet.tfkeras as efn 

import os

from matplotlib import pyplot as plt

from kaggle_datasets import KaggleDatasets
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

GCS_DS_PATH = KaggleDatasets().get_gcs_path('test-task-otp')
img_size = 256

img_dir = '/256x256/'

train_df = pd.DataFrame(columns=['image_id', 'gender'])

img_ids, genders = [], []

for file in os.listdir('/kaggle/input/test-task-otp/256x256/'):

    if file.split('_')[0] == 'female':

        img_ids.append(file)

        genders.append('f')

    if file.split('_')[0] == 'male':

        img_ids.append(file)

        genders.append('m')

    elif file.split('_')[0] == 'other':

        img_ids.append(file)

        genders.append('o')

train_df['image_id'] = img_ids

train_df['gender'] = genders

train_df['image_id'] = train_df['image_id'].apply(lambda x: GCS_DS_PATH + img_dir + x)

x = train_df['image_id']

y = pd.get_dummies(train_df['gender']).astype('int32').values

train_df['gender'].value_counts()
x_train, x_test, y_train, y_test = train_test_split(x ,y , test_size=0.05, random_state=666)
img_size = 256

HEIGHT = img_size

WIDTH = img_size

CHANNELS = 3

batch_size = 8 * strategy.num_replicas_in_sync

epochs = 20

AUTO = tf.data.experimental.AUTOTUNE
def decode_image(filename, label=None, image_size=(img_size, img_size)):

    bits = tf.io.read_file(filename)

    image = tf.image.decode_jpeg(bits, channels=3)

    image = tf.cast(image, tf.float32) / 255.0

    image = tf.image.resize(image, image_size)

    if label is None:

        return image

    else:

        return image, label

def image_augment(filename, label):

    image = decode_image(filename)

    image = tf.image.random_flip_left_right(image)

    return image, label
img = image_augment(x_train[0],label = None)

plt.imshow(img[0])
train_ds = (

    tf.data.Dataset

    .from_tensor_slices((x_train, y_train))

    .map(image_augment, num_parallel_calls=AUTO)

    .cache()

    .shuffle(512)

    .repeat()

    .batch(batch_size)

    .prefetch(AUTO)

    )



test_ds = (

    tf.data.Dataset

    .from_tensor_slices((x_test, y_test))

    .map(decode_image, num_parallel_calls=AUTO)

    .cache()

    .repeat()

    .batch(batch_size)

    .prefetch(AUTO)

    )
def build_model():

    base = efn.EfficientNetB4(weights='imagenet', input_shape=(img_size, img_size, 3), pooling = 'avg', include_top = False)

    x = base.output

    x = Dense(512, activation = 'elu')(x)

    x = BatchNormalization(axis = -1)(x)

    x = Dropout(rate = 0.5)(x)

    x = Dense(512, activation = 'elu')(x)

    x = BatchNormalization(axis = -1)(x)

    output = Dense(3, activation="softmax")(x)

    model = tf.keras.models.Model(inputs=base.input, outputs = output)

    return model
opt = tf.keras.optimizers.Adam(learning_rate=0.0005)

with strategy.scope():

    model = build_model()

    model.compile(optimizer=opt, loss=CategoricalCrossentropy(), metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', 

                              patience=3,  

                              verbose=1,  

                              factor=0.5,   

                              min_lr=0.00001)
history = model.fit(

            train_ds, 

            validation_data = test_ds, 

            steps_per_epoch=x_train.shape[0] // batch_size,            

            validation_steps=x_test.shape[0] // batch_size,    

            callbacks = [reduce_lr],

            epochs=epochs

)
fig = plt.subplots(figsize=(12,10))

plt.plot(history.history['loss'], color='b', label="training loss")

plt.plot(history.history['val_loss'], color='r', label="validation loss")

plt.legend(loc='best', shadow=True)
model.save('otp_task_b0.h5')