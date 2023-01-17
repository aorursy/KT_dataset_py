# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf



import os

import re

import seaborn as sns

import numpy as np

import pandas as pd

import math

from numpy import expand_dims

from numpy import ones

from numpy import zeros

from numpy.random import rand

from numpy.random import randint

from matplotlib import pyplot as plt



from sklearn import metrics

from sklearn.model_selection import train_test_split



import tensorflow as tf

import tensorflow.keras.layers as L

import tensorflow_addons as tfa

from sklearn.model_selection import StratifiedKFold

from tqdm import tqdm



#import efficientnet.tfkeras as efn



from kaggle_datasets import KaggleDatasets

from tensorflow.keras import backend as K

import tensorflow_addons as tfa

from numpy.random import randn
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
# The inputs are 28x28 RGB images with `channels_last` and the batch  

# size is 4.  

input_shape = (4, 384, 384, 3)

x = tf.random.normal(input_shape)

y = tf.keras.layers.Conv2D(3, (5,5), activation='relu', input_shape=input_shape[1:])(x)

print(y.shape)
tflayer = tf.keras.layers
def define_vgg16_encoder(in_shape=(384,384,3)):

    # Relu modified to LeakyRelu 

    # as described in paper works better for GAN discriminator

    # using VGG16 as backbone for this

    with strategy.scope():

        model = tf.keras.Sequential(name='encoder')



        model.add(tflayer.Conv2D(input_shape=in_shape,filters=64,kernel_size=(3,3),padding="same", activation=tflayer.LeakyReLU(0.2)))

        model.add(tflayer.Conv2D(filters=64,kernel_size=(3,3),padding="same", activation=tflayer.LeakyReLU(0.2)))

        model.add(tflayer.MaxPool2D(pool_size=(2,2),strides=(2,2)))



        model.add(tflayer.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation=tflayer.LeakyReLU(0.2)))

        model.add(tflayer.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation=tflayer.LeakyReLU(0.2)))

        model.add(tflayer.MaxPool2D(pool_size=(2,2),strides=(2,2)))



        model.add(tflayer.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation=tflayer.LeakyReLU(0.2)))

        model.add(tflayer.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation=tflayer.LeakyReLU(0.2)))

        model.add(tflayer.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation=tflayer.LeakyReLU(0.2)))

        model.add(tflayer.MaxPool2D(pool_size=(2,2),strides=(2,2)))



        model.add(tflayer.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation=tflayer.LeakyReLU(0.2)))

        model.add(tflayer.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation=tflayer.LeakyReLU(0.2)))

        model.add(tflayer.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation=tflayer.LeakyReLU(0.2)))

        model.add(tflayer.MaxPool2D(pool_size=(2,2),strides=(2,2)))



        model.add(tflayer.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation=tflayer.LeakyReLU(0.2)))

        model.add(tflayer.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation=tflayer.LeakyReLU(0.2)))

        model.add(tflayer.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation=tflayer.LeakyReLU(0.2)))

        model.add(tflayer.MaxPool2D(pool_size=(2,2),strides=(2,2)))

    

        #This is extra layer----- 

        model.add(tflayer.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation=tflayer.LeakyReLU(0.2)))

        model.add(tflayer.MaxPool2D(pool_size=(2,2),strides=(2,2)))

        # ------------------------

        #volumeSize = K.int_shape(model)

    

        model.add(tflayer.Flatten())



        model.add(tflayer.Dense(4096, activation=tflayer.LeakyReLU(0.2)))

        #model.add(tflayer.Dense(1, activation='sigmoid'))

        # compile model

        #opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)

        #model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])



        return model

        #model.add(tflayer.Dense(units=4096,activation="relu"))
encoder_model = define_vgg16_encoder((384,384,3))

encoder_model.summary()
# define the standalone generator model

def define_decoder(latent_dim):

    

    with strategy.scope():

        

        

        model = tf.keras.Sequential(name='decoder')

        # same size as just above the falt layer of discriminator

        n_nodes = 512 * 6 * 6

        model.add(tflayer.Dense(n_nodes, input_dim=latent_dim))

        model.add(tflayer.LeakyReLU(alpha=0.2))

    

        model.add(tflayer.Reshape((6, 6, 512)))

        # upsample 

        model.add(tflayer.Conv2DTranspose(512, (4,4), strides=(2,2), padding='same'))

        model.add(tflayer.LeakyReLU(alpha=0.2))



        # upsample 

        model.add(tflayer.Conv2DTranspose(512, (4,4), strides=(2,2), padding='same'))

        model.add(tflayer.LeakyReLU(alpha=0.2))



        # upsample 

        model.add(tflayer.Conv2DTranspose(512, (4,4), strides=(2,2), padding='same'))

        model.add(tflayer.LeakyReLU(alpha=0.2))

    

        # upsample 

        model.add(tflayer.Conv2DTranspose(256, (4,4), strides=(2,2), padding='same'))

        model.add(tflayer.LeakyReLU(alpha=0.2))

    

        # upsample 

        model.add(tflayer.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))

        model.add(tflayer.LeakyReLU(alpha=0.2))

    

        # upsample 

        model.add(tflayer.Conv2DTranspose(64, (4,4), strides=(2,2), padding='same'))

        model.add(tflayer.LeakyReLU(alpha=0.2))

    

        # output layer

        model.add(tflayer.Conv2D(3, (3,3), activation='sigmoid', padding='same'))

        return model
latent_dim = 4096

decoder_model = define_decoder(latent_dim)

decoder_model.summary()
with strategy.scope():

    

    inputShape = (384, 384, 3)

    inputs = tf.keras.Input(shape=inputShape)



    autoencoder = tf.keras.Model(inputs, decoder_model(encoder_model(inputs)),name="autoencoder")



    opt = tfa.optimizers.RectifiedAdam(lr=0.0003)

    autoencoder.compile(loss="mse", optimizer=opt)
faeture_list = ['image_name','target','tfrecord']



siim20_csv = pd.read_csv('../input/jpeg-melanoma-384x384/train.csv',usecols=faeture_list)

siim19_csv = pd.read_csv('../input/jpeg-isic2019-384x384/train.csv',usecols=faeture_list)
siim19_csv['year'] = '2019' 

siim20_csv['year'] = '2020'



siim_all = pd.concat([siim19_csv,siim20_csv],ignore_index = True)



train = siim_all.loc[siim_all.target == 1]

print('Number of Class 1 images ')

print(train.target.value_counts())
# REMOVE duplicate images

filter_train = train[train.tfrecord != -1 ]



idx_list = []

for img_name in filter_train.image_name.values:

    if img_name.endswith('downsampled'):

        idx = filter_train.index[filter_train['image_name'] == img_name].to_list()

        #print(str(idx) + str(len(idx)) + ':' +img_name )

        if len(idx) == 1:

            idx_list.append(idx[0])



print(len(idx_list))

filter_train = filter_train.drop(idx_list)

# shuffle the rows

filter_train.reset_index(inplace=True)



filter_train.drop('index',axis=1)



print(filter_train.head())
# Taking only 2020 images

filter_train = siim20_csv

filter_train.target.value_counts()
AUTO = tf.data.experimental.AUTOTUNE



# Data access

GCS_PATH_19 = KaggleDatasets().get_gcs_path('jpeg-isic2019-384x384')

GCS_PATH_20 = KaggleDatasets().get_gcs_path('jpeg-melanoma-384x384')



#

SEED_VALUE = 3435



# Configuration

EPOCHS = 5

BATCH_SIZE = 4 * strategy.num_replicas_in_sync

img_size = 384

IMAGE_SIZE = [img_size,img_size]
def add_gcs_path(image_id):

    

    year_nb = filter_train.loc[filter_train.image_name == image_id].year.to_numpy()[0]

    #print(year_nb)

    GCS_PATH = ''

    

    if year_nb == '2019':

        GCS_PATH = GCS_PATH_19 + '/train/' + image_id + '.jpg'

    else:

        GCS_PATH = GCS_PATH_20 + '/train/' + image_id + '.jpg'

    

    return GCS_PATH



def file_path(image_id):

    

    year_nb = filter_train.loc[filter_train.image_name == image_id].year.to_numpy()[0]

    #print(year_nb)

    GCS_PATH = ''

    

    if year_nb == '2019':

        #print('19')

        GCS_PATH = '../input/jpeg-isic2019-384x384' + '/train/' + image_id + '.jpg'

    else:

        #print('20')

        GCS_PATH = '../input/jpeg-melanoma-384x384' + '/train/' + image_id + '.jpg'

    

    return GCS_PATH
filter_train["image_path"] = filter_train["image_name"].apply(lambda x : add_gcs_path(x))

#filter_train["image_jpg_id"] = filter_train["image_name"].apply(lambda x: file_path(x))



print(filter_train.head())
# shuffle the rows

filter_train = filter_train.sample(frac=1).reset_index(drop=True)



xtrain, xval, ytrain, yval = train_test_split(filter_train["image_path"], filter_train["target"], 

                                              test_size = 0.10, stratify = filter_train["target"],

                                              random_state=SEED_VALUE)



df_train = pd.DataFrame({"image_path":xtrain, "target":ytrain})

df_val = pd.DataFrame({"image_path":xval, "target":yval})



df_train["target"] = df_train["target"].astype('int')

df_val["target"] = df_val["target"].astype('int')
train_paths = df_train.image_path.values

val_paths   = df_val.image_path.values



train_labels = df_train.target

val_labels   = df_val.target
def seed_everything(seed):

    random.seed(seed)

    np.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    tf.random.set_seed(seed)

    

def decode_image(filename, label=None, image_size=(img_size, img_size)):

    bits = tf.io.read_file(filename)

    image = tf.image.decode_jpeg(bits, channels=3)

    image = tf.cast(image, tf.float32)

    # scaling to [-1,1]

    image = image / 255.0

    image = tf.image.resize(image, size = image_size)

    

    if label is None:

        return image

    else:

        return image, image #label



def int_div_round_up(a, b):

    return (a + b - 1) // b
train_dataset = (

    tf.data.Dataset

    .from_tensor_slices((train_paths, train_labels))

    .map(decode_image, num_parallel_calls=AUTO)

    #.map(data_augment, num_parallel_calls=AUTO)

    #.map(transform, num_parallel_calls = AUTO)

    .repeat()

    .shuffle(512)

    .batch(BATCH_SIZE)

    .prefetch(AUTO))



valid_dataset = (

    tf.data.Dataset

    .from_tensor_slices((val_paths, val_labels))

    .map(decode_image, num_parallel_calls=AUTO)

    .batch(BATCH_SIZE)

    .prefetch(AUTO))



NUM_TRAINING_IMAGES = df_train.shape[0]

NUM_VALIDATION_IMAGES = df_val.shape[0]



STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

VALIDATION_STEPS = int_div_round_up(NUM_VALIDATION_IMAGES, BATCH_SIZE)

print('Dataset: {} training images, {} validation images'.format(NUM_TRAINING_IMAGES, NUM_VALIDATION_IMAGES))
history1 = autoencoder.fit(

    train_dataset, 

    validation_data = valid_dataset,

    validation_steps = VALIDATION_STEPS,

    epochs=EPOCHS,

    steps_per_epoch = STEPS_PER_EPOCH

    )
#acc = history1.history['mse']

#val_acc = history1.history['val_mse']



loss = history1.history['loss']

val_loss = history1.history['val_loss']



epochs = range(len(loss))



#plt.plot(epochs, acc, 'b', label='Training mse')

#plt.plot(epochs, val_acc, 'r', label='Validation mse')

#plt.title('Training and validation accuracy')

#plt.legend()

 

plt.figure()

 

plt.plot(epochs, loss, 'b', label='Training loss')

plt.plot(epochs, val_loss, 'r', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()
autoencoder.save('siim_autoencoder_v1.h5')