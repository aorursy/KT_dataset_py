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
# Configuration

EPOCHS = 3

BATCH_SIZE = 4 # * strategy.num_replicas_in_sync

img_size = 192

IMAGE_SIZE = [img_size,img_size]
def define_vgg16_discriminator(in_shape=(img_size,img_size,3)):

    # Relu modified to LeakyRelu 

    # as described in paper works better for GAN discriminator

    # using VGG16 as backbone for this

    with strategy.scope():

        model = tf.keras.Sequential()

        tflayer = tf.keras.layers



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

        #model.add(tflayer.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation=tflayer.LeakyReLU(0.2)))

        #model.add(tflayer.MaxPool2D(pool_size=(2,2),strides=(2,2)))

        # ------------------------

    

        model.add(tflayer.Flatten())



        model.add(tflayer.Dense(4096, activation=tflayer.LeakyReLU(0.2)))

        model.add(tflayer.Dense(1, activation='sigmoid'))

        # compile model

        opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)

        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])



        return model

    #model.add(tflayer.Dense(units=4096,activation="relu"))
model = define_vgg16_discriminator((img_size,img_size,3))

model.summary()
faeture_list = ['image_name','target','tfrecord']



siim20_csv = pd.read_csv('../input/jpeg-melanoma-192x192/train.csv',usecols=faeture_list)

#siim19_csv = pd.read_csv('../input/jpeg-isic2019-192x192/train.csv',usecols=faeture_list)
#siim19_csv['year'] = '2019' 

siim20_csv['year'] = '2020'



#siim_all = pd.concat([siim19_csv,siim20_csv],ignore_index = True)



#train = siim_all.loc[siim_all.target == 1]

train = siim20_csv.loc[siim20_csv.target == 1]

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

filter_train = siim20_csv.loc[siim20_csv.target == 1]



# take only 10% of data

filter_train = filter_train.sample(frac = 1.0)

filter_train.target.value_counts()
AUTO = tf.data.experimental.AUTOTUNE



# Data access

GCS_PATH_19 = KaggleDatasets().get_gcs_path('jpeg-isic2019-384x384')

GCS_PATH_20 = KaggleDatasets().get_gcs_path('jpeg-melanoma-192x192')
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

filter_train["image_jpg_id"] = filter_train["image_name"].apply(lambda x: file_path(x))



print(filter_train.head())
train_paths = filter_train.image_path.values

#val_paths   = df_val.image_path.values



train_labels = filter_train.target

#val_labels   = df_val.target
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

    image = (image-127.5) / 127.5  

    image = tf.image.resize(image, size = image_size)

    

    if label is None:

        return image

    else:

        return image, label



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



NUM_TRAINING_IMAGES = filter_train.shape[0]

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE



print('Dataset: {} training images, '.format(NUM_TRAINING_IMAGES))
# just a test case

'''

for i in range(146):

    step_nb = i

    if step_nb == 0:

        startIndex = 0

        endIndex = BATCH_SIZE

        print('Start Index: {} ,End Index :{} '.format(startIndex,endIndex))

    else:

        startIndex = endIndex

        endIndex = startIndex + BATCH_SIZE

        print('Start Index: {} ,End Index :{} '.format(startIndex,endIndex))

'''
def generate_real_samples(startIndex, endIndex, half_batch):

    train = []

    for filename in train_paths[startIndex:endIndex]:

        bits = tf.io.read_file(filename)

        image = tf.image.decode_jpeg(bits, channels=3)

        image = tf.cast(image, tf.float32)

        # scaling to [-1,1]

        image = (image-127.5) / 127.5

        train.append(image)

        

    train = np.array(train)

    y = ones((half_batch, 1))

    

    return train, y

    
#X,y = generate_real_samples(0, 4, 4)

#print(X.shape)

#print(X)

#print(y)
# generate n fake samples with class labels

# batch_size is same as BATCH_SIZE

# It is because need to keep same number of images

def generate_fake_samples(batch_size):

    

# generate uniform random numbers in [0,1]

    X = rand(img_size * img_size * 3 * batch_size)

# update to have the range [-1, 1]

    X = -1 + X * 2

# reshape into a batch of color images

    X = X.reshape((batch_size, img_size, img_size, 3))

# generate 'fake' class labels (0)

    y = zeros((batch_size, 1))

    return X, y
# train the discriminator model

#img_dataset = train_dataset.enumerate(start=1)



def train_discriminator(model, n_iter=20, n_batch=BATCH_SIZE):

    half_batch = int(n_batch / 2)

    # manually enumerate epochs

    for i in range(n_iter):

        print('Epoch :' + str(i))

        step_count = 0

        for img_tuple in train_dataset.as_numpy_iterator():

            step_count = step_count+1

            print('Batch Number : '+str(step_count))

            # get randomly selected 'real' samples

            #X_real, y_real = generate_real_samples(dataset, half_batch)

            # update discriminator on real samples

            _, real_acc = model.train_on_batch(img_tuple[0], img_tuple[1])

            

            # generate 'fake' examples

            X_fake, y_fake = generate_fake_samples(half_batch)

            # update discriminator on fake samples

            _, fake_acc = model.train_on_batch(X_fake, y_fake)

            # summarize performance

            print('>%d real=%.0f%% fake=%.0f%%' % (i+1, real_acc*100, fake_acc*100))
# define the discriminator model

disc_model = define_vgg16_discriminator((img_size,img_size,3))



# fit the model

#train_discriminator(disc_model)
# define the standalone generator model

def define_generator(latent_dim):

    

    with strategy.scope():

        

        model = tf.keras.Sequential()

        # same size as just above the falt layer of discriminator

        tflayer = tf.keras.layers

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

        model.add(tflayer.Conv2DTranspose(256, (4,4), strides=(2,2), padding='same'))

        model.add(tflayer.LeakyReLU(alpha=0.2))

    

        # upsample 

        #model.add(tflayer.Conv2DTranspose(256, (4,4), strides=(2,2), padding='same'))

        #model.add(tflayer.LeakyReLU(alpha=0.2))

    

        # upsample 

        model.add(tflayer.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))

        model.add(tflayer.LeakyReLU(alpha=0.2))

    

        # upsample 

        model.add(tflayer.Conv2DTranspose(64, (4,4), strides=(2,2), padding='same'))

        model.add(tflayer.LeakyReLU(alpha=0.2))

    

        # output layer

        model.add(tflayer.Conv2D(3, (3,3), activation='tanh', padding='same'))

        return model
latent_dim = 4096

gen_model = define_generator(latent_dim)

gen_model.summary()
# generate points in latent space as input for the generator

def generate_latent_points(latent_dim, n_samples):

    # generate points in the latent space

    x_input = randn(latent_dim * n_samples)

    # reshape into a batch of inputs for the network

    x_input = x_input.reshape(n_samples, latent_dim)

    return x_input



# use the generator to generate n fake examples, with class labels

def generate_fake_samples(g_model, latent_dim, n_samples):

    # generate points in latent space

    x_input = generate_latent_points(latent_dim, n_samples)

    # predict outputs

    X = g_model.predict(x_input)

    # create 'fake' class labels (0)

    y = zeros((n_samples, 1))

    return X, y
from matplotlib import pyplot



X, _ = generate_fake_samples(gen_model, latent_dim, BATCH_SIZE)



X = (X + 1) / 2.0



# plot the generated samples

for i in range(BATCH_SIZE):

    # define subplot

    pyplot.subplot(7, 7, 1 + i)

    # turn off axis labels

    pyplot.axis('off')

    # plot single image

    pyplot.imshow(X[i])

# show the figure

pyplot.show()

# define the combined generator and discriminator model, for updating the generator

def define_gan(g_model, d_model):

    with strategy.scope():

        # make weights in the discriminator not trainable

        d_model.trainable = False

        # connect them

        model = tf.keras.Sequential()

        # add generator

        model.add(g_model)

        # add the discriminator

        model.add(d_model)

        # compile model

        opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)

        model.compile(loss='binary_crossentropy', optimizer=opt)

        return model
with strategy.scope():

    gan_model = define_gan(gen_model, disc_model)

    # summarize gan model

    gan_model.summary()
from keras.utils.vis_utils import plot_model

# plot gan model

plot_model(gan_model, to_file='gan_plot.png', show_shapes=True, show_layer_names=True)

%%time

# train the generator and discriminator

def train(g_model, d_model, gan_model, latent_dim, n_epochs=1, n_batch=128):

    step_per_epoch = int(filter_train.shape[0] / n_batch)

    half_batch = int(n_batch / 2)

    # manually enumerate epochs

    for i in range(n_epochs):

        # enumerate batches over the training set

        for j in range(step_per_epoch):

            # get randomly selected 'real' samples

            step_nb = j

            if step_nb == 0:

                startIndex = 0

                endIndex = half_batch

                #print('Epoch: {} / Start Index: {} | End Index :{} '.format(i,startIndex,endIndex))

            else:

                startIndex = endIndex

                endIndex = startIndex + half_batch

                #print('Epoch: {} / Start Index: {} ,End Index :{} '.format(i, startIndex,endIndex))

        

            # get real images

            X_real, y_real = generate_real_samples(startIndex, endIndex, half_batch)

            # update discriminator model weights

            d_loss1, _ = d_model.train_on_batch(X_real, y_real)



            # generate 'fake' examples

            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)

            

            # update discriminator model weights

            d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)

            

            # prepare points in latent space as input for the generator

            X_gan = generate_latent_points(latent_dim, n_batch)

            

            # create inverted labels for the fake samples

            y_gan = ones((n_batch, 1))

            

            # update the generator via the discriminator's error

            g_loss = gan_model.train_on_batch(X_gan, y_gan)

            

            # summarize loss on this batch

            print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %

                (i+1, j+1, step_per_epoch, d_loss1, d_loss2, g_loss))
%%time

# Start training of model

train(gen_model, disc_model, gan_model, latent_dim, n_epochs=3, n_batch=BATCH_SIZE)
gen_model.save('generator_model_192.h5')

disc_model.save('discriminator_model_192.h5')

gan_model.save('gan_model_192.h5')