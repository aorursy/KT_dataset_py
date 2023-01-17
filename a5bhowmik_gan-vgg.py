# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from random import random

import os

from numpy import zeros

from numpy import ones

from numpy.random import randn

from numpy.random import randint

import tensorflow as tf

import cv2

#import keras

from kaggle_datasets import KaggleDatasets

#for dirname, _, filenames in os.walk('/kaggle/input'):

 #   for filename in filenames:

  #      print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from tensorflow.keras.preprocessing.image import *

from matplotlib import pyplot

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv2DTranspose, Reshape, LeakyReLU, Dropout, Input, BatchNormalization

from tensorflow.keras.layers import Concatenate, Activation

from tensorflow.keras.models import Sequential, Model, load_model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.initializers import RandomNormal

import re



%matplotlib inline
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
tf.test.is_gpu_available()
input_path = '/kaggle/input/mela-domain-data/'

output_path = '/kaggle/working/'

#train_data = pd.read_csv(input_path+'/train.csv', usecols = ['image_name', 'target'])

#test_data = pd.read_csv(input_path+'/test.csv')



#NUM_TEST_IMAGES = test_data.shape[0]
IMAGE_SIZE=[256, 256]

def decode_image(image_data):

    image = tf.image.decode_jpeg(image_data, channels=3)

    image = tf.cast(image, tf.float32) 

    image = (image - 127.5) /127.5   # convert image to floats in [-1, 1] range

    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU

    return image





def load_dataset(filenames, labeled=True, ordered=False):

    ignore_order = tf.data.Options()

    if not ordered:

        ignore_order.experimental_deterministic = False # disable order, increase speed



    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files

    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order

    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)

    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False

    return dataset





def read_labeled_tfrecord(example):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "target": tf.io.FixedLenFeature([], tf.int64)  # shape [] means single element

    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    #label = tf.cast(example['class'], tf.int32)

    #label = tf.cast(example['target'], tf.int32)

    return image #, label # returns a dataset of (image, label) pairs

#GCS_PATH = KaggleDatasets().get_gcs_path('siim-isic-melanoma-classification')

GCS_PATH = KaggleDatasets().get_gcs_path('mela-domain-data')

AUTO = tf.data.experimental.AUTOTUNE



BATCH_SIZE = 1

EPOCHS = 20



trainB = tf.io.gfile.glob(GCS_PATH+'/train00-584.tfrec')

trainA = tf.io.gfile.glob(GCS_PATH+'/train*-[0-9][0-9][0-9][0-9].tfrec')



trainA_dataset = load_dataset(trainA, labeled=True).repeat()#.shuffle(2048).batch(1)

trainB_dataset = load_dataset(trainB, labeled=True).shuffle(584)#.batch(1)

    

def count_data_items(filenames):

    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]

    return np.sum(n)
def build_discriminator(image_shape,name):

    init = RandomNormal(stddev=0.02)

    

    in_image = Input(shape=image_shape)

    d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(in_image)

    d = LeakyReLU(alpha=0.2)(d)

    

    d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)

    d = BatchNormalization()(d)

    d = LeakyReLU(alpha=0.2)(d)

               

    d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)

    d = BatchNormalization()(d)

    d = LeakyReLU(alpha=0.2)(d)

    

    d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)

    d = BatchNormalization()(d)

    d = LeakyReLU(alpha=0.2)(d)

    

    d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)

    d = BatchNormalization()(d)

    d = LeakyReLU(alpha=0.2)(d)

    

    output = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)

    

    model = Model(in_image, output, name=name)

    model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])

    

    return model
#Define Residual Block



def resnet_block(n_filters, input_layer):

    init = RandomNormal(stddev=0.02)

    

    r = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(input_layer)

    r = BatchNormalization()(r)

    r = Activation('relu')(r)

    

    r = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(r)

    r = BatchNormalization()(r)

    

    r = Concatenate()([r, input_layer])

    

    return r
def build_generator(name, image_shape, n_resnet=9):

    init = RandomNormal(stddev=0.02)

    

    in_image = Input(shape=image_shape)

    

    g = Conv2D(64, (7,7), padding='same', kernel_initializer=init)(in_image)

    g = BatchNormalization()(g)

    g = Activation('relu')(g)

    

    g = Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)

    g = BatchNormalization()(g)

    g = Activation('relu')(g)

    

    g = Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)

    g = BatchNormalization()(g)

    g = Activation('relu')(g)

    

    for _ in range(n_resnet):

        g = resnet_block(256, g)

    

    g = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)

    g = BatchNormalization()(g)

    g = Activation('relu')(g)

    

    g = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)

    g = BatchNormalization()(g)

    g = Activation('relu')(g) 

    

    g = Conv2D(3, (7,7), padding='same', kernel_initializer=init)(g)

    g = BatchNormalization()(g)

    out_image = Activation('tanh')(g)

    

    

    model = Model(in_image, out_image, name=name)

    return model
def composite_model(g_model_1, d_model, g_model_2, image_shape,name):

    with strategy.scope():

        g_model_1.trainable = True

        d_model.trainable = False

        g_model_2.trainable = False

        

        input_gen = Input(shape=image_shape)

        gen1_out = g_model_1(input_gen)

        output_d = d_model(gen1_out)

        

        input_id = Input(shape=image_shape)

        output_id = g_model_1(input_id)

        

        output_f = g_model_2(gen1_out)

        

        gen2_out = g_model_2(input_id)

        output_b = g_model_1(gen2_out)

        

        model = Model([input_gen, input_id], [output_d, output_id, output_f, output_b],name=name)

        

        opt = Adam(lr=0.0002, beta_1=0.5)

        

        model.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=opt)

        

        return model
# select real samples

def generate_real_samples(X, n_samples, patch_shape):

    y = tf.ones([n_samples, patch_shape, patch_shape, 1])

    return X, y
def generate_fake_samples(g_model, dataset, patch_shape):

    x = g_model.predict(dataset)

    y = tf.zeros([len(x), patch_shape, patch_shape, 1])

    return x, y
def save_models(step, g_model_AtoB, g_model_BtoA, d_model_A, d_model_B):

    filename1 = 'g_model_AtoB_%06d.h5' % (step+1)

    g_model_AtoB.save(filename1)

    filename2 = 'g_model_BtoA_%06d.h5' % (step+1)

    g_model_BtoA.save(filename2)

    filename3 = 'd_model_A_%06d.h5' % (step+1)

    d_model_A.save(filename3)

    filename4 = 'd_model_B_%06d.h5' % (step+1)

    d_model_B.save(filename4)

    print('>Saved: %s , %s , %s and %s' % (filename1, filename2, filename3, filename4))
def summarize_performance(step, g_model, trainX, name, n_samples=1):

    X_in, _ = generate_real_samples(trainX, n_samples, 0)

    X_out, _ = generate_fake_samples(g_model, X_in, 0)

    X_in = (X_in + 1) / 2.0

    X_out = (X_out + 1) / 2.0

    for i in range(n_samples):

        pyplot.subplot(2, n_samples, 1 + i)

        pyplot.axis('off')

        pyplot.imshow(X_in[i])

    for i in range(n_samples):

        pyplot.subplot(2, n_samples, 1 + n_samples + i)

        pyplot.axis('off')

        pyplot.imshow(X_out[i])

    filename1 = '%s_generated_plot_%06d.png' % (name, (step+1))

    pyplot.savefig(filename1)

    pyplot.close()
def update_image_pool(pool, images, max_size=50):

    selected = list()

    for image in images:

        if len(pool) < max_size:

            pool.append(image)

            selected.append(image)

        elif random() < 0.5:

            selected.append(image)

        else:

            ix = np.random.randint(0, len(pool))

            selected.append(pool[ix])

            pool[ix] = image

    return np.asarray(selected)
def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, train_ben, train_mal):

    n_batch = 1

    n_patch = d_model_A.output_shape[1]

    poolA, poolB = list(), list()

    bat_per_epo = 584

    train_a = []

    batch_B = train_mal.batch(1)

    

    

    for i in range(EPOCHS):

        train_a_dataset = train_ben.skip(i*4000).take(4000)

        for x in train_a_dataset:

            train_a.append(x.numpy())

            

        a = tf.random.uniform(shape=[bat_per_epo,], maxval=4000, dtype=tf.int32)

        train_a = np.array(train_a)

        batch_A = tf.gather(train_a, a)

        train_a = []  

        batch_A = tf.data.Dataset.from_tensor_slices((batch_A)).batch(1)

            

            

        for trainA,trainB in tf.data.Dataset.zip((batch_A, batch_B)):

            X_realA, y_realA = generate_real_samples(trainA, n_batch, n_patch)

            X_realB, y_realB = generate_real_samples(trainB, n_batch, n_patch)

            X_fakeA, y_fakeA = generate_fake_samples(g_model_BtoA, X_realB, n_patch)

            X_fakeB, y_fakeB = generate_fake_samples(g_model_AtoB, X_realA, n_patch)

            X_fakeA = update_image_pool(poolA, X_fakeA)

            X_fakeB = update_image_pool(poolB, X_fakeB)

            g_loss2, _, _, _, _ = c_model_BtoA.train_on_batch([X_realB, X_realA], [y_realA,X_realA, X_realB, X_realA])

            dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)

            dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)

            g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB,X_realB, X_realA, X_realB])

            dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)

            dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)

            print('>%d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]' % (i+1, dA_loss1,dA_loss2, dB_loss1,dB_loss2, g_loss1,g_loss2))

            

        if (i+1) % 10 == 0:

            summarize_performance(i, g_model_AtoB, trainA, 'AtoB')

            summarize_performance(i, g_model_BtoA, trainB, 'BtoA')

        if (i+1) % 20 == 0: 

            save_models(i, g_model_AtoB, g_model_BtoA, d_model_A, d_model_B)

                        
image_shape = (256, 256, 3)

MODEL_PATH = '/kaggle/input/gan-model'

#KaggleDatasets().get_gcs_path('gan-model')





#g_model_AtoB = build_generator('g_model_AtoB',image_shape)

#g_model_BtoA = build_generator('g_model_BtoA',image_shape)

g_model_AtoB = load_model(MODEL_PATH+'/g_model_AtoB_000040.h5')

g_model_BtoA = load_model(MODEL_PATH+'/g_model_BtoA_000040.h5')

#d_model_A = build_discriminator(image_shape, 'd_model_A')

#d_model_B = build_discriminator(image_shape, 'd_model_B')

d_model_A = load_model(MODEL_PATH+'/d_model_A_000040.h5')

d_model_B = load_model(MODEL_PATH+'/d_model_B_000040.h5')

c_model_AtoB = composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape, 'c_model_AtoB')

c_model_BtoA = composite_model(g_model_BtoA, d_model_A, g_model_AtoB, image_shape, 'c_model_BtoA')
train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, trainA_dataset, trainB_dataset)
def _bytes_feature(value):

    if isinstance(value, type(tf.constant(0))):

        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



def _float_feature(value):

  """Returns a float_list from a float / double."""

  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))



def _int64_feature(value):

  """Returns an int64_list from a bool / enum / int / uint."""

  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def serialize_example(feature0, feature1):

    feature = {

      'image': _bytes_feature(feature0),

      'target': _int64_feature(feature1)

    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

    return example_proto.SerializeToString()
with tf.device('/gpu:0'):

    def create_tfrecords(dataset, count,j):

        CT = count

        with tf.io.TFRecordWriter('generated_mela%.2i-%i.tfrec'%(j,CT)) as writer:

            for k in dataset:

                img = tf.image.convert_image_dtype(k, tf.uint8)

                img = tf.image.resize(img, [256, 256], method="nearest")              

                img = tf.image.encode_jpeg(img, quality=94, optimize_size=True)

                        #img = cv2.imread(PATH+IMGS['image_name'].iloc[SIZE*j+k]+'.jpg')

                        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Fix incorrect colors

                        #img = cv2.resize(img, (256, 256))

                        #img = cv2.imencode('.jpg', img, (cv2.IMWRITE_JPEG_QUALITY, 94))[1].tostring()

                example = serialize_example(img, 1)

                writer.write(example)

#gen_dataset = np.array(gen_dataset)

#dataset = tf.data.Dataset.from_tensor_slices((gen_dataset)).unbatch()

#create_tfrecords(dataset, 2000)
#g_model_AtoB = load_model('/kaggle/input/gan-model/g_model_AtoB_000050.h5')



def generate_img(dataset):

    for i in range(10):

        gen_dataset = []

        train = dataset.skip(2071*i).batch(1)

        for j,x in enumerate(train):

            img,_ = generate_fake_samples(g_model_AtoB, x, 16)

            img = (img + 1) / 2.0

            gen_dataset.append(img)

            #filename1 = '%s_generated_plot_%06d.png' % (name, (step+1))

            if((j+1) %2071 == 0):

                break

            else:

                pass

            

        gen_dataset = np.array(gen_dataset)

        test = tf.data.Dataset.from_tensor_slices((gen_dataset)).unbatch()

        create_tfrecords(test, 2071,i)
generate_img(trainA_dataset)