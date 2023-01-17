!pip install tensorflow==2.1.0

import tensorflow as tf

print(tf.__version__)
tf.test.is_gpu_available()
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: |https://github.com/kaggle/docker-python

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
import math

import tensorflow as tf

import matplotlib.pyplot as plt



# Corresponding changes are to be made here

# if the feature description in tf2_preprocessing.py

# is changed

feature_description = {

    'segment': tf.io.FixedLenFeature([], tf.string),

    'file': tf.io.FixedLenFeature([], tf.string),

    'num': tf.io.FixedLenFeature([], tf.int64)

}





def build_dataset(dir_path, batch_size=16, file_buffer=500*1024*1024,

                  shuffle_buffer=1024, label=1):

    '''Return a tf.data.Dataset based on all TFRecords in dir_path

    Args:

    dir_path: path to directory containing the TFRecords

    batch_size: size of batch ie #training examples per element of the dataset

    file_buffer: for TFRecords, size in bytes

    shuffle_buffer: #examples to buffer while shuffling

    label: target label for the example

    '''

    # glob pattern for files

    file_pattern = os.path.join(dir_path, '*.tfrecord')

    # stores shuffled filenames

    file_ds = tf.data.Dataset.list_files(file_pattern)

    # read from multiple files in parallel

    ds = tf.data.TFRecordDataset(file_ds,

                                 num_parallel_reads=tf.data.experimental.AUTOTUNE,

                                 buffer_size=file_buffer)

    # randomly draw examples from the shuffle buffer

    ds = ds.shuffle(buffer_size=shuffle_buffer,

                    reshuffle_each_iteration=True)

    # batch the examples

    # dropping remainder for now, trouble when parsing - adding labels

    ds = ds.batch(batch_size, drop_remainder=True)

    # parse the records into the correct types

    ds = ds.map(lambda x: _my_parser(x, label, batch_size),

                num_parallel_calls=tf.data.experimental.AUTOTUNE)

    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds





def _my_parser(examples, label, batch_size):

    '''Parses a batch of serialised tf.train.Example(s)

    Args:

    example: a batch serialised tf.train.Example(s)

    Returns:

    a tuple (segment, label)

    where segment is a tensor of shape (#in_batch, #frames, h, w, #channels)

    '''

    # ex will be a tensor of serialised tensors

    ex = tf.io.parse_example(examples, features=feature_description)

    ex['segment'] = tf.map_fn(lambda x: _parse_segment(x),

                              ex['segment'], dtype=tf.uint8)

    # ignoring filename and segment num for now

    # returns a tuple (tensor1, tensor2)

    # tensor1 is a batch of segments, tensor2 is the corresponding labels

    return (ex['segment'], tf.fill((batch_size, 1), label))





def _parse_segment(segment):

    '''Parses a segment and returns it as a tensor

    A segment is a serialised tensor of a number of encoded jpegs

    '''

    # now a tensor of encoded jpegs

    parsed = tf.io.parse_tensor(segment, out_type=tf.string)

    # now a tensor of shape (#frames, h, w, #channels)

    parsed = tf.map_fn(lambda y: tf.io.decode_jpeg(y), parsed, dtype=tf.uint8)

    return parsed





def display_segment(segment, batch_size):

    fig = plt.figure(figsize=(16, 16))

    columns = int(math.sqrt(batch_size))

    rows = math.ceil(batch_size / float(columns))

    for i in range(1, columns*rows + 1):

        img = segment[i-1]

        fig.add_subplot(rows, columns, i)

        plt.imshow(img)

    plt.show()
from tensorflow import keras

from tensorflow.keras.layers import (Input, Activation,

                                     BatchNormalization, Conv3D,

                                     LeakyReLU, Conv3DTranspose)

from tensorflow.keras.layers import MaxPool3D

from tensorflow.keras.models import Model

import tensorflow.keras.backend as K





def AutoEncoderModel():

    # encoder

    X_input = Input((16, 128, 128, 3))



    X = Conv3D(32, 3, padding='same')(X_input)

    X = BatchNormalization()(X)

    X = LeakyReLU()(X)

    X = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid')(X)

    # current shape is 8x64x64x32

    X = Conv3D(48, 3, padding='same')(X)

    X = BatchNormalization()(X)

    X = LeakyReLU()(X)

    X = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid')(X)

    # current shape is 4x32x32x48

    X = Conv3D(64, 3, padding='same')(X)

    X = BatchNormalization()(X)

    X = LeakyReLU()(X)

    X = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid')(X)

    # current shape is 2x16x16x64

    X = Conv3D(64, 3, padding='same')(X)

    X = BatchNormalization()(X)

    X = LeakyReLU()(X)

    X = MaxPool3D(pool_size=(2, 2, 2), strides=(1, 1, 1), padding='same')(X)

    # current shape is 2x16x16x64

    # decoder



    X = Conv3DTranspose(48, 2, strides=(2, 2, 2), padding='valid')(X)

    X = BatchNormalization()(X)

    X = LeakyReLU()(X)

    # current shape is 4x32x32x48

    X = Conv3DTranspose(32, 2, strides=(2, 2, 2), padding='valid')(X)

    X = BatchNormalization()(X)

    X = LeakyReLU()(X)

    # current shape is 8x64x64x32

    X = Conv3DTranspose(32, 2, strides=(2, 2, 2), padding='valid')(X)

    X = BatchNormalization()(X)

    X = LeakyReLU()(X)

    # current shape is 16x128x128x32

    X = Conv3D(3, 3, strides=(1, 1, 1), padding='same')(X)

    X = Activation('sigmoid')(X)

    # current shape is 16x128x128x3



    model = Model(inputs=X_input, outputs=X, name='AutoEncoderModel')

    return model





def custom_loss(new, original):

    reconstruction_error = K.mean(K.square(new-original))

    return reconstruction_error



autoEncoderModel = AutoEncoderModel()

opt = keras.optimizers.Adam(lr=0.001)

autoEncoderModel.compile(

    loss=custom_loss, optimizer=opt, metrics=['accuracy'])

print(autoEncoderModel.summary())
from tensorflow.keras.losses import BinaryCrossentropy

from tensorflow.keras.layers import Flatten,Dense

from tensorflow.keras import Sequential

def create_discriminator_model():



    X_input = Input((16, 128, 128, 3))



    # not sure about the axis in batch norm

    # do we also add dropout after batchnorm/pooling?



    # Convolutional Layers

    # changed the no of filters

    model= Sequential()

    model.add(Conv3D(filters=48, kernel_size=(2, 2, 2), padding="same",input_shape=(16, 128, 128, 3)))

    model.add(BatchNormalization())

#     model.add(Activation('relu'))

    model.add(LeakyReLU(alpha=0.2))

    model.add(MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))



    model.add(Conv3D(filters=64, kernel_size=(2, 2, 2), padding="same"))

    model.add(BatchNormalization())

#     model.add(Activation('relu'))

    model.add(LeakyReLU(alpha=0.2))

    model.add(MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))



    model.add(Conv3D(filters=128, kernel_size=(2, 2, 2), padding="same"))

    model.add(BatchNormalization())

#     model.add(Activation('relu'))

    model.add(LeakyReLU(alpha=0.2))

    model.add(MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))



    model.add(Conv3D(filters=128, kernel_size=(2, 2, 2), padding="same"))

    model.add(BatchNormalization())

#     model.add(Activation('relu'))

    model.add(LeakyReLU(alpha=0.2))

    model.add(MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))



    # to add the 5th layer change the cap to 32 frames



    # X=Conv3D(filters=256,kernel_size=(2,2,2),padding="same")(X)

    # X=BatchNormalization()(X)

    # X=Activation('relu')(X)

    # X=MaxPool3D(pool_size=(2,2,2),strides=(2,2,2))(X)



    # Fully connected layers



    model.add(Flatten())



    model.add(Dense(256, activation='relu'))

    # add batch norm to dense layer

    model.add(BatchNormalization())

    # activation done with loss fn

    # for numerical stability

    model.add(Dense(1, activation='sigmoid'))



    return model



discriminator = create_discriminator_model()

opt = keras.optimizers.Adam(lr=0.001)

loss = BinaryCrossentropy()

discriminator.compile(loss=loss,

                      optimizer=opt,

                      metrics=['accuracy'])

print(discriminator.summary())
import tensorflow as tf

class GAN():

    def __init__(self, mini_batch_size):

        

        self.image_shape=(16,128,128,3)

        

        learning_rate=0.0002

        beta_1=0.5

        

        opt=keras.optimizers.Adam(learning_rate,beta_1)

        opt1=keras.optimizers.Adam(learning_rate,beta_1)

        opt_slow=keras.optimizers.Adam(lr=0.0001)

        

        #Build and compile the discriminator

        self.discriminator=create_discriminator_model()

        self.discriminator.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])

        #Build and compile the generator

        self.generator=AutoEncoderModel()

        

        self.generator.compile(loss='mse',optimizer=opt_slow)



        #the generator takes a video as input and generates a modified video

        z = Input(shape=(self.image_shape))

        img = self.generator(z)

        self.discriminator.trainable = False

        validity = self.discriminator(img)

        self.combined = Model(z, validity)

        self.combined.compile(loss='binary_crossentropy', optimizer=opt1,metrics=['accuracy'])

        self.dir_path = '/kaggle/input/ucf-crime-training-subset/tfrecords2/'

        self.ds = build_dataset(self.dir_path, batch_size=mini_batch_size,file_buffer=512*1024)



    def train(self,epochs,mini_batch_size):

        #this function will need to be added later

        tf.summary.trace_off()

        for epoch in range(epochs):

            d_loss_sum=tf.zeros(2)

            reconstruct_error_sum=0

            g_loss_sum=tf.zeros(2)

            no_of_minibatches=0

            for minibatch,labels in self.ds:

                # ---------------------

                #  Train Discriminator

                # ---------------------

                #normalize inputs

                no_of_minibatches+=1

                minibatch=tf.cast(tf.math.divide(minibatch,255), tf.float32)

                gen_vids=self.generator.predict(minibatch)

                #might have to combine these to improve batch norm

                d_loss_real=self.discriminator.train_on_batch(minibatch,tf.ones((mini_batch_size,1)))

                d_loss_fake=self.discriminator.train_on_batch(gen_vids,tf.zeros((mini_batch_size,1)))

                d_loss=0.5*tf.math.add(d_loss_real,d_loss_fake)

                # ---------------------

                #  Train Generator

                # ---------------------

                # The generator wants the discriminator to label the generated samples as valid (ones)

                valid_y = tf.ones((mini_batch_size,1))

                # Train the generator

                g_loss = self.combined.train_on_batch(minibatch,valid_y)

                

                reconstruct_error=self.generator.train_on_batch(minibatch,minibatch)

#                 reconstruct_error=((minibatch-gen_vids)**2)/mini_batch_size

                d_loss_sum+=d_loss

                g_loss_sum+=g_loss

                reconstruct_error_sum+=reconstruct_error

            print(no_of_minibatches)

            

            #change save type 

            #model.save_model('filename')

            #model = load_model('filename')

            

            #existingModel.save_weights('weightsfile.h5')

            #existingModel.load_weights('weightsfile.h5')

            

            self.combined.save_weights('/kaggle/working/weights_leaky_relu_epoch%d.h5' %(epoch+21))

            

            

            g_loss=g_loss_sum/no_of_minibatches

            d_loss=d_loss_sum/no_of_minibatches

            reconstruct_error=reconstruct_error_sum/no_of_minibatches

            # Plot the progress

            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f, accuracy %.2f%% from which %f is combined loss and %f is reconstruction loss]" % (epoch+4, d_loss[0], 100*d_loss[1], g_loss[0]+reconstruct_error,g_loss[1]*100,g_loss[0],reconstruct_error))

        tf.summary.trace_on()



# BATCH SIZE WAS MOVED TO INIT, PROBABLY NOT THE BEST WAY TO DO IT

gan = GAN(16)

#gan.combined.load_weights('/kaggle/input/weightsanomaly-detection-04/weights_epoch4')

gan.combined.load_weights('../input/saved-models/weights_leaky_relu_epoch20.h5')

print(gan.combined.summary())

print(gan.discriminator.summary())

print(gan.generator.summary())

gan.train(10,16)