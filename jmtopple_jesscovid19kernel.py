#Start by loading packages.

import matplotlib.pyplot as plt

from matplotlib.pyplot import imshow



import imageio #let's you read images in as numpy arrays

import argparse

import random

import pydot

#from pydub import AudioSegment

from IPython.display import SVG

import cv2

import h5py

import random

import sys

import io

import os

import glob

import IPython

import PIL

from PIL import Image #Python Imaging Library image module

import time

import math

import numpy as np

from numpy import genfromtxt

import pandas as pd

import geopandas as gpd

import tensorflow as tf



from sklearn.metrics import confusion_matrix

import scipy.io as sio

from scipy.io import wavfile

from scipy import ndimage

from scipy import misc



#from pydub import AudioSegment

from numpy import genfromtxt



from datetime import datetime

from datetime import date



%matplotlib inline

%load_ext autoreload

%autoreload 2
# Check which version of tensorflow you're using. I'm using TF2

print(tf.__version__) #should be 1.14.0 for OLD tensorflow or 2.1.0 for NEW

print(sys.version) #see which python version, should be 3.7.7
#For TF1

'''from keras import backend as K

from keras.models import Model, load_model, Sequential

from keras.layers import merge, Conv2D, ZeroPadding2D, Dense, Dropout, Reshape, Lambda, RepeatVector

from keras.layers import Dense, Activation, Dropout, Masking, TimeDistributed, LSTM, Conv1D

from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape

from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, Concatenate, concatenate

from keras.layers import Conv1D, ZeroPadding1D, MaxPooling1D

from keras.layers import Input, Add, LeakyReLU

from keras.layers.pooling import MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D

from keras.layers.core import Lambda, Flatten, Dense

from keras.layers.embeddings import Embedding

from keras.preprocessing import sequence

from keras.callbacks import ModelCheckpoint

from keras.optimizers import Adam, SGD

from keras.initializers import glorot_uniform

from keras.engine.topology import Layer

from keras.utils.vis_utils import model_to_dot

from keras.utils import plot_model

#K.set_image_data_format('channels_first')'''



#For TF2

from tensorflow.keras.models import Model, load_model, Sequential

from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Dense, Dropout, Reshape, Lambda, RepeatVector

from tensorflow.keras.layers import Dense, Activation, Dropout, Masking, TimeDistributed, LSTM, Conv1D

from tensorflow.keras.layers import GRU, Bidirectional, BatchNormalization, Reshape

from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Activation, Input, Concatenate, concatenate

from tensorflow.keras.layers import Conv1D, ZeroPadding1D, MaxPooling1D

from tensorflow.keras.layers import Input, Add, LeakyReLU

from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D

from tensorflow.keras.layers import Lambda, Flatten, Dense

from tensorflow.keras.layers import Embedding

from tensorflow.keras.preprocessing import sequence

from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.optimizers import Adam, SGD

from tensorflow.keras.initializers import glorot_uniform

from tensorflow.keras.layers import Layer

from tensorflow.keras.utils import model_to_dot

from tensorflow.keras.utils import plot_model

from tensorflow.keras.models import load_model

#K.set_image_data_format('channels_first')
#Edit this up later

def LSTMmodel(input_shape):

    """

    Function creating the LSTM model's graph in Keras.

    

    Arguments:

    input_shape -- shape of each training example



    Returns:

    model -- a Model() instance in Keras

    """    

    

    #########################################################

    #1. Start with inputs

    #########################################################

    

    # Define the input placeholder as a tensor with shape input_shape. 

    # Think of this as your input dataset (the input card feature vectors) being fed to the graph.

    # This is supplied when you actually call this function later

    X_input = Input(shape=input_shape) 

    

    #########################################################

    #2. Assemble the model layers

    #########################################################

    

    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state

    # Be careful, the returned output should be a batch of sequences to pass on 

    # the batch of sequences state = c is the hidden memory cell, and activation/output = a

    X = LSTM(128,return_sequences=True)(X_input)

    # Add dropout with a probability of 0.5 (to improve the robustness of training)

    X = Dropout(0.2)(X)

    

    # Propagate X through another LSTM layer with 128-dimensional hidden state

    # Be careful, the returned output should be a single hidden state, not a batch of sequences.

    X = LSTM(128)(X)

    # Add dropout with a probability of 0.5

    X = Dropout(0.5)(X)

    

    # Propagate X through a Dense layer with softmax activation to get back a batch of 2-dimensional vectors.

    X = Dense(2, activation='softmax')(X)

    # Add a softmax activation

    X = Activation('softmax')(X) #output the desired Y=0/1 probabilities at the output layer

    #NOTE: LEAVE THESE AS SOFTMAXES FOR NOW

    

    #########################################################

    #3. Create model instance with the correct "inputs" and "outputs"

    #########################################################

    

    # The model takes as input an array X_inputs of feature vectors of shape (m, Ncards, Flen) defined by input_shape. 

    # It should output a softmax probability vector (the final X) of shape (m, C = 2).

    # This step creates your Keras model instance, which will be used to train/test the model.

    model = Model(inputs = X_input, outputs = X, name='LSTMmodel')

    

    

    return model
#Edit this up later

def GRUmodel(input_shape):

    """

    Function creating the GRU model's graph in Keras.

    

    Argument:

    input_shape -- shape of the model's input data (using Keras conventions)



    Returns:

    model -- Keras model instance

    """

    

    #########################################################

    #1. Start with inputs

    #########################################################

    

    # Define the input placeholder as a tensor with shape input_shape. 

    # Think of this as your input dataset (the input card feature vectors) being fed to the graph.

    # This is supplied when you actually call this function later

    X_input = Input(shape=input_shape) 

    

    #########################################################

    #2. Assemble the model layers

    #########################################################



    # Step 1: First GRU Layer

    X = GRU(units = 128, return_sequences = True)(X_input) # GRU or could use LSTM

    #return_sequences = True ensures that all the GRU's hidden states are fed to the next layer

    X = Dropout(0.2)(X) 

    X = BatchNormalization()(X) 

    

    # Step 2: Second GRU Layer

    X = GRU(units = 128)(X) # GRU or could use LSTM

    #Again, return_sequences = True means all units give output. This is many-to-many

    #If you want many-to-one instead, just use

    #X = GRU(128)(X) #many-to-one

    X = Dropout(0.2)(X) 

    X = BatchNormalization()(X)

    

    # Step 3:  Dense layer

    # Propagate X through a Dense layer with softmax activation to get back a batch of 128-dimensional vectors.

    #X = TimeDistributed(Dense(64, activation = "sigmoid"))(X) #many-to-many

    X = Dense(128, activation='relu')(X) #many-to-one let's just see how this goes with relu

     

    #Optional: Add another Dropout + dense

    # Add dropout with a probability of 0.1

    X = Dropout(0.1)(X)

    # Propagate X through a Dense layer with softmax activation to get back a batch of 128-dimensional vectors.

    #X = Dense(64, activation='sigmoid')(X)

    X = Dense(2)(X) #2 output elements between 0 and 1



    #########################################################

    #3. Create model instance with the correct "inputs" and "outputs"

    #########################################################

    

    # The model takes as input an array X_inputs of feature vectors of shape (m, Ncards, Flen) defined by input_shape. 

    # It should output a softmax probability vector (the final X) of shape (m, C = 2).

    # This step creates your Keras model instance, which will be used to train/test the model.

    model = Model(inputs = X_input, outputs = X, name='GRUmodel')

    

    return model  
#Edit this up later

def inceptionModel(input_shape):

    '''

    1D inceptionCNN base network for time domain A(t) processing.

    Takes input shape (m, t, n_aspectrows) = (m, 512, 4)

    '''

    

    # Tweak channel numbers easily

    ch00 = 8 #32 #16 #8 #12 #10 #16

    ch0 = 16 #64 #32 #16 #24 #20 #32

    ch1 = 16 #128 #64 #34 #48 #40 #64

    ch2 = 32 #256 #128 #64 #96 #80 #128

    ch3 = 64

    droprate = 0.1 #fraction of input units to use for dropout

    

    

    #########################################################

    #1. Start with inputs

    #########################################################

    

    # Define the input placeholder as a tensor with shape input_shape. 

    # Think of this as your input dataset (the input card feature vectors) being fed to the graph.

    # This is supplied when you actually call this function later

    X_inputs = Input(shape=input_shape) 

    #This will be (m,f,t,n_aspectrows)

    

    #########################################################

    #2. Assemble the model layers

    #########################################################

    

    # First Layer = the stem, bringing data into the network

    #keras.layers.Conv1D(filters, kernel_size, strides=1, padding='valid', 

    #                    data_format='channels_last', dilation_rate=1, activation=None, 

    #                    use_bias=True, kernel_initializer='glorot_uniform', 

    #                    bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 

    #                    activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

    #X = Conv1D(ch00, 1, strides=1, padding = 'valid', name='conv0')(X_inputs)

    #X = Dropout(droprate)(X)

    #X = BatchNormalization(epsilon=0.00001, name='bn0')(X)

    #X = LeakyReLU(alpha=0.1, name = 'relu0')(X)

    

    '''X = concatenate([X1_1, X1_3, X1_5, X1_pool], axis=-1)

    

    # Second Layer - might make this an inception block!

    X = Conv1D(ch1, 5, strides=2, padding = 'valid', name='conv2')(X) #switched same to valid

    X = Dropout(droprate)(X)

    X = BatchNormalization(epsilon=0.00001, name='bn2')(X)

    X = LeakyReLU(alpha=0.1, name = 'relu2')(X)

    

    # Third Layer

    X = Conv1D(ch2, 5, strides = 2, padding = 'valid', name='conv3')(X) #switched same to valid

    X = Dropout(droprate)(X)

    X = BatchNormalization(epsilon=0.00001, name='bn3')(X)

    X = LeakyReLU(alpha=0.1, name = 'relu3')(X)'''

    

    # Inception 1:

    ######################################################################

    #The 1x1 convolution maintains the incoming pixels but changes channels

    X1_1 = Conv1D(ch1, 1, name='inception_1_1_conv')(X_inputs)

    X1_1 = BatchNormalization(axis=-1, epsilon=0.00001, name='inception_1_1_bn')(X1_1)

    X1_1 = Activation('relu')(X1_1)

    

    #The 3x1 convolves 3x3 filters over the data pixels

    X1_3 = Conv1D(ch1, 1, name ='inception_1_3_conv1')(X_inputs)

    X1_3 = BatchNormalization(axis=-1, epsilon=0.00001, name = 'inception_1_3_bn1')(X1_3)

    X1_3 = Activation('relu')(X1_3)

    X1_3 = ZeroPadding1D(padding=1)(X1_3)

    X1_3 = Conv1D(ch1, 3, name='inception_1_3_conv2')(X1_3)

    X1_3 = BatchNormalization(axis=-1, epsilon=0.00001, name='inception_1_3_bn2')(X1_3)

    X1_3 = Activation('relu')(X1_3)

    

    #The 5x1 convolves 5x1 filters over the data pixels 

    X1_5 = Conv1D(ch1, 1, name='inception_1_5_conv1')(X_inputs)

    X1_5 = BatchNormalization(axis=-1, epsilon=0.00001, name='inception_1_5_bn1')(X1_5)

    X1_5 = Activation('relu')(X1_5)

    X1_5 = ZeroPadding1D(padding=2)(X1_5)

    X1_5 = Conv1D(ch1, 5, name='inception_1_5_conv2')(X1_5)

    X1_5 = BatchNormalization(axis=-1, epsilon=0.00001, name='inception_1_5_bn2')(X1_5)

    X1_5 = Activation('relu')(X1_5)

    

    #The 7x1 convolves 7x1 filters over the data pixels 

    X1_7 = Conv1D(ch1, 1, name='inception_1_7_conv1')(X_inputs)

    X1_7 = BatchNormalization(axis=-1, epsilon=0.00001, name='inception_1_7_bn1')(X1_7)

    X1_7 = Activation('relu')(X1_7)

    X1_7 = ZeroPadding1D(padding=3)(X1_7)

    X1_7 = Conv1D(ch1, 7, name='inception_1_7_conv2')(X1_7)

    X1_7 = BatchNormalization(axis=-1, epsilon=0.00001, name='inception_1_7_bn2')(X1_7)

    X1_7 = Activation('relu')(X1_7)



    #The MaxPooling layer is probably not so helpful but you never know

    X1_pool = MaxPooling1D(pool_size=3, strides=1)(X_inputs)

    X1_pool = Conv1D(ch1, 1, name='inception_1_pool_conv')(X1_pool)

    X1_pool = BatchNormalization(axis=-1, epsilon=0.00001, name='inception_1_pool_bn')(X1_pool)

    X1_pool = Activation('relu')(X1_pool)

    X1_pool = ZeroPadding1D(1)(X1_pool)

        

    # CONCATENATE them all together along the channel axis

    X = concatenate([X1_1, X1_3, X1_5, X1_7, X1_pool], axis=-1)

    # DOWNSAMPLE

    #X = MaxPooling1D(pool_size=3, strides=2)(X)

    ######################################################################

    

    # Inception 2:

    ######################################################################

    #The 1x1 convolution maintains the incoming pixels but changes channels

    X2_1 = Conv1D(ch2, 1, name='inception_2_1_conv')(X)

    X2_1 = BatchNormalization(axis=-1, epsilon=0.00001, name='inception_2_1_bn')(X2_1)

    X2_1 = Activation('relu')(X2_1)

    

    #The 3x1 convolves 3x3 filters over the data pixels

    X2_3 = Conv1D(ch2, 1, name ='inception_2_3_conv1')(X)

    X2_3 = BatchNormalization(axis=-1, epsilon=0.00001, name = 'inception_2_3_bn1')(X2_3)

    X2_3 = Activation('relu')(X2_3)

    X2_3 = ZeroPadding1D(padding=1)(X2_3)

    X2_3 = Conv1D(ch2, 3, name='inception_2_3_conv2')(X2_3)

    X2_3 = BatchNormalization(axis=-1, epsilon=0.00001, name='inception_2_3_bn2')(X2_3)

    X2_3 = Activation('relu')(X2_3)

    

    #The 5x1 convolves 5x1 filters over the data pixels 

    X2_5 = Conv1D(ch2, 1, name='inception_2_5_conv1')(X)

    X2_5 = BatchNormalization(axis=-1, epsilon=0.00001, name='inception_2_5_bn1')(X2_5)

    X2_5 = Activation('relu')(X2_5)

    X2_5 = ZeroPadding1D(padding=2)(X2_5)

    X2_5 = Conv1D(ch2, 5, name='inception_2_5_conv2')(X2_5)

    X2_5 = BatchNormalization(axis=-1, epsilon=0.00001, name='inception_2_5_bn2')(X2_5)

    X2_5 = Activation('relu')(X2_5)

    

    #The 7x1 convolves 7x1 filters over the data pixels 

    X2_7 = Conv1D(ch1, 1, name='inception_2_7_conv1')(X)

    X2_7 = BatchNormalization(axis=-1, epsilon=0.00001, name='inception_2_7_bn1')(X2_7)

    X2_7 = Activation('relu')(X2_7)

    X2_7 = ZeroPadding1D(padding=3)(X2_7)

    X2_7 = Conv1D(ch1, 7, name='inception_2_7_conv2')(X2_7)

    X2_7 = BatchNormalization(axis=-1, epsilon=0.00001, name='inception_2_7_bn2')(X2_7)

    X2_7 = Activation('relu')(X2_7)



    #The MaxPooling layer is probably not so helpful but you never know

    X2_pool = MaxPooling1D(pool_size=3, strides=1)(X)

    X2_pool = Conv1D(ch2, 1, name='inception_2_pool_conv')(X2_pool)

    X2_pool = BatchNormalization(axis=-1, epsilon=0.00001, name='inception_2_pool_bn')(X2_pool)

    X2_pool = Activation('relu')(X2_pool)

    X2_pool = ZeroPadding1D(1)(X2_pool)

        

    # CONCATENATE them all together along the channel axis

    X = concatenate([X2_1, X2_3, X2_5, X2_7, X2_pool], axis=-1)

    #X = MaxPooling1D(pool_size=3, strides=2)(X)

    ######################################################################



    # Inception 3:

    ######################################################################

    #The 1x1 convolution maintains the incoming pixels but changes channels

    X3_1 = Conv1D(ch2, 1, name='inception_3_1_conv')(X)

    X3_1 = BatchNormalization(axis=-1, epsilon=0.00001, name='inception_3_1_bn')(X3_1)

    X3_1 = Activation('relu')(X3_1)

    

    #The 3x1 convolves 3x3 filters over the data pixels

    X3_3 = Conv1D(ch2, 1, name ='inception_3_3_conv1')(X)

    X3_3 = BatchNormalization(axis=-1, epsilon=0.00001, name = 'inception_3_3_bn1')(X3_3)

    X3_3 = Activation('relu')(X3_3)

    X3_3 = ZeroPadding1D(padding=1)(X3_3)

    X3_3 = Conv1D(ch2, 3, name='inception_3_3_conv2')(X3_3)

    X3_3 = BatchNormalization(axis=-1, epsilon=0.00001, name='inception_3_3_bn2')(X3_3)

    X3_3 = Activation('relu')(X3_3)

    

    #The 5x1 convolves 5x1 filters over the data pixels 

    X3_5 = Conv1D(ch2, 1, name='inception_3_5_conv1')(X)

    X3_5 = BatchNormalization(axis=-1, epsilon=0.00001, name='inception_3_5_bn1')(X3_5)

    X3_5 = Activation('relu')(X3_5)

    X3_5 = ZeroPadding1D(padding=2)(X3_5)

    X3_5 = Conv1D(ch2, 5, name='inception_3_5_conv2')(X3_5)

    X3_5 = BatchNormalization(axis=-1, epsilon=0.00001, name='inception_3_5_bn2')(X3_5)

    X3_5 = Activation('relu')(X3_5)

    

    #The 7x1 convolves 7x1 filters over the data pixels 

    X3_7 = Conv1D(ch1, 1, name='inception_3_7_conv1')(X)

    X3_7 = BatchNormalization(axis=-1, epsilon=0.00001, name='inception_3_7_bn1')(X3_7)

    X3_7 = Activation('relu')(X3_7)

    X3_7 = ZeroPadding1D(padding=3)(X3_7)

    X3_7 = Conv1D(ch1, 7, name='inception_3_7_conv2')(X3_7)

    X3_7 = BatchNormalization(axis=-1, epsilon=0.00001, name='inception_3_7_bn2')(X3_7)

    X3_7 = Activation('relu')(X3_7)



    #The MaxPooling layer is probably not so helpful but you never know

    X3_pool = MaxPooling1D(pool_size=3, strides=1)(X)

    X3_pool = Conv1D(ch2, 1, name='inception_3_pool_conv')(X3_pool)

    X3_pool = BatchNormalization(axis=-1, epsilon=0.00001, name='inception_3_pool_bn')(X3_pool)

    X3_pool = Activation('relu')(X3_pool)

    X3_pool = ZeroPadding1D(1)(X3_pool)

        

    # CONCATENATE them all together along the channel axis

    X = concatenate([X3_1, X3_3, X3_5, X3_7, X3_pool], axis=-1)

    #X = MaxPooling1D(pool_size=3, strides=2)(X)

    ######################################################################

    

    # Inception 4:

    ######################################################################

    #The 1x1 convolution maintains the incoming pixels but changes channels

    X4_1 = Conv1D(ch2, 1, name='inception_4_1_conv')(X)

    X4_1 = BatchNormalization(axis=-1, epsilon=0.00001, name='inception_4_1_bn')(X4_1)

    X4_1 = Activation('relu')(X4_1)

    

    #The 3x1 convolves 3x3 filters over the data pixels

    X4_3 = Conv1D(ch2, 1, name ='inception_4_3_conv1')(X)

    X4_3 = BatchNormalization(axis=-1, epsilon=0.00001, name = 'inception_4_3_bn1')(X4_3)

    X4_3 = Activation('relu')(X4_3)

    X4_3 = ZeroPadding1D(padding=1)(X4_3)

    X4_3 = Conv1D(ch2, 3, name='inception_4_3_conv2')(X4_3)

    X4_3 = BatchNormalization(axis=-1, epsilon=0.00001, name='inception_4_3_bn2')(X4_3)

    X4_3 = Activation('relu')(X4_3)

    

    #The 5x1 convolves 5x1 filters over the data pixels 

    X4_5 = Conv1D(ch2, 1, name='inception_4_5_conv1')(X)

    X4_5 = BatchNormalization(axis=-1, epsilon=0.00001, name='inception_4_5_bn1')(X4_5)

    X4_5 = Activation('relu')(X4_5)

    X4_5 = ZeroPadding1D(padding=2)(X4_5)

    X4_5 = Conv1D(ch2, 5, name='inception_4_5_conv2')(X4_5)

    X4_5 = BatchNormalization(axis=-1, epsilon=0.00001, name='inception_4_5_bn2')(X4_5)

    X4_5 = Activation('relu')(X4_5)

    

    #The 7x1 convolves 7x1 filters over the data pixels 

    X4_7 = Conv1D(ch1, 1, name='inception_4_7_conv1')(X)

    X4_7 = BatchNormalization(axis=-1, epsilon=0.00001, name='inception_4_7_bn1')(X4_7)

    X4_7 = Activation('relu')(X4_7)

    X4_7 = ZeroPadding1D(padding=3)(X4_7)

    X4_7 = Conv1D(ch1, 7, name='inception_4_7_conv2')(X4_7)

    X4_7 = BatchNormalization(axis=-1, epsilon=0.00001, name='inception_4_7_bn2')(X4_7)

    X4_7 = Activation('relu')(X4_7)



    #The MaxPooling layer is probably not so helpful but you never know

    X4_pool = MaxPooling1D(pool_size=3, strides=1)(X)

    X4_pool = Conv1D(ch2, 1, name='inception_4_pool_conv')(X4_pool)

    X4_pool = BatchNormalization(axis=-1, epsilon=0.00001, name='inception_4_pool_bn')(X4_pool)

    X4_pool = Activation('relu')(X4_pool)

    X4_pool = ZeroPadding1D(1)(X4_pool)

        

    # CONCATENATE them all together along the channel axis

    X = concatenate([X4_1, X4_3, X4_5, X4_7, X4_pool], axis=-1)

    #X = MaxPooling1D(pool_size=3, strides=2)(X)

    ######################################################################

    

    # Last Conv layer

    X = Conv1D(ch2, 5, strides = 2, padding = 'valid', name='convX')(X) #switched same to valid

    X = Dropout(droprate)(X)

    X = BatchNormalization(epsilon=0.00001, name='bnX')(X)

    X = LeakyReLU(alpha=0.1, name = 'reluX')(X)

    

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED

    X = Flatten()(X)

    

    # OUTPUT LAYER - Fully Connected

    # Propagate X through a Dense layer with sigmoid activation to get back a batch of 128-dimensional vectors.

    X = Dense(ch2, activation='sigmoid', name='fc_1')(X)

    # Then relu output

    #X = Dense(2, activation='relu', name='fc_2')(X)

    X = Dense(2, name='fc_2')(X) #Remove relu activation since my trick for log 0 makes 0's neg!

    # Add dropout with a probability of 0.1

    #X = Dropout(0.1)(X)

    # Propagate X through a Dense layer with softmax activation to get back a batch of 128-dimensional vectors.

    #X = Dense(ch2, name='fc_2')(X)

    #Optional: Add another Dropout + dense

    

    #########################################################

    #3. Create model instance with the correct "inputs" and "outputs"

    #########################################################

    

    # This step creates your Keras model instance, which will be used to train/test the model.

    model = Model(inputs = X_inputs, outputs = X, name='Jess1DInceptionNet')

    

    #Same as return Model(input, x)

    return model
#Define RMS Log Error
# The data is in csv files

# Read in data to a pandas dataframe



#At home:

#df_train = pd.read_csv('data/train.csv') #NOTE: later, use more data!

#df_test = pd.read_csv('data/test.csv')

#df_submission = pd.read_csv('data/submission.csv')



#On kernel:

df_train = gpd.read_file("/kaggle/input/covid19-global-forecasting-week-2/train.csv") #NOTE: later, use more data!

df_test = gpd.read_file("/kaggle/input/covid19-global-forecasting-week-2/test.csv")

df_submission = gpd.read_file("/kaggle/input/covid19-global-forecasting-week-2/submission.csv")
#Look at train data

df_train
#Look at Canada

(df_train[df_train["Country_Region"]=='Canada'])
#Look at test data

len(df_train[df_train["Country_Region"]=='Afghanistan']) #how many days in this one eg
dates_train = df_train.Date.unique()

print("Train dates: ")

print("From " + str(min(dates_train)) + " to " + str(max(dates_train)))

print("Number of days: " + str(len(dates_train)))



dates_test = df_test.Date.unique()

print("Test dates: ")

print("From " + str(min(dates_test)) + " to " + str(max(dates_test)))

print("Number of days: " + str(len(dates_test)))





#dates_submission = df_submission.Date.unique()

#print(dates_submission)

print("Submission")

print("Number of entries: " + str(len(df_submission)))

print("I think this should match the number of entries in test.csv: " + str(len(df_test)))



df_submission
#Let's make a new dataframe with just one row per country

countries = df_train.Country_Region.unique()

#print(countries)



#Now get all the unique places

places = []

for country in countries:

    #n_entries.append(len(df_train[df_train["Country_Region"]==country]))

    #Check and see if there are entries in Province_State or not

    

    #Stupid Denmark has both nans and states. Fix this crap.

    if len(df_train[df_train['Province_State'].notnull() &

                    (df_train["Country_Region"]==country)]) == 0:

        #This means it's just the whole country with no provinces or states

        places.append([country,np.nan])

        if len(df_train[df_train["Country_Region"]==country]) != len(dates_train):

            print("Weirdo country: " + country)

            print("Days: " + str(len(df_train[df_train["Country_Region"]==country])))

    else:

        #There means there are are provinces or states

        #Get a list of them

        provstates = df_train[df_train["Country_Region"]==country].Province_State.unique()

        for ps in provstates:

            if type(ps)!=str:

                #print('Its nan')

                places.append([country,np.nan])

            else:

                #print("It's not nan")

                places.append([country,ps])

            df_region = df_train[(df_train['Province_State']==ps)& (df_train['Country_Region'] == country)]

print("Total number of countries: " + str(len(countries)))

print("Total number of places: " + str(len(places)))

print("Total number of entries = ndays*places: " + str(len(dates_train)*len(places)))

print("Expected number of entries: " + str(len(df_train)))
#Start with a function version

def generate_data_fn(m, n_days, places, df_daydata, df_constdata, A_noise):

    '''Function version of data generator for making X and Y

    INPUTS

    m: number of training examples

    n_days: number of consecutive days to use = number of input elements

    places: list of all unique places to generate predictions for

    df_daydata: dataframe containing daily elements for all places

    df_constdata: dataframe containing constant elements for all places

    A_noise: amplitude of noise to add to X cases and fatalities

    '''

    IDs = [] # Id

    X = [] # Data examples

    Y = [] # Labels

    

    for i in range(m):

        # 1. Randomly select which place to use

        country,ps = random.choice(places)

        #rint(country)

        #print(ps)

        #Now see if a province or state is given

        #type(var) == str

        if type(ps)!=str:

            #print('Its nan')

            df_country = df_train[(df_train['Province_State'].isnull()) &

                                  (df_train["Country_Region"]==country)]

        else:

            #print('not nan')

            df_country = df_train[(df_train['Province_State']==ps) &

                        (df_train["Country_Region"]==country)]

        Data = df_country[['Id','ConfirmedCases','Fatalities']].to_numpy()

        Data = Data.astype(np.float) #Convert these to float since apparently they're strings '0.0'

        

        # 2. Randomly select n_days consecutive values, making sure they all fit

        day0range = len(Data) - n_days

        day0 = np.random.randint(day0range)

        ids = Data[day0:(day0+n_days),0]

        #x = Data[day0:(day0+n_days-1),1:]

        #y = Data[(day0+n_days),1:]

        xy = Data[day0:(day0+n_days),1:]

        

        # 3. Add a little noise to augment the x and y data, if desired

        # Uniform random noise about zero

        #x_noise = 2*np.random.rand(*x.shape)-1

        #y_noise = 2*np.random.rand()-1

        xy_noise = 2*np.random.rand(*xy.shape)-1

        #x = x*(1+A_noise*x_noise)

        #y = y*(1+A_noise*y_noise)

        #print("xy type is: " + str(type(xy)))

        #print("xy_noise type is: " + str(type(xy_noise)))

        #print("A_noise type is: " + str(type(A_noise)))

        #print("xy is: " + str(xy))

        #print("xy_noise is: " + str(xy_noise))

        #print("A_noise is: " + str(A_noise))

        

        xy = xy*(1+A_noise*xy_noise) #THIS GIVES AN ERROR IN KERNEL VERSION BUT NOT HOME VERSION

        #TypeError: can't multiply sequence by non-int of type 'float'

        

        #Make sure it's still monotonically increasing though since it's cumulative!

        #Use np.maximum.accumulate along the column axis

        xy = np.maximum.accumulate(xy,axis=0)

        #Then make them all back into integer numbers? Meh don't bother since logging

        

        # 4. Scale x and y. I think log is a good idea. Lets assume normalize by log(10^6)=6

        #Should give us a good number between 0 and 1 (unless things really get out of hand)

        #x = np.log10(x+0.5)/6 #The 0.5 prevents log(0) = -inf

        #y = np.log10(y+0.5)/6 #The 0.5 prevents log(0) = -inf

        xy = np.log10(xy+0.5)/6 #The 0.5 prevents log(0) = -inf

        

        #Now chop x and y apart

        x = xy[:-1,:]

        y = xy[-1,:]

        

        IDs.append(ids)

        X.append(x)

        Y.append(y)

    #Now make those lists into arrays

    IDs = np.array(IDs)

    Y = np.array(Y)

    X = np.array(X)

    # Force these babies to be float32 instead of uint8 and float64

    IDs = IDs.astype(np.float32, copy=False)

    Y = Y.astype(np.float32, copy=False)

    X = X.astype(np.float32, copy=False)

        

    return IDs, X, Y
#Next, a generator version that can be used endlessly! Mwahahaha

def generate_data(m, n_days, places, df_daydata, df_constdata, A_noise):

    '''Function version of data generator for making X and Y

    INPUTS

    m: number of training examples

    n_days: number of consecutive days to use = number of input elements

    places: list of all unique places to generate predictions for

    df_daydata: dataframe containing daily elements for all places

    df_constdata: dataframe containing constant elements for all places

    A_noise: amplitude of noise to add to X cases and fatalities

    '''

    

    #Now, m-sized make batches of training data

    #note: "while True:" is an infinite loop to keep generator going during model.fit

    while True:

        

        IDs = [] # Id

        X = [] # Data examples

        Y = [] # Labels



        for i in range(m):

            # 1. Randomly select which place to use

            country,ps = random.choice(places)

            #rint(country)

            #print(ps)

            #Now see if a province or state is given

            #type(var) == str

            if type(ps)!=str:

                #print('Its nan')

                df_country = df_train[(df_train['Province_State'].isnull()) &

                                      (df_train["Country_Region"]==country)]

            else:

                #print('not nan')

                df_country = df_train[(df_train['Province_State']==ps) &

                            (df_train["Country_Region"]==country)]

            Data = df_country[['Id','ConfirmedCases','Fatalities']].to_numpy()

            Data = Data.astype(np.float) #Convert these to float since apparently they're strings '0.0'



            # 2. Randomly select n_days consecutive values, making sure they all fit

            day0range = len(Data) - n_days

            day0 = np.random.randint(day0range)

            ids = Data[day0:(day0+n_days),0]

            #x = Data[day0:(day0+n_days-1),1:]

            #y = Data[(day0+n_days),1:]

            xy = Data[day0:(day0+n_days),1:]



            # 3. Add a little noise to augment the x and y data, if desired

            # Uniform random noise about zero

            #x_noise = 2*np.random.rand(*x.shape)-1

            #y_noise = 2*np.random.rand()-1

            xy_noise = 2*np.random.rand(*xy.shape)-1

            #x = x*(1+A_noise*x_noise)

            #y = y*(1+A_noise*y_noise)

            xy = xy*(1+A_noise*xy_noise)

            #Make sure it's still monotonically increasing though since it's cumulative!

            #Use np.maximum.accumulate along the column axis

            xy = np.maximum.accumulate(xy,axis=0)

            #Then make them all back into integer numbers? Meh don't bother since logging



            # 4. Scale x and y. I think log is a good idea. Lets assume normalize by log(10^6)=6

            #Should give us a good number between 0 and 1 (unless things really get out of hand)

            #x = np.log10(x+0.5)/6 #The 0.5 prevents log(0) = -inf

            #y = np.log10(y+0.5)/6 #The 0.5 prevents log(0) = -inf

            xy = np.log10(xy+0.5)/6 #The 0.5 prevents log(0) = -inf



            #Now chop x and y apart

            x = xy[:-1,:]

            y = xy[-1,:]



            IDs.append(ids)

            X.append(x)

            Y.append(y)

        #Now make those lists into arrays

        IDs = np.array(IDs)

        Y = np.array(Y)

        X = np.array(X)

        # Force these babies to be float32 instead of uint8 and float64

        IDs = IDs.astype(np.float32, copy=False)

        Y = Y.astype(np.float32, copy=False)

        X = X.astype(np.float32, copy=False)

        

        #NOTE: I took IDs out of the generator!

        yield X, Y
#TRAIN DATA

m=1000

n_days = 41

A_noise = 0.1

IDs_train, X_train, Y_train = generate_data_fn(m,n_days, places, df_train, df_train, A_noise)

print("IDs shape: " + str(IDs_train.shape))

print("X shape: " + str(X_train.shape))

print("Y shape: " + str(Y_train.shape))
#Check train data to make sure it looks reasonable

k=0

print("The kth place's last values of X are: " + str(X_train[k,-1,:]))

print("The successive values of Y are: " + str(Y_train[k,:]))



# summarize history for loss

plt.rcParams['figure.facecolor'] = 'w'

plt.plot(X_train[k,:,0])# confirmed casescases

plt.plot(X_train[k,:,1])# fatalities

plt.scatter(41,Y_train[k,0])# confirmed casescases

plt.scatter(41,Y_train[k,1])# fatalities

plt.legend(['Xcc', 'Xf', 'Ycc', 'Yf'])#, loc='upper right') #toggle

plt.title('Log Scale cases/fatalities')

plt.ylabel('n')

plt.xlabel('day')

plt.show()
#DUMMY TEST DATA - later import the real test sets

m_test=20

n_days = 41

A_testnoise = 0

IDs_test, X_test, Y_test = generate_data_fn(m_test,n_days, places, df_train, df_train, A_testnoise)

print("IDs shape: " + str(IDs_test.shape))

print("X shape: " + str(X_test.shape))

print("Y shape: " + str(Y_test.shape))
#Check test data to make sure it looks reasonable

k=9

print("The kth place's last values of X are: " + str(X_test[k,-1,:]))

print("The successive values of Y are: " + str(Y_test[k,:]))



# summarize history for loss

plt.rcParams['figure.facecolor'] = 'w'

plt.plot(X_test[k,:,0])# confirmed casescases

plt.plot(X_test[k,:,1])# fatalities

plt.scatter(41,Y_test[k,0])# confirmed casescases

plt.scatter(41,Y_test[k,1])# fatalities

plt.legend(['Xcc', 'Xf', 'Ycc', 'Yf'])#, loc='upper right') #toggle

plt.title('Log Scale cases/fatalities')

plt.ylabel('n')

plt.xlabel('day')

plt.show()
#Step 1: create the model



#del model #if you need to get rid of an old one



############### MAKE NEW MODEL ################

#input shape = shape of the data training/test examples

model = inceptionModel(input_shape = (X_train.shape[1], X_train.shape[2]))

#MODEL OPTIONS: inceptionModel #GRUmodel #LSTMmodel



############### LOAD MODEL ################

#Or load a previously trained, saved version of your model. To do so:

#model = load_model('models/InceptionModelv1_2020-03-31.h5') #or whatever it is
#look at it

model.summary()
# Step 2: compile the model to configure the learning process. 

"""Choose the 3 arguments of compile() wisely."""



#Choose an optimizer

lr = 0.00005

decay = 0.00001

opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, decay=decay)

#opt = SGD(lr=0.01) # More computationally efficient but slower down the gradient than Adam



# compile the model using the accuracy defined above

model.compile(loss='mean_squared_error', optimizer=opt, metrics=["accuracy"])

#or try LSTM...

#LSTMmodel.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

#use loss='binary_crossentropy' when you are using a binary classifier (2 classes)

#use loss='categorical_crossentropy' for multiclass classification (3+ classes)
# Step 3: train the model!!! Choose the number of epochs and the batch size.

""" Batch size is usually a power of 2 ranging from around 8 to 128

    Epochs can be however many you want. Start with a few and see how fast it learns

    Note: if you run fit() again, the model will continue to train with the parameters 

    it has already learned instead of reinitializing them """



#Starting crap

#model.fit(X_train, Y_train, batch_size = 5, epochs=4) 

#LSTMmodel.fit(X_train, Y_train, batch_size = 5, epochs=1) 

#later increase the number of epochs and start with a higher learning rate, eg lr = 0.005



#################################

#FUNCTION VERSION

#################################

#This uses X_train and Y_train data that you already created with the generate_data_fn function

#IDs_train, X_train, Y_train = generate_data_fn(m,n_days, places, df_train, df_train, A_noise)

'''history = model.fit(X_train, Y_train,

          batch_size=5,

          epochs=20,

          #validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y)

         )'''



#################################

#TRAIN MODEL USING DATA GENERATOR

#################################

# Model.fit_generator (from tensorflow.python.keras.engine.training) is deprecated

#Use Model.fit instead now, which supports generators.

batch_size = 40 #this is the batch size for the generator, use this instead of m_train

n_days = 40 #This is the number of days that will be input to the model, 1 less than other fn

A_noise = 0.08

steps_per_epoch = 10

epochs = 200

df_daydata = df_train

df_constdata = df_train #just as a placeholder for now

#generate_data(m, n_days, places, df_daydata, df_constdata, A_noise)

dataGen = generate_data(batch_size, n_days, places, df_daydata, df_constdata, A_noise)



#Could set up a validation set the same way if desired

#validation_data=([x1, x2], y)



history = model.fit(dataGen,

                    steps_per_epoch=steps_per_epoch, #how many batches to use per epoch

                    #validation_data=valGen,

                    #validation_steps=4

                    epochs=epochs)



#Plot the LEARNING CURVE



#matplotlib.use('QT4Agg')

#plt.get_backend()

#with plt.xkcd():

plt.rcParams["figure.figsize"]=8,8

plt.rcParams['figure.facecolor'] = 'w'

plt.rcParams.update({'font.size': 20})

plt.plot(history.history['loss'])

#plt.plot(history.history['val_loss']) #toggle

#opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01) batch_size, steps_per_epoch

plt.title('lr='+str(lr) +

          ', decay='+str(decay)#+ 

          #', batch_sz='+str(batch_size)+

          #', steps/epoch='+str(steps_per_epoch)

         )

plt.ylabel('Loss')

plt.xlabel('Epoch')

#plt.legend(['Training Dataset', 'Validataion Dataset'])#, loc='upper right') #toggle

#plt.annotate('OOH LOOK AT THIS,\nPYTHON HAS A COOL\nXKCD PLOT PACKAGE',

#             xy=(3, 25), arrowprops=dict(arrowstyle='->'), xytext=(12, 12))

plt.show()
# Step 4: test/evaluate the model using the dev/test dataset 

""" Ideally it hasn't seen this data before or been trained on it """



preds = model.evaluate(x = X_test, y = Y_test)#or try LSTM...

#preds = LSTMmodel.evaluate(x = X_test, y = Y_test)

print ("Loss = " + str(preds[0]))

print ("Test Accuracy = " + str(preds[1]))
#Step 5: Use your model to generate predictions on dummy test set first

testpred = model.predict(X_test)

print("Here are three predictions: " + str(testpred))

print("Here are the actual values: " + str(Y_test))
#Check data to make sure it looks reasonable

k=7

print("The kth place's last values of X are: " + str(X_test[k,-1,:]))

print("The successive values of Y are: " + str(Y_test[k,:]))

print("The model's predicted values of Yhat are: " + str(testpred[k,:]))



# summarize history for loss

plt.rcParams['figure.facecolor'] = 'w'

plt.plot(X_test[k,:,0])# confirmed casescases

plt.plot(X_test[k,:,1])# fatalities

plt.scatter(41,Y_test[k,0])# confirmed casescases

plt.scatter(41,Y_test[k,1])# fatalities

plt.scatter(41,testpred[k,0],marker="v")# confirmed casescases

plt.scatter(41,testpred[k,1],marker="v")# fatalities

plt.legend(['Xcc', 'Xf', 'Ycc', 'Yf','Tcc', 'Tf'])#, loc='upper right') #toggle

plt.title('Log Scale cases/fatalities')

plt.ylabel('n')

plt.xlabel('day')

plt.show()
#Create X input for each date in test



#We already have a list of all the places

#print("Total number of places: " + str(len(places)))



#Go through each place, and see how many training examples we need for each

lastday = '2020-03-19'

lastday = datetime.strptime(lastday, "%Y-%m-%d") #string to date



predictions = [] #store predictions in a list for now



for place in places:

    country,ps = place

    print("Now processing: " + str(place))

    #First, get the data for that place

    if type(ps)!=str:

        #print('Its nan')

        df_place_train = df_train[(df_train['Province_State'].isnull()) &

                                  (df_train["Country_Region"]==country)]

        df_place_test = df_test[(df_test['Province_State'].isnull()) &

                                  (df_test["Country_Region"]==country)]

    else:

        #print('not nan')

        #First, get that place's training data for the n_days preceeding the forecast period

        df_place_train = df_train[(df_train['Province_State']==ps) &

                            (df_train["Country_Region"]==country)]

        #Also, get the true data to make the ground label (and get the forecast ID)

        df_place_test = df_test[(df_test['Province_State']==ps) &

                            (df_test["Country_Region"]==country)]

    #print("len of df_place_test: " + str(len(df_place_test)))

    ids_test= df_place_test['ForecastId'].to_numpy()

    

    #Now slice the desired date range to make x0,

    #the first input data sample for that place

    #Specify end and periods, the number of periods (days).

    daterange0 = pd.date_range(end=lastday, periods=n_days) 

    #print(daterange0)

    #Force Date into the right format

    #df['date'] = pd.to_datetime(df['date']) 

    df_data0 = df_place_train[pd.to_datetime(df_place_train['Date']).isin(daterange0)]

    #print("len of df_data0: " + str(len(df_data0)))

    #Now make an array of the columns we care about

    data0 = df_data0[['Id','ConfirmedCases','Fatalities']].to_numpy()

    data0 = data0.astype(np.float) #Convert these to float since apparently they're strings '0.0'

    #Slice it into 2 arrrays

    ids_train = data0[:,0] #just in case we need it later

    x0 = data0[:,1:] #This is the first data sample to run through the network

    # 4. Log scale x and normalize by log(10^6)=6

    #Should give us a good number between 0 and 1 (unless things really get out of hand)

    x0 = np.log10(x0+0.5)/6 #The 0.5 prevents log(0) = -inf

    #It expects some value for m. Put in a 1

    x = np.expand_dims(x0, axis=0) #x will have shape (1, 40, 2) = (m,day,elements)

    #print("First x shape: " + str(x.shape))

    

    for id in ids_test:

        #Now, run that through the model to generate a prediction.

        pred = model.predict(x) #pred will have shape (1,2)

        #Make sure that prediction is greater than or equal to the preceeding value

        if pred[0,0]<x[0,-1,0]:

            pred[0,0] = x[0,-1,0]

        if pred[0,1]<x[0,-1,1]:

            pred[0,1] = x[0,-1,1]

        #And since I trained it to generate log scale stuff, will already be log scaled

        predictions.append([id,pred[0][0], pred[0][1]])

        newrow = np.expand_dims(pred, axis=0) #x will have shape (1, 1, 2) = (m,day,elements)

        #print("Newrow shape: " + str(newrow.shape))

        #Now chop the first day out of our previous value of x and tack newrow on the end

        #np.concatenate((a, b), axis=0)

        x = np.concatenate((x[:,1:,:],newrow), axis=1)

        #print("Next x shape: " + str(x.shape))

        

#Now make those lists into arrays

predictions = np.array(predictions)

# Force these babies to be float32 instead of uint8 and float64

#predictions = predictions.astype(np.float32, copy=False)
print("Predictions shape:" + str(predictions.shape))

print("Expected number of pedictions:" + str(len(df_submission)))

print("First prediction ID:" + str(predictions[0,0]))

print("Last prediction ID:" + str(predictions[-1,0]))

print("First test ID:" + str(df_test.loc[0 ,['ForecastId'] ]))

print("Last test ID:" + str(df_test.loc[len(predictions)-1 ,['ForecastId'] ]))
#Post process predictions back from weirdo log scale stuff

#pred = np.log10(x+0.5)/6

predictions = predictions.astype(np.float) #Convert these to float since apparently they're strings '0.0'

linearpreds = np.copy(predictions) #Copy predictions over and then un-log the values

linearpreds[:,1:] = 10**(6*linearpreds[:,1:]) - 0.5 #This is screwy

#Now round them all off to the nearest integer

linearpreds = np.around(linearpreds, decimals=1) # np.rint(linearpreds)
#Look at the log predictions along with the log inputs to see how the trends look



#First, get all the data for each country from the training set

allTrainData = df_train[['Id','ConfirmedCases','Fatalities']].to_numpy()

allTrainData = allTrainData.astype(np.float) #Convert these to float since apparently they're strings '0.0'

allTrainLogData = np.log10(allTrainData+0.5)/6 #The 0.5 prevents log(0) = -inf

allTrainLogData[:,0] = allTrainData[:,0]



trainperplace = len(dates_train)

predsperplace = len(dates_test)
i=4 #Start at 1



# Plot the log stuff

plt.rcParams['figure.facecolor'] = 'w'

#Remember predictions only overlap the training data by 1 week, 

#so shift them into the negative before preds with 1 week overlap

xtr = np.arange(trainperplace)

xpr = np.arange(predsperplace) + trainperplace - 7

#days to plot

t0 = ((i-1)*trainperplace)

t1 = (i*trainperplace)

p0 = ((i-1)*predsperplace)

p1 = (i*predsperplace)

#Training data

plt.plot(xtr, allTrainLogData[t0:t1,1])# confirmed casescases

plt.plot(xtr, allTrainLogData[t0:t1,2])# fatalities

#Predictions

plt.plot(xpr, predictions[p0:p1,1])# confirmed casescases

plt.plot(xpr,predictions[p0:p1,2])# fatalities

plt.axvline(x=trainperplace - 7,color='y', linestyle='--')

plt.legend(['Actual cases', 'Actual fatalitites', 

            'Predicted cases', 'Predicted fatalities'], loc='upper left') #toggle

plt.title('Log Scale cases/fatalities')

plt.ylabel('n')

plt.xlabel('day')

plt.show()
# Plot the linear stuff

plt.rcParams['figure.facecolor'] = 'w'



#Training data

plt.plot(xtr, allTrainData[t0:t1,1])# confirmed casescases

plt.plot(xtr, allTrainData[t0:t1,2])# fatalities



#Predictions

plt.plot(xpr, linearpreds[p0:p1,1])# confirmed casescases

plt.plot(xpr,linearpreds[p0:p1,2])# fatalities

plt.axvline(x=trainperplace - 7,color='y', linestyle='--')

plt.legend(['Actual cases', 'Actual fatalitites', 

            'Predicted cases', 'Predicted fatalities'], loc='upper left') #toggle

plt.title('Linear Scale cases/fatalities')

plt.ylabel('n')

plt.xlabel('day')

plt.show()
# Generate submission

linearpreds = linearpreds.astype(np.int) #Convert these to integers %10.5f

#np.savetxt("submission.csv", linearpreds, fmt='%d', delimiter=",")

#Whoops need header, ugh just use pandas again and hope for the right format

df = pd.DataFrame({'ForecastId': linearpreds[:, 0], 

                   'ConfirmedCases': linearpreds[:, 1], 

                   'Fatalities': linearpreds[:, 2]})

df.to_csv('submission.csv', index=False)
############## TO SAVE ##############

model.save('InceptionModelv0_' + str(date.today()) + '.h5')  # creates a HDF5 file



############## TO LOAD ##############

#model = load_model('models/InceptionModelv0_29032020_.h5')