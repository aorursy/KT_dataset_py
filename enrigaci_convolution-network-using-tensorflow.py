import pandas as pd

import math

import numpy as np

import h5py

import matplotlib.pyplot as plt

import scipy

from PIL import Image

from scipy import ndimage

import tensorflow as tf

from tensorflow.python.framework import ops

from keras import backend as K

from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate

from keras.models import Model

from keras.layers.normalization import BatchNormalization

from keras.layers.pooling import MaxPooling2D, AveragePooling2D

from keras.layers.core import Lambda, Flatten, Dense
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



# Retrieve the actual numbers stored in the label column

Y_train_orig = np.array(train['label'][:]).reshape((1, train['label'].shape[0]))

# Store the pixels of each image without the label column

X_train = np.array(train.T[1:].T)

X_train = X_train / 255



# Store the pixels of each image without the label column

X_test = np.array(test.T[0:].T)

X_test = X_test / 255



# Transform the input labels to (1,10) arrays with 1 in the place the number is

# i.e if the number is 2 then Y[position_containin_num_2] = [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]

# We will need this later for our convolution

Y_train = np.eye(10)[Y_train_orig[0]]
X_train.shape, Y_train.shape
# Example of a picture

index = 7

plt.imshow(np.array(X_train[index:index+1]).reshape(28,28))

print ("y = " + str(np.squeeze(Y_train[index])))
def conv2d_bn(x,

			  layer=None,

			  cv1_out=None,

			  cv1_filter=(1, 1),

			  cv1_strides=(1, 1),

			  cv2_out=None,

			  cv2_filter=(3, 3),

			  cv2_strides=(1, 1),

			  padding=None):

	num = '' if cv2_out == None else '1'

	tensor = Conv2D(cv1_out, cv1_filter, strides=cv1_strides, data_format='channels_first', name=layer+'_conv'+num)(x)

	tensor = BatchNormalization(axis=1, epsilon=0.00001, name=layer+'_bn'+num)(tensor)

	tensor = Activation('relu')(tensor)

	if padding == None:

		return tensor

	tensor = ZeroPadding2D(padding=padding, data_format='channels_first')(tensor)

	if cv2_out == None:

		return tensor

	tensor = Conv2D(cv2_out, cv2_filter, strides=cv2_strides, data_format='channels_first', name=layer+'_conv'+'2')(tensor)

	tensor = BatchNormalization(axis=1, epsilon=0.00001, name=layer+'_bn'+'2')(tensor)

	tensor = Activation('relu')(tensor)

	return tensor





def inception_block_1a(X):

	"""

	Implementation of an inception block

	"""

	X_3x3 = Conv2D(96, (1, 1), data_format='channels_first', name ='inception_3a_3x3_conv1')(X)

	X_3x3 = BatchNormalization(axis=1, epsilon=0.00001, name = 'inception_3a_3x3_bn1')(X_3x3)

	X_3x3 = Activation('relu')(X_3x3)

	X_3x3 = ZeroPadding2D(padding=(1, 1), data_format='channels_first')(X_3x3)

	X_3x3 = Conv2D(128, (3, 3), data_format='channels_first', name='inception_3a_3x3_conv2')(X_3x3)

	X_3x3 = BatchNormalization(axis=1, epsilon=0.00001, name='inception_3a_3x3_bn2')(X_3x3)

	X_3x3 = Activation('relu')(X_3x3)

	X_5x5 = Conv2D(16, (1, 1), data_format='channels_first', name='inception_3a_5x5_conv1')(X)

	X_5x5 = BatchNormalization(axis=1, epsilon=0.00001, name='inception_3a_5x5_bn1')(X_5x5)

	X_5x5 = Activation('relu')(X_5x5)

	X_5x5 = ZeroPadding2D(padding=(2, 2), data_format='channels_first')(X_5x5)

	X_5x5 = Conv2D(32, (5, 5), data_format='channels_first', name='inception_3a_5x5_conv2')(X_5x5)

	X_5x5 = BatchNormalization(axis=1, epsilon=0.00001, name='inception_3a_5x5_bn2')(X_5x5)

	X_5x5 = Activation('relu')(X_5x5)

	X_pool = MaxPooling2D(pool_size=3, strides=2, data_format='channels_first')(X)

	X_pool = Conv2D(32, (1, 1), data_format='channels_first', name='inception_3a_pool_conv')(X_pool)

	X_pool = BatchNormalization(axis=1, epsilon=0.00001, name='inception_3a_pool_bn')(X_pool)

	X_pool = Activation('relu')(X_pool)

	X_pool = ZeroPadding2D(padding=((1, 2), (48, 49)), data_format='channels_first')(X_pool)

	X_1x1 = Conv2D(64, (1, 1), data_format='channels_first', name='inception_3a_1x1_conv')(X)

	X_1x1 = BatchNormalization(axis=1, epsilon=0.00001, name='inception_3a_1x1_bn')(X_1x1)

	X_1x1 = Activation('relu')(X_1x1)

	# CONCAT

	inception = concatenate([X_3x3, X_5x5, X_pool, X_1x1], axis=1)

	return inception



def inception_block_1b(X):

	X_3x3 = Conv2D(96, (1, 1), data_format='channels_first', name='inception_3b_3x3_conv1')(X)

	X_3x3 = BatchNormalization(axis=1, epsilon=0.00001, name='inception_3b_3x3_bn1')(X_3x3)

	X_3x3 = Activation('relu')(X_3x3)

	X_3x3 = ZeroPadding2D(padding=(1, 1), data_format='channels_first')(X_3x3)

	X_3x3 = Conv2D(128, (3, 3), data_format='channels_first', name='inception_3b_3x3_conv2')(X_3x3)

	X_3x3 = BatchNormalization(axis=1, epsilon=0.00001, name='inception_3b_3x3_bn2')(X_3x3)

	X_3x3 = Activation('relu')(X_3x3)

	X_5x5 = Conv2D(32, (1, 1), data_format='channels_first', name='inception_3b_5x5_conv1')(X)

	X_5x5 = BatchNormalization(axis=1, epsilon=0.00001, name='inception_3b_5x5_bn1')(X_5x5)

	X_5x5 = Activation('relu')(X_5x5)

	X_5x5 = ZeroPadding2D(padding=(2, 2), data_format='channels_first')(X_5x5)

	X_5x5 = Conv2D(64, (5, 5), data_format='channels_first', name='inception_3b_5x5_conv2')(X_5x5)

	X_5x5 = BatchNormalization(axis=1, epsilon=0.00001, name='inception_3b_5x5_bn2')(X_5x5)

	X_5x5 = Activation('relu')(X_5x5)

	X_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3), data_format='channels_first')(X)

	X_pool = Conv2D(64, (1, 1), data_format='channels_first', name='inception_3b_pool_conv')(X_pool)

	X_pool = BatchNormalization(axis=1, epsilon=0.00001, name='inception_3b_pool_bn')(X_pool)

	X_pool = Activation('relu')(X_pool)

	X_pool = ZeroPadding2D(padding=((1, 2), (64, 64)), data_format='channels_first')(X_pool)

	X_1x1 = Conv2D(64, (1, 1), data_format='channels_first', name='inception_3b_1x1_conv')(X)

	X_1x1 = BatchNormalization(axis=1, epsilon=0.00001, name='inception_3b_1x1_bn')(X_1x1)

	X_1x1 = Activation('relu')(X_1x1)

	inception = concatenate([X_3x3, X_5x5, X_pool, X_1x1], axis=1)

	return inception





def inception_block_1c(X):

	X_3x3 = conv2d_bn(X,

						   layer='inception_3c_3x3',

						   cv1_out=128,

						   cv1_filter=(1, 1),

						   cv2_out=256,

						   cv2_filter=(3, 3),

						   cv2_strides=(2, 2),

						   padding=(1, 1))

	X_5x5 = conv2d_bn(X,

						   layer='inception_3c_5x5',

						   cv1_out=32,

						   cv1_filter=(1, 1),

						   cv2_out=64,

						   cv2_filter=(5, 5),

						   cv2_strides=(2, 2),

						   padding=(2, 2))

	X_pool = MaxPooling2D(pool_size=3, strides=2, data_format='channels_first')(X)

	X_pool = ZeroPadding2D(padding=((0, 1), (0, 1)), data_format='channels_first')(X_pool)

	inception = concatenate([X_3x3, X_5x5, X_pool], axis=1)

	return inception



def inception_block_2a(X):

	X_3x3 = conv2d_bn(X,

						   layer='inception_4a_3x3',

						   cv1_out=96,

						   cv1_filter=(1, 1),

						   cv2_out=192,

						   cv2_filter=(3, 3),

						   cv2_strides=(1, 1),

						   padding=(1, 1))

	X_5x5 = conv2d_bn(X,

						   layer='inception_4a_5x5',

						   cv1_out=32,

						   cv1_filter=(1, 1),

						   cv2_out=64,

						   cv2_filter=(5, 5),

						   cv2_strides=(1, 1),

						   padding=(2, 2))

	X_pool = AveragePooling2D(pool_size=(2, 2), strides=(3, 3), data_format='channels_first')(X)

	X_pool = conv2d_bn(X_pool,

						   layer='inception_4a_pool',

						   cv1_out=128,

						   cv1_filter=(1, 1),

						   padding=((1,0), (32, 32)))

	X_1x1 = conv2d_bn(X,

						   layer='inception_4a_1x1',

						   cv1_out=256,

						   cv1_filter=(1, 1))

	inception = concatenate([X_3x3, X_5x5, X_pool, X_1x1], axis=1)

	return inception





def inception_block_2b(X):

	#inception4e

	X_3x3 = conv2d_bn(X,

						   layer='inception_4e_3x3',

						   cv1_out=160,

						   cv1_filter=(1, 1),

						   cv2_out=256,

						   cv2_filter=(3, 3),

						   cv2_strides=(2, 2),

						   padding=(1, 1))

	X_5x5 = conv2d_bn(X,

						   layer='inception_4e_5x5',

						   cv1_out=64,

						   cv1_filter=(1, 1),

						   cv2_out=128,

						   cv2_filter=(5, 5),

						   cv2_strides=(2, 2),

						   padding=(2, 2))

	X_pool = MaxPooling2D(pool_size=2, strides=2, data_format='channels_first')(X)

	X_pool = ZeroPadding2D(padding=((0, 0), (0, 0)), data_format='channels_first')(X_pool)

	inception = concatenate([X_3x3, X_5x5, X_pool], axis=1)

	return inception





def inception_block_3a(X):

	X_3x3 = conv2d_bn(X,

						   layer='inception_5a_3x3',

						   cv1_out=96,

						   cv1_filter=(1, 1),

						   cv2_out=384,

						   cv2_filter=(3, 3),

						   cv2_strides=(1, 1),

						   padding=(1, 1))

	X_pool = AveragePooling2D(pool_size=(1, 1), strides=(3, 3), data_format='channels_first')(X)

	X_pool = conv2d_bn(X_pool,

						   layer='inception_5a_pool',

						   cv1_out=96,

						   cv1_filter=(1, 1),

						   padding=((0, 0), (16,16)))

	X_1x1 = conv2d_bn(X,

						   layer='inception_5a_1x1',

						   cv1_out=256,

						   cv1_filter=(1, 1))

	inception = concatenate([X_3x3, X_pool, X_1x1], axis=1)

	return inception



def inception_block_3b(X):

	X_3x3 = conv2d_bn(X,

						   layer='inception_5b_3x3',

						   cv1_out=96,

						   cv1_filter=(1, 1),

						   cv2_out=384,

						   cv2_filter=(3, 3),

						   cv2_strides=(1, 1),

						   padding=(1, 1))

	X_pool = MaxPooling2D(pool_size=1, strides=2, data_format='channels_first')(X)

	X_pool = conv2d_bn(X_pool,

						   layer='inception_5b_pool',

						   cv1_out=96,

						   cv1_filter=(1, 1))

	X_pool = ZeroPadding2D(padding=((0, 0),(12, 12)), data_format='channels_first')(X_pool)

	X_1x1 = conv2d_bn(X,

						   layer='inception_5b_1x1',

						   cv1_out=256,

						   cv1_filter=(1, 1))

	inception = concatenate([X_3x3, X_pool, X_1x1], axis=1)

	return inception

def numberModel(input_shape):

	"""

	Implementation of the Inception model used for FaceNet

	Arguments:

	input_shape -- shape of the images of the dataset

	Returns:

	model -- a Model() instance in Keras

	"""

	# Define the input as a tensor with shape input_shape

	X_input = Input(input_shape)

	# Zero-Padding

	X = ZeroPadding2D((3, 3))(X_input)

	# First Block

	X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1')(X)

	X = BatchNormalization(axis = 1, name = 'bn1')(X)

	X = Activation('relu')(X)

	# Zero-Padding + MAXPOOL

	X = ZeroPadding2D((1, 1))(X)

	X = MaxPooling2D((3, 3), strides = 2)(X)

	# Second Block

	X = Conv2D(64, (1, 1), strides = (1, 1), name = 'conv2')(X)

	X = BatchNormalization(axis = 1, epsilon=0.00001, name = 'bn2')(X)

	X = Activation('relu')(X)

	# Zero-Padding + MAXPOOL

	X = ZeroPadding2D((1, 1))(X)

	# Second Block

	X = Conv2D(192, (3, 3), strides = (1, 1), name = 'conv3')(X)

	X = BatchNormalization(axis = 1, epsilon=0.00001, name = 'bn3')(X)

	X = Activation('relu')(X)

	# Zero-Padding + MAXPOOL

	X = ZeroPadding2D((1, 1))(X)

	X = MaxPooling2D(pool_size = 3, strides = 2)(X)

	# Inception 1: a/b/c

# 	X = inception_block_1a(X)

# 	X = inception_block_1b(X)

# 	X = inception_block_1c(X)

	# Inception 2: a/b

# 	X = inception_block_2a(X)

# 	X = inception_block_2b(X)

	# # Inception 3: a/b

# 	X = inception_block_3a(X)

# 	X = inception_block_3b(X)

	# Top layer

	X = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), data_format='channels_first')(X)

	X = Flatten()(X)

	X = Dense(10, name='dense_layer')(X)

	# L2 normalization

	X = Lambda(lambda  x: K.l2_normalize(x,axis=1))(X)

	# Create model instance

	model = Model(inputs = X_input, outputs = X, name='FaceRecoModel')

	return model
model = numberModel(input_shape=(1,28,28))
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

model.fit(x = X_train.reshape((42000, 1,28,28)), y = Y_train, epochs = 5, batch_size = 14)
# Prediction phase

preds = model.predict(x = X_test.reshape(28000,1,28,28))

y_preds = np.argmax(preds, axis=1)
print( "The model predicted : " + str(y_preds[10]) + " for the 10th label id")