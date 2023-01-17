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
import numpy as np

from keras import layers

from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D

from keras.models import Model, load_model

from keras.preprocessing import image

from keras.utils import layer_utils

from keras.utils.data_utils import get_file

from keras.applications.imagenet_utils import preprocess_input

import pydot

from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot

from keras.utils import plot_model

from resnets_utils import *

from keras.initializers import glorot_uniform

import scipy.misc

from matplotlib.pyplot import imshow

%matplotlib inline



import keras.backend as K

K.set_image_data_format('channels_last')

K.set_learning_phase(1)
# GRADED FUNCTION: identity_block



def identity_block(X, f, filters, stage, block):

    """

    Implementation of the identity block as defined in Figure 4

    

    Arguments:

    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)

    f -- integer, specifying the shape of the middle CONV's window for the main path

    filters -- python list of integers, defining the number of filters in the CONV layers of the main path

    stage -- integer, used to name the layers, depending on their position in the network

    block -- string/character, used to name the layers, depending on their position in the network

    

    Returns:

    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)

    """

    

    # defining name basis

    conv_name_base = 'res' + str(stage) + block + '_branch'

    bn_name_base = 'bn' + str(stage) + block + '_branch'

    

    # Retrieve Filters

    F1, F2, F3 = filters

    

    # Save the input value. You'll need this later to add back to the main path. 

    X_shortcut = X

    

    # First component of main path

    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a')(X)

    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)

    X = Activation('relu')(X)

    

    ### START CODE HERE ###

    

    # Second component of main path (≈3 lines)

    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b')(X)

    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)

    X = Activation('relu')(X)



    # Third component of main path (≈2 lines)

    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c')(X)

    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)



    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)

    X = Add()([X, X_shortcut])

    X = Activation('relu')(X)

    

    ### END CODE HERE ###

    

    return X
import tensorflow as tf



tf.reset_default_graph()



with tf.Session() as test:

    np.random.seed(1)

    A_prev = tf.placeholder("float", [3, 4, 4, 6])

    X = np.random.randn(3, 4, 4, 6)

    A = identity_block(A_prev, f = 2, filters = [2, 4, 6], stage = 1, block = 'a')

    test.run(tf.global_variables_initializer())

    out = test.run([A], feed_dict={A_prev: X, K.learning_phase(): 0})

    print("out = " + str(out[0][1][1][0]))
# GRADED FUNCTION: convolutional_block



def convolutional_block(X, f, filters, stage, block, s = 2):

    """

    Implementation of the convolutional block as defined in Figure 4

    

    Arguments:

    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)

    f -- integer, specifying the shape of the middle CONV's window for the main path

    filters -- python list of integers, defining the number of filters in the CONV layers of the main path

    stage -- integer, used to name the layers, depending on their position in the network

    block -- string/character, used to name the layers, depending on their position in the network

    s -- Integer, specifying the stride to be used

    

    Returns:

    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)

    """

    

    # defining name basis

    conv_name_base = 'res' + str(stage) + block + '_branch'

    bn_name_base = 'bn' + str(stage) + block + '_branch'

    

    # Retrieve Filters

    F1, F2, F3 = filters

    

    # Save the input value

    X_shortcut = X





    ##### MAIN PATH #####

    # First component of main path 

    X = Conv2D(F1, (1, 1), strides = (s,s),  padding = 'valid', name = conv_name_base + '2a')(X)

    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)

    X = Activation('relu')(X)

    

    ### START CODE HERE ###



    # Second component of main path (≈3 lines)

    X = Conv2D(F2, (f, f), strides = (1,1), padding = 'same',  name = conv_name_base + '2b')(X)

    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)

    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)

    X = Conv2D(F3, (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c')(X)

    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### (≈2 lines)

    X_shortcut = Conv2D(F3, (1, 1), strides = (s,s), name = conv_name_base + '1')(X_shortcut)

    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)

    X = Add()([X, X_shortcut])

    X =  Activation('relu')(X) 

    

    ### END CODE HERE ###

    

    return X
def ResNet50(input_shape = (64, 64, 3), classes = 6):

    """

    Implementation of the popular ResNet50 the following architecture:

    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3

    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER



    Arguments:

    input_shape -- shape of the images of the dataset

    classes -- integer, number of classes



    Returns:

    model -- a Model() instance in Keras

    """

    

    # Define the input as a tensor with shape input_shape

    X_input = Input(input_shape)



    

    # Zero-Padding

    X = ZeroPadding2D((3, 3))(X_input)

    

    # Stage 1

    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1')(X)

    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)

    X = Activation('relu')(X)

    X = MaxPooling2D((3, 3), strides=(2, 2))(X)



    # Stage 2

    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)

    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')

    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')



    ### START CODE HERE ###



    # Stage 3 (≈4 lines)

    X = convolutional_block(X, f = 3, filters = [128,128,512], stage = 3, block='a', s = 2)

    X = identity_block(X, 3, [128,128,512], stage=3, block='b')

    X = identity_block(X, 3, [128,128,512], stage=3, block='c')

    X = identity_block(X, 3, [128,128,512], stage=3, block='d')



    # Stage 4 (≈6 lines)

    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)

    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')

    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')

    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')

    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')

    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')



    # Stage 5 (≈3 lines) 

    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)

    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')

    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')



    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"

    X = AveragePooling2D()(X)

    

    ### END CODE HERE ###



    # output layer

    X = Flatten()(X)

    X = Dense(classes, activation='softmax', name='fc' + str(classes))(X)

    

    

    # Create model

    model = Model(inputs = X_input, outputs = X, name='ResNet50')



    return model
model = ResNet50(input_shape = (64, 64, 3), classes = 6)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
x_l = np.load("/kaggle/input/sign-language-digits-dataset/X.npy")

y_l = np.load("/kaggle/input/sign-language-digits-dataset/Y.npy")
X = np.concatenate((x_l[204:409], x_l[822:1027]),axis = 0)

z = np.zeros(205)

o = np.ones(205)

Y = np.concatenate((z, o), axis = 0).reshape(X.shape[0],1)

X.shape,Y.shape
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train,Y_test = train_test_split(X,Y, test_size=0.15, random_state = 42)

number_of_train = X_train.shape[0]

number_of_test = X_test.shape[0]
x_train= X_train.reshape(number_of_train,X_train.shape[1]*X_train.shape[2])

x_test = X_test.reshape(number_of_test,X_test.shape[1]*X_test.shape[2])

x_train.shape,x_test.shape

def per_image_normalization(X, constant=10.0, copy=True):

    if copy:

        X_res = X.copy()

    else:

        X_res = X



    means = np.mean(X, axis=1)

    variances = np.var(X, axis=1) + constant

    X_res = (X_res.T - means).T

    X_res = (X_res.T / np.sqrt(variances)).T

    return X_res
x_train = per_image_normalization(x_train)

x_test = per_image_normalization(x_test)
x_train=x_train.reshape(x_train.shape[0],64,64,3)

x_test=x_test.reshape(x_test.shape[0],64,64,3)


print ("number of training examples = " + str(X_train.shape[0]))

print ("number of test examples = " + str(X_test.shape[0]))

print ("X_train shape: " + str(X_train.shape))

print ("Y_train shape: " + str(Y_train.shape))

print ("X_test shape: " + str(X_test.shape))

print ("Y_test shape: " + str(Y_test.shape))
model.fit(X_train, Y_train, epochs = 2, batch_size = 32)