# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
model = load_model('my_ResNet50_model.hdf5')

print("Model loaded")
import os

import numpy as np

import tensorflow as tf

import h5py

import math



def load_dataset():

    train_dataset = h5py.File('datasets/train_signs.h5', "r")

    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features

    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels



    test_dataset = h5py.File('datasets/test_signs.h5', "r")

    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features

    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels



    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))

    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes





def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):

    """

    Creates a list of random minibatches from (X, Y)

    

    Arguments:

    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)

    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)

    mini_batch_size - size of the mini-batches, integer

    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

    

    Returns:

    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)

    """

    

    m = X.shape[0]                  # number of training examples

    mini_batches = []

    np.random.seed(seed)

    

    # Step 1: Shuffle (X, Y)

    permutation = list(np.random.permutation(m))

    shuffled_X = X[permutation,:,:,:]

    shuffled_Y = Y[permutation,:]



    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.

    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning

    for k in range(0, num_complete_minibatches):

        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]

        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]

        mini_batch = (mini_batch_X, mini_batch_Y)

        mini_batches.append(mini_batch)

    

    # Handling the end case (last mini-batch < mini_batch_size)

    if m % mini_batch_size != 0:

        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]

        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]

        mini_batch = (mini_batch_X, mini_batch_Y)

        mini_batches.append(mini_batch)

    

    return mini_batches





def convert_to_one_hot(Y, C):

    Y = np.eye(C)[Y.reshape(-1)].T

    return Y





def forward_propagation_for_predict(X, parameters):

    """

    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

    

    Arguments:

    X -- input dataset placeholder, of shape (input size, number of examples)

    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"

                  the shapes are given in initialize_parameters



    Returns:

    Z3 -- the output of the last LINEAR unit

    """

    

    # Retrieve the parameters from the dictionary "parameters" 

    W1 = parameters['W1']

    b1 = parameters['b1']

    W2 = parameters['W2']

    b2 = parameters['b2']

    W3 = parameters['W3']

    b3 = parameters['b3'] 

                                                           # Numpy Equivalents:

    Z1 = tf.add(tf.matmul(W1, X), b1)                      # Z1 = np.dot(W1, X) + b1

    A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)

    Z2 = tf.add(tf.matmul(W2, A1), b2)                     # Z2 = np.dot(W2, a1) + b2

    A2 = tf.nn.relu(Z2)                                    # A2 = relu(Z2)

    Z3 = tf.add(tf.matmul(W3, A2), b3)                     # Z3 = np.dot(W3,Z2) + b3

    

    return Z3



def predict(X, parameters):

    

    W1 = tf.convert_to_tensor(parameters["W1"])

    b1 = tf.convert_to_tensor(parameters["b1"])

    W2 = tf.convert_to_tensor(parameters["W2"])

    b2 = tf.convert_to_tensor(parameters["b2"])

    W3 = tf.convert_to_tensor(parameters["W3"])

    b3 = tf.convert_to_tensor(parameters["b3"])

    

    params = {"W1": W1,

              "b1": b1,

              "W2": W2,

              "b2": b2,

              "W3": W3,

              "b3": b3}

    

    x = tf.placeholder("float", [12288, 1])

    

    z3 = forward_propagation_for_predict(x, params)

    p = tf.argmax(z3)

    

    sess = tf.Session()

    prediction = sess.run(p, feed_dict = {x: X})

        

    return prediction
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

#from resnets_utils import *

from keras.initializers import glorot_uniform

import scipy.misc

from matplotlib.pyplot import imshow

%matplotlib inline



import keras.backend as K

K.set_image_data_format('channels_last')

K.set_learning_phase(1)
def identity_block(X , f , filters , stage , block):

    #defining name basis

    conv_name_base = 'res' + str(stage) + block + '_branch'

    bn_name_base = 'bn' + str(stage) + block + '_branch'

    

    #Retrieving Filters

    F1, F2, F3 = filters

    

    #saving X now for shortcut

    X_shortcut = X

    

    #First component

    X = Conv2D(filters = F1 , kernel_size = (1,1) , strides = (1,1) , padding = 'valid' , name = conv_name_base + '2a' , kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3 , name = bn_name_base + '2a')(X)

    X = Activation('relu')(X)

    

    #Second component

    X = Conv2D(filters = F2 , kernel_size = (f,f) , strides = (1,1) , padding = 'same' , name = conv_name_base + '2b' , kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3 , name = bn_name_base + '2b')(X)

    X = Activation('relu')(X)

    

    #Third component

    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3 , name = bn_name_base +'2c')(X)

    

    #Add shortcut

    X = Add()([X , X_shortcut])

    X = Activation('relu')(X)

    

    return X
tf.reset_default_graph()



with tf.Session() as sess:

    np.random.seed(1)

    A_prev = tf.placeholder("float" , [3,4,4,6])

    X = np.random.randn(3,4,4,6)

    A = identity_block(A_prev , f = 2, filters = [2,4,6] , stage = 1 , block='a')

    init = tf.global_variables_initializer()

    sess.run(init)

    out = sess.run([A] , feed_dict={A_prev: X , K.learning_phase():0})

    print("out = " + str(out[0][1][1][0]))
def convolutional_block(X , f , filters , stage , block , s=2):

    # defining name basis

    conv_name_base = 'res' + str(stage) + block +  '_branch'

    bn_name_base = 'bn' + str(stage) + block + '_branch'

    

    #Retrieve filters

    F1 , F2 , F3 = filters

    

    #X_shortcut for short_cut

    X_shortcut = X

    

    #First component

    X = Conv2D(F1,(1,1),strides=(s,s),name=conv_name_base+'2a',padding='valid',kernel_initializer=glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis=3,name=bn_name_base+'2a')(X)

    X = Activation('relu')(X)

    

    #Second component

    X = Conv2D(F2,(f,f),strides=(1,1),name=conv_name_base+'2b',padding='same',kernel_initializer=glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis=3,name=bn_name_base+'2b')(X)

    X = Activation('relu')(X)

    

    #Third component

    X = Conv2D(F3,(1,1),strides=(1,1),name=conv_name_base+'2c',padding='valid',kernel_initializer=glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis=3,name=bn_name_base+'2c')(X)

    

    #Calculating shortcut path

    X_shortcut = Conv2D(F3,(1,1),strides=(s,s),name=conv_name_base+'1',padding='valid',kernel_initializer=glorot_uniform(seed=0))(X_shortcut)

    X_shortcut = BatchNormalization(axis=3,name=bn_name_base+'1')(X_shortcut)

    

    #Final Step

    X = Add()([X,X_shortcut])

    X = Activation('relu')(X)

    

    return X

tf.reset_default_graph()



with tf.Session() as sess:

    np.random.seed(1)

    A_prev = tf.placeholder("float",[3,4,4,6])

    X = np.random.randn(3,4,4,6)

    A = convolutional_block(A_prev,f=2,filters=[2,4,6],stage=1,block='a')

    init = tf.global_variables_initializer()

    sess.run(init)

    out = sess.run([A] , feed_dict={A_prev:X , K.learning_phase():0})

    print("out  = "+ str(out[0][1][1][0]))
def ResNet50(input_shape=(227,227,3),classes=38):

    #input tensor with shape input_shape

    X_input = Input(input_shape)

    

    #Zero Padding

    X = ZeroPadding2D((3,3))(X_input)

    

    #stage 1

    X = Conv2D(64,(7,7),strides=(2,2),name='conv1',kernel_initializer=glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis=3,name='bn_conv1')(X)

    X = Activation('relu')(X)

    X = MaxPooling2D((3,3),strides=(2,2))(X)

    

    #stage 2

    X = convolutional_block(X,f=3,filters=[64,64,256],stage=2,block='a',s=1)

    X = identity_block(X,3,[64,64,256],stage=2,block='b')

    X = identity_block(X,3,[64,64,256],stage=2,block='c')

    

    #stage 3

    X = convolutional_block(X,f=3,filters=[128,128,512],stage=3,block='a',s=2)

    X = identity_block(X,3,[128,128,512],stage=3,block='b')

    X = identity_block(X,3,[128,128,512],stage=3,block='c')

    X = identity_block(X,3,[128,128,512],stage=3,block='d')

    

    #stage 4

    X = convolutional_block(X,f=3,filters=[256,256,1024],stage=4,block='a',s=2)

    X = identity_block(X,3,[256,256,1024],stage=4,block='b')

    X = identity_block(X,3,[256,256,1024],stage=4,block='c')

    X = identity_block(X,3,[256,256,1024],stage=4,block='d')

    X = identity_block(X,3,[256,256,1024],stage=4,block='e')

    X = identity_block(X,3,[256,256,1024],stage=4,block='f')

    

    #stage 5

    X = convolutional_block(X,f=3,filters=[512,512,2048],stage=5,block='a',s=2)

    X = identity_block(X,3,[512,512,2048],stage=5,block='b')

    X = identity_block(X,3,[512,512,2048],stage=5,block='c')

    

    #AVGPool

    X = AveragePooling2D()(X)

    

    #Output layer

    X = Flatten()(X)

    X = Dense(classes,activation='softmax',name='fc'+str(classes),kernel_initializer=glorot_uniform(seed=0))(X)

    

    #create model

    model = Model(inputs=X_input,output=X,name='ResNet50')

    

    return model
model = ResNet50(input_shape=(227,227,3) , classes=38)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
with h5py.File('../input/data_train.h5','r') as hf:

#     for key in hf.keys():

#         print(key)

    X_train = hf['X'].value

    Y_train = hf['Y'].value

    list_classes = hf['list_classes'].value

    inverted = hf['inverted'].value

print(X_train.shape)

print(Y_train.shape)

print(list_classes)

print(inverted.shape)

hf.close()
model.load_weights('../input/ResNet50.hdf5')
with h5py.File('../input/data_test.h5','r') as hf:

#     for key in hf.keys():

#         print(key)

    X_test = hf['X'].value

    Y_test = hf['Y'].value

    list_classes = hf['list_classes'].value

    inverted = hf['inverted'].value

print(X_test.shape)

print(Y_test.shape)

print(list_classes)

print(inverted.shape)

hf.close()
loss, acc = model.evaluate(X_train, Y_train)

print("Restored model, accuracy: {:5.2f}%".format(100*acc))
loss, acc = model.evaluate(X_test, Y_test)

print("Tested model, accuracy: {:5.2f}%".format(100*acc))
from keras.utils import plot_model

plot_model(model, to_file='model.png')