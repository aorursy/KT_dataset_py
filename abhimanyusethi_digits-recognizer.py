# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from tensorflow.python.client import device_lib

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split # sklearn, train_test_split

import matplotlib.pyplot as plt # matplotlib, plt

from keras.utils.np_utils import to_categorical # categorical - one-hot-encoding



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

# CNN libraries

from sklearn.metrics import confusion_matrix

import itertools

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop,Adam

from keras.callbacks import ReduceLROnPlateau

from keras.preprocessing.image import ImageDataGenerator

import math

import h5py

import scipy

from PIL import Image

from scipy import ndimage

from tensorflow.python.framework import ops

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()



%matplotlib inline

# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
def separate_labels():

    return [train["label"], train.drop(labels = ["label"],axis=1)]
y_train, x_train = separate_labels()
print(x_train.head(5))

print(y_train.head(5))
def show_figure():

    for index in range(0, 9):

        fig = x_train.iloc[index].to_numpy()

        fig = fig.reshape((28,28))

        plt.subplot(340 + (index + 1))

        plt.imshow(fig)

        plt.title(y_train[index])
show_figure()
x_train = x_train.to_numpy()

test = test.to_numpy()
#Normalizing

x_train = x_train/255

test = test/255
#x_train.shape

#y_train.shape

test.shape
x_train = x_train.reshape(-1,28,28,1)

print(x_train.shape)

test = test.reshape(-1,28,28,1)

print(test.shape)

print(y_train.shape)
y_train = tf.keras.utils.to_categorical(y_train,10)
print(y_train.shape)
#Implementing forward prop by hand, for reference, we won't be using this
def conv_single_step(slice,W,b):

    mul = slice * W

    return (np.sum(mul) + b)
def conv_forward(A_prev,W,b,hyperpms):

    # Retrieve dimensions from A_prev's shape (≈1 line)  

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape[0],A_prev.shape[1],A_prev.shape[2],A_prev.shape[3]

    

    # Retrieve dimensions from W's shape (≈1 line)

    (f, f, n_C_prev, n_C) = W.shape[0],W.shape[1],W.shape[2],W.shape[3]

    

    stride = hyperpms['stride']

    

    n_H = int((n_H_prev - f)/stride + 1)

    n_W = int((n_W_prev - f)/stride + 1)

    

    z = np.zeros((m,n_H,n_W,n_C))

    

    for i in range(m):

        a_prev = A_prev[i]

        for h in range(n_H):

            vert_start = h*stride

            vert_end = h*stride + f

            for w in range(n_W):

                hor_start = w*stride

                hor_end = w*stride + f

                for c in range(n_C):

                    a_slice_prev = a_prev[vert_start:vert_end,hor_start:hor_end,:]

                    weights = W[:,:,:,c]

                    biases = b[:,:,:,c]

                    z[i,h,w,c] = conv_single_step(a_slice_prev,weights,biases)

                    

    cache = (A_prev, W, b, hyperpms)

    return z,cache

                
def pool_forward(A_prev,hyperpms,mode='max'):

    

    (m,n_H_prev,n_W_prev,n_C_prev) = A_prev.shape

    

    f = hyperpms["f"]

    stride = hyperpms["stride"]

    

    n_H = int((n_H_prev - f)/stride) + 1

    n_W = int((n_W_prev - f)/stride) + 1

    n_C = n_C_prev

    

    A = np.zeros((m,n_H,n_W,n_C))

    

    for i in range(m):

        for h in range(n_H):

            vert_start = h*stride

            vert_end = h*stride + f

            for w in range(n_W):

                hor_start = w*stride

                hor_end = w*stride + f

                for c in range(n_C):

                    a_slice_prev = A_prev[i,vert_start:vert_end,hor_start:hor_end,c]

                    if mode == "max":

                        A[i, h, w, c] = np.max(a_prev_slice)

                    elif mode == "average":

                        A[i, h, w, c] = np.mean(a_prev_slice)

    cache = (A_prev,hyperpms)

    return A,cache
#Tensorflow implementation from this point onwards
def create_placeholders(n_H0,n_W0,n_C0,n_y):

    X =  tf.compat.v1.placeholder('float',[None, n_H0, n_W0, n_C0])

    Y =  tf.compat.v1.placeholder('float',[None, n_y])

    

    return X,Y
def initialize_parameters():

    W1 = tf.compat.v1.get_variable("W1", [4,4,1,8], initializer=tf.glorot_uniform_initializer())

    W2 = tf.compat.v1.get_variable("W2", [2,2,1,16], initializer=tf.glorot_uniform_initializer())

    

    parameters = {"W1":W1, "W2":W2}

    

    return parameters
#Forward prop using tensorflow this point onwards

#CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

def forward_prop(X,parameters):

    

    W1 = parameters["W1"]

    W2 = parameters["W2"]

    

    z1 = tf.nn.conv2d(X,W1,strides=[1,1,1,1],padding = 'SAME')

    a1 = tf.nn.relu(z1)

    p1 = tf.nn.max_pool(a1, ksize=[1,4,4,1],strides=[1,4,4,1],padding = 'SAME')

    

    z2 = tf.nn.conv2d(p1,W2,strides=[1,1,1,1],padding="SAME")

    a2 = tf.nn.relu(z2)

    p2 = tf.nn.max_pool(a2,ksize=[1,2,2,1],strides=[1,1,1,1],padding="SAME")

    

    f =  tf.keras.layers.Flatten()(p2)

    z3 = tf.keras.layers.Dense(10,activation=None)(f)

    

    return z3
def compute_cost(z3,Y):

    

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = z3,labels = Y))

    

    return cost
def random_mini_batches(X, Y, mini_batch_size = 3000):

    m = X.shape[0]                  # number of training examples

    mini_batches = []

        

    # Step 1: Shuffle (X, Y)

    permutation = list(np.random.permutation(m))

    shuffled_X = X[permutation,:,:,:]

    shuffled_Y = Y[permutation,:]#.reshape((Y.shape[0],m))

    

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.

    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning

    for k in range(0, num_complete_minibatches):

        mini_batch_X = shuffled_X[k * mini_batch_size: (k+1)*mini_batch_size,:,:,:]

        mini_batch_Y = shuffled_Y[k * mini_batch_size: (k+1)*mini_batch_size,:]

        mini_batch = (mini_batch_X, mini_batch_Y)

        mini_batches.append(mini_batch)

    

    # Handling the end case (last mini-batch < mini_batch_size)

    if m % mini_batch_size != 0:

        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size:,:,:,:]

        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size:,:]

        mini_batch = (mini_batch_X, mini_batch_Y)

        mini_batches.append(mini_batch)

    return mini_batches
#Model

#create placeholders

#initialize parameters

#forward propagate

#compute the cost

#create an optimizer

def model(x_train,y_train,test,learning_rate=0.009,no_of_epochs=40,minibatch_size=3000,print_cost=True):

    

    ops.reset_default_graph()   

    (m,n_H0,n_W0,n_C0) = x_train.shape

    n_y = y_train.shape[1]

    X,Y = create_placeholders(n_H0,n_W0,n_C0,n_y)

    

    costs=[]

    

    parameters = initialize_parameters()

    

    z3 = forward_prop(X,parameters)

    

    cost = compute_cost(z3,Y)

    

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    

    init = tf.global_variables_initializer()

    

    with tf.compat.v1.Session() as sess:

        

        sess.run(init)

        

        for epoch in range(no_of_epochs):

            

            minibatch_cost = 0

            num_minibatches = int(m/minibatch_size)

            minibatches = random_mini_batches(x_train, y_train, minibatch_size)

            

            for minibatch in minibatches:

                (minibatch_x,minibatch_y) = minibatch

                _ , temp_cost = sess.run(fetches=[optimizer,cost],feed_dict={X:minibatch_x,Y:minibatch_y}) #this is why we had put 'None' in placholders

                minibatch_cost += temp_cost/num_minibatches

                    

            if print_cost==True:

                print("Cost after epoch %i: %f" % (epoch, minibatch_cost))

            costs.append(minibatch_cost)

        

        plt.plot(np.squeeze(costs))

        plt.ylabel('cost')

        plt.xlabel('iterations (per tens)')

        plt.title("Learning rate =" + str(learning_rate))

        plt.show()



        # Calculate the correct predictions

        predict_op = tf.argmax(z3, 1)

        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        

        # Calculate accuracy on the test set

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print(accuracy)

        train_accuracy = accuracy.eval({X: x_train, Y: y_train})

        print("Train Accuracy:", train_accuracy)

        a = sess.run([tf.argmax(z3,1)], feed_dict={X: test})

        a = np.transpose(a)

        print(a)

        sample_submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

        sample_submission['Label'] = a

        sample_submission.head()

        # saving the file

        sample_submission.to_csv('submission.csv', index=False)

                

        return train_accuracy, parameters
_, parameters = model(x_train, y_train, test)