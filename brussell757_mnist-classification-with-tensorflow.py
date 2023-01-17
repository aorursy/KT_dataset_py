# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import matplotlib.pyplot as plt

import math

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_dataset = pd.read_csv('../input/train.csv')

test_dataset = pd.read_csv('../input/test.csv')
X_train = np.array(train_dataset.drop(['label'], axis = 1)).reshape(-1,28,28,1)

X_train = X_train / 255.



Y_train = np.array(train_dataset['label'])

Y_train = tf.one_hot(Y_train, len(np.unique(Y_train)))

with tf.Session() as sess:

    Y_train = Y_train.eval()

    

X_test = np.array(test_dataset).reshape(-1,28,28,1)

X_test = X_test / 255.
from sklearn.model_selection import train_test_split



X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 0)
indexes = [0,1,2,3,4]



plt.figure(figsize = (25, 3))



i = 1

for index in indexes:

    plt.subplot(1,5, i)

    plt.imshow(np.squeeze(X_train[index]), cmap = 'binary')

    plt.title('y = {}'.format(np.argmax(Y_train[index])))

    plt.xlabel('train image: index {}'.format(index))

    i += 1

    

plt.show()
def create_placeholders(n_h, n_w, n_c, n_y):

    X = tf.placeholder(dtype = 'float', shape = (None, n_h, n_w, n_c))

    Y = tf.placeholder(dtype = 'float', shape = (None, n_y))

    

    return X, Y
def initialize_parameters(layer_dims):

    tf.set_random_seed(1)

    

    parameters = {}

    i = 1

    for l in layer_dims:

        parameters['W' + str(i)] = tf.get_variable('W' + str(i), shape = (l), initializer = tf.contrib.layers.xavier_initializer(seed = 0))

        i += 1

        

    return parameters
tf.reset_default_graph()



layer_dims = [[4,4,3,8],[2,2,8,16]]

initialize_parameters(layer_dims)
def forward_propagation(X, parameters, strides, padding, ksize, num_outputs):

    L = len(parameters)

    A = X

    

    s = 0

    k = 0

    for l in range(L):

        stride = strides[s]

        Z = tf.nn.conv2d(A, parameters['W' + str(l+1)], strides = [stride[0], stride[1], stride[2], stride[3]], padding = padding)

        A = tf.nn.relu(Z)

        

        stride = strides[s + 1]

        P = tf.nn.max_pool(A, ksize = ksize[k], strides = [stride[0], stride[1], stride[2], stride[3]], padding = padding)

        s += 2

        k += 1

        

    P = tf.contrib.layers.flatten(P)

    Z = tf.contrib.layers.fully_connected(P, num_outputs, activation_fn = None)

    

    return Z
strides = [[1,1,1,1], [1,8,8,1], [1,1,1,1], [1,4,4,1]]

ksize = [[1,8,8,1], [1,4,4,1]]

padding = 'SAME'

num_outputs = int(Y_train.shape[-1])



dims = [[4, 4, 3, 8], [2, 2, 8, 16]]

tf.reset_default_graph()

with tf.Session() as sess:

    np.random.seed(1)

    x,y = create_placeholders(64,64,3,10)

    params = initialize_parameters(dims)

    Z = forward_propagation(x, params, strides, padding, ksize, num_outputs)

    init = tf.global_variables_initializer()

    sess.run(init)

    a = sess.run(Z, {x:np.random.randn(2,64,64,3), y:np.random.randn(2,10)})

    print('Z = ' + str(a))
def compute_cost(Z, Y):

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z, labels = Y))

    return cost
def get_minibatches(X, Y, batch_size, seed = 0):

    m = X.shape[0]

    minibatches = []

    np.random.seed(seed)

    

    permutation = list(np.random.permutation(m))

    shuffled_X = X[permutation,:,:,:]

    shuffled_Y = Y[permutation,:]

    

    num_complete_minibatches = math.floor(m/batch_size)

    for k in range(0, num_complete_minibatches):

        minibatch_X = shuffled_X[k * batch_size : k * batch_size + batch_size, :, :, :]

        minibatch_Y = shuffled_Y[k * batch_size : k * batch_size + batch_size, :]

        minibatch = (minibatch_X, minibatch_Y)

        minibatches.append(minibatch)

        

    if m % batch_size != 0:

        minibatch_X = shuffled_X[num_complete_minibatches * batch_size : m, :, :, :]

        minibatch_Y = shuffled_Y[num_complete_minibatches * batch_size : m, :]

        minibatch = (minibatch_X, minibatch_Y)

        minibatches.append(minibatch)

        

    return minibatches
def model(X_train, Y_train, X_test, X_val, Y_val, layer_dims, strides, padding, ksize, num_outputs, learning_rate = 0.001, num_epochs = 100, batch_size = 32, print_cost = True):

    tf.reset_default_graph()

    tf.set_random_seed(1)

    seed = 3

    

    (m, n_H0, n_W0, n_C0) = X_train.shape

    n_y = Y_train.shape[1]

    costs = []

    

    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)

    parameters = initialize_parameters(layer_dims)

    Z = forward_propagation(X, parameters, strides, padding, ksize, num_outputs)

    cost = compute_cost(Z, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    

    init = tf.global_variables_initializer()

    

    with tf.Session() as sess:

        sess.run(init)

        

        for epoch in range(num_epochs):

            epoch_cost = 0.

            num_minibatches = m // batch_size

            seed = seed + 1

            minibatches = get_minibatches(X_train, Y_train, batch_size, seed)

            

            for batch in minibatches:

                (batch_X, batch_Y) = batch

                _, temp_cost = sess.run([optimizer, cost], feed_dict = {X:batch_X, Y:batch_Y})

                epoch_cost += temp_cost / num_minibatches

                

            if print_cost == True and epoch % 10 == 0:

                print('Cost after epoch %i: %f' % (epoch, epoch_cost))

            

            if print_cost == True:

                costs.append(epoch_cost)

                

        plt.plot(np.squeeze(costs))

        plt.ylabel('cost')

        plt.xlabel('epochs')

        plt.title('Learning Rate = ' + str(learning_rate))

        plt.show()

        

        preds = tf.argmax(Z, axis = 1)

        correct_preds = tf.equal(preds, tf.argmax(Y, axis = 1))

        

        accuracy = tf.reduce_mean(tf.cast(correct_preds, 'float'))

        print(accuracy)

        train_accuracy = accuracy.eval({X:X_train, Y:Y_train})

        val_accuracy = accuracy.eval({X:X_val, Y:Y_val})

        

        print('Training Accuracy:', train_accuracy)

        print('Validation Accuracy:', val_accuracy)

        

        test_predictions = sess.run(tf.argmax(Z, 1), feed_dict = {X:X_test})

        

        return train_accuracy, val_accuracy, test_predictions, parameters
layer_dims = [[4, 4, 1, 8], [2, 2, 8, 16]]

strides = [[1,1,1,1], [1,8,8,1], [1,1,1,1], [1,4,4,1]]

ksize = [[1,8,8,1], [1,4,4,1], [1,2,2,1]]

padding = 'SAME'

num_outputs = int(Y_train.shape[-1])

learning_rate = 0.005

epochs = 50

batch_size = 64

    

_,_,test_predictions,parameters = model(X_train, Y_train, X_test, X_val, Y_val, layer_dims, strides, padding, ksize, num_outputs, learning_rate, epochs, batch_size)
indexes = list(np.arange(0,25,1))



plt.figure(figsize = (25,27))



i = 1

for index in indexes:

    plt.subplot(5,5, i)

    plt.imshow(np.squeeze(X_test[index]), cmap = 'binary')

    plt.title('y = {}'.format(test_predictions[index]))

    plt.xlabel('train image: index {}'.format(index))

    i += 1

    

plt.show()