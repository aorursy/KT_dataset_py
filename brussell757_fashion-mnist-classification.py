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
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
train_data = pd.read_csv('../input/fashion-mnist_train.csv')
test_data = pd.read_csv('../input/fashion-mnist_test.csv')

train_data.head()
def prepare_data(data):
    images = data.iloc[:, 1:].values
    labels = data.iloc[:, 0].values
    num_classes = len(np.unique(labels))
    
    images = images / 255.
    labels = tf.one_hot(labels, depth = num_classes)
    
    with tf.Session() as sess:
        labels = labels.eval(session = sess)
        
    return (images, labels)
def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, (n_x, None), name = 'X')
    Y = tf.placeholder(tf.float32, (n_y, None), name = 'Y')
    
    return X, Y
def initialize_parameters(layer_dims):
    parameters = {}
    L = len(layer_dims)
    
    for l in range(1, L):
        parameters['W' + str(l)] = tf.get_variable('W' + str(l), (layer_dims[l], layer_dims[l - 1]), initializer = tf.contrib.layers.xavier_initializer(seed = 42))
        parameters['b' + str(l)] = tf.get_variable('b' + str(l), (layer_dims[l], 1), initializer = tf.zeros_initializer())
    return parameters
def forward_propagation(X, parameters):
    L = len(parameters) // 2
    A = X
    
    for l in range(L - 1):
        W = parameters['W' + str(l + 1)]
        b = parameters['b' + str(l + 1)]
        
        Z = tf.matmul(W, A) + b
        A = tf.nn.relu(Z)
        
    W = parameters['W' + str(L)]
    b = parameters['b' + str(L)]
    
    Z = tf.matmul(W, A) + b
    
    return Z
def compute_cost(Z, Y):
    logits = tf.transpose(Z)
    labels = tf.transpose(Y)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
    
    return cost
def model(train, test, layer_dims, learning_rate = 0.01, epochs = 50, batch_size = 32, print_cost = True):
    ops.reset_default_graph()
    tf.set_random_seed(42)
    seed = 42
    
    train_data, train_labels = prepare_data(train)
    (n_x, m) = train_data.T.shape
    n_y = train_labels.T.shape[0]
    
    test_data, test_labels = prepare_data(test)
    
    costs = []
    
    X, Y = create_placeholders(n_x, n_y)
    parameters = initialize_parameters(layer_dims)
    
    Z = forward_propagation(X, parameters)
    cost = compute_cost(Z, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        
        for epoch in range(epochs):
            epoch_cost = 0.
            num_minibatches = int(m / batch_size)
            seed = seed + 1
            
            for i in range(num_minibatches):
                offset = (i * batch_size) % (train_labels.shape[0] - batch_size)
                minibatch_X = train_data[offset:(offset + batch_size), :]
                minibatch_Y = train_labels[offset:(offset + batch_size), :]

                _, minibatch_cost = sess.run([optimizer, cost], feed_dict = {X:minibatch_X.T, Y:minibatch_Y.T})
                epoch_cost += minibatch_cost / num_minibatches
            
            if m % batch_size != 0:
                minibatch_X = train_data[num_minibatches * batch_size:m, :]
                minibatch_Y = train_labels[num_minibatches * batch_size:m, :]
                
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict = {X:minibatch_X.T, Y:minibatch_Y.T})
                epoch_cost += minibatch_cost / num_minibatches
                
            if epoch % 50 == 0 and print_cost == True:
                print('Cost after epoch {}: {}'.format(epoch, epoch_cost))
            
            if print_cost == True:
                costs.append(epoch_cost)
                
        plt.figure(figsize = (15, 5))
        plt.plot(np.squeeze(costs), c = 'b')
        plt.xlim(0, epochs - 1)
        plt.ylabel('cost')
        plt.xlabel('epochs')
        plt.title('Learning Rate: {}'.format(learning_rate))
        plt.show()
        
        parameters = sess.run(parameters)
        print('Parameters have been trained')
        
        predictions = {'classes': tf.argmax(Z, axis = 1),
                       'probabilities': tf.nn.softmax(Z)}
        
        correct_preds = tf.equal(tf.argmax(Z), tf.argmax(Y))
        accuracy = tf.reduce_mean(tf.cast(correct_preds, 'float'))
        
        print('Train Accuracy: ', accuracy.eval({X:train_data.T, Y:train_labels.T}))
        print('Test Accuracy: ', accuracy.eval({X:test_data.T, Y:test_labels.T}))
        
        return parameters, predictions
layer_dims = [train_data.drop('label', axis = 1).shape[1], 64, 8, 10]
learning_rate = 0.001
epochs = 500
batch_size = 64

parameters, predictions = model(train_data, test_data, layer_dims, learning_rate, epochs, batch_size)
