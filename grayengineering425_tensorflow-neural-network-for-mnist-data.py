# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import pandas as pd
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
def load_dataset():
    train_set = pd.read_csv('../input/train.csv')
    test_set  = pd.read_csv('../input/test.csv' )
    
    train_set_x      = train_set.drop(columns=['label'])
    train_set_x_orig = np.array(train_set_x)
    train_set_y_orig = np.array(train_set['label'][:])
    
    test_set_x_orig = np.array(test_set)
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig
X_train_orig, Y_train_orig, X_test_orig = load_dataset()
print(X_train_orig.shape)
print(Y_train_orig.shape)
print(X_test_orig.shape)
X_train, X_dev, Y_train, Y_dev = train_test_split(X_train_orig, Y_train_orig, test_size = 0.1, random_state=42)
def convert_to_one_hot(labels, C):
    C = tf.constant(C, name = 'C')    
    one_hot_matrix = tf.one_hot(labels, C, axis=0)

    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()
    
    return one_hot
Y_train = convert_to_one_hot(Y_train, 10)

X_train = X_train.reshape(X_train.shape[0], -1).T
X_train = X_train/255

Y_dev = convert_to_one_hot(Y_dev, 10)

X_dev = X_dev.reshape(X_dev.shape[0], -1).T
X_dev = X_dev/255

X_test = X_test_orig.reshape(X_test_orig.shape[0], -1).T
X_test = X_test/255
print(X_train.shape)
print(Y_train.shape)
print(X_dev.shape)
print(Y_dev.shape)
print(X_test .shape)
def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, shape=(n_x, None), name='X')
    Y = tf.placeholder(tf.float32, shape=(n_y, None), name='Y')

    return X, Y
def initialize_parameters():
    W1 = tf.get_variable('W1', [300, 784], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable('b1', [300,   1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable('W2', [200,  300], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable('b2', [200,   1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable('W3', [100,  200], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable('b3', [100,   1], initializer = tf.zeros_initializer())
    W4 = tf.get_variable('W4', [10,  100], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b4 = tf.get_variable('b4', [10,   1], initializer = tf.zeros_initializer())
    
    parameters = { 'W1' : W1,
                   'b1' : b1,
                   'W2' : W2,
                   'b2' : b2,
                   'W3' : W3,
                   'b3' : b3,
                   'W4' : W4,
                   'b4' : b4 }
    
    return parameters
def forward_propagation(X, parameters, keep_prob):
    
    W1 = parameters['W1'] 
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']
    
    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    drop_out1 = tf.nn.dropout(A1, keep_prob)
    Z2 = tf.add(tf.matmul(W2, drop_out1), b2)
    A2 = tf.nn.relu(Z2)
    #drop_out2 = tf.nn.dropout(A2, keep_prob)
    Z3 = tf.add(tf.matmul(W3, A2), b3)
    A3 = tf.nn.relu(Z3)
    Z4 = tf.add(tf.matmul(W4, A3), b4)
    
    return Z4
def compute_cost(parameters, Z3, Y):
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    W4 = parameters['W4']
    
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels = labels) +
        0.001*tf.nn.l2_loss(W1) +
        0.001*tf.nn.l2_loss(W2) +
        0.001*tf.nn.l2_loss(W3) +
        0.001*tf.nn.l2_loss(W4))
    
    return cost
def random_mini_batches(X, Y, mini_batch_size = 32, seed = 0):
    m = X.shape[1]
    mini_batches = []
    np.random.seed(seed)
    
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches
def model(X_train, Y_train, learning_rate = 0.0001, num_epochs = 1500, minibatch_size = 32, print_cost = True, k_prob=0.5):
    ops.reset_default_graph()
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []
    
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    
    X, Y = create_placeholders(n_x, n_y) 
    parameters = initialize_parameters()
    Z4 = forward_propagation(X, parameters, keep_prob)
    cost = compute_cost(parameters, Z4, Y) 
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    #optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cost)  
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            epoch_cost = 0
            num_minibatches = int(m / minibatch_size)
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)
            
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                
                _, mini_batch_cost = sess.run([optimizer, cost], feed_dict = { X : minibatch_X, Y : minibatch_Y, keep_prob : k_prob })
                epoch_cost += mini_batch_cost / num_minibatches
                
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                plt.plot(np.squeeze(costs))
        
        #plt.ylabel('cost')
        #plt.xlabel('iterations (per tens)')
        #plt.title("Learning rate =" + str(learning_rate))
        #plt.show()

        parameters = sess.run(parameters)
        
        correct_prediction = tf.equal(tf.argmax(Z4), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        train_acc = accuracy.eval({X: X_train, Y: Y_train, keep_prob : 1.0})
        test_acc  = accuracy.eval({X: X_dev  , Y: Y_dev  , keep_prob : 1.0})
        print("Keep_Prob: ", k_prob)
        print ("Train Accuracy:", train_acc)
        print ("Test Accuracy:", test_acc)
        print ("-------------------------------------------")
        
        return parameters, train_acc, test_acc
def forward_propagation_for_predict(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']
                                        
    Z1 = tf.add(tf.matmul(W1, X), b1)                      
    A1 = tf.nn.relu(Z1)                                    
    Z2 = tf.add(tf.matmul(W2, A1), b2)                     
    A2 = tf.nn.relu(Z2)                                    
    Z3 = tf.add(tf.matmul(W3, A2), b3)
    A3 = tf.nn.relu(Z3)                                    
    Z4 = tf.add(tf.matmul(W4, A3), b4)    
    
    return Z4
def predict(X, parameters):
    
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
    W4 = tf.convert_to_tensor(parameters["W4"])
    b4 = tf.convert_to_tensor(parameters["b4"])
    
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3,
              "W4": W4,
              "b4": b4 }
    
    x = tf.placeholder("float", [784, 1])
    
    z4 = forward_propagation_for_predict(x, params)
    p = tf.argmax(z4)
    
    sess = tf.Session()
    prediction = sess.run(p, feed_dict = {x: X})
        
    return prediction
#parameters, train_acc, test_acc = model(X_train, Y_train, num_epochs = 100, k_prob=0.4, print_cost=True)
#train_set_verify = pd.read_csv('../input/train.csv' )
#test_set_submit  = pd.read_csv('../input/test.csv'  )
#index = 18
#
#train_set_test = train_set_verify.iloc[index]
#train_set_test = train_set_test.drop(['label'], axis=0)
#np_train_set_test = np.array(train_set_test)
#np_train_set_test = np_train_set_test.reshape(np_train_set_test.shape[0], -1)
#np_train_set_test = np_train_set_test/255
#np_train_set_test.shape
#my_image_prediction = predict(np_train_set_test, parameters)
#print("My algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))
#print("Actual Value is: y = " + str(train_set_verify.iloc[index][0]))
#columns = ['ImageId', 'Label']
#predictions = pd.DataFrame(columns = columns)
#W1 = tf.convert_to_tensor(parameters["W1"])
#b1 = tf.convert_to_tensor(parameters["b1"])
#W2 = tf.convert_to_tensor(parameters["W2"])
#b2 = tf.convert_to_tensor(parameters["b2"])
#W3 = tf.convert_to_tensor(parameters["W3"])
#b3 = tf.convert_to_tensor(parameters["b3"])
#W4 = tf.convert_to_tensor(parameters["W4"])
#b4 = tf.convert_to_tensor(parameters["b4"])
#
#x = tf.placeholder("float", [784, 1])
#Z1 = tf.add(tf.matmul(W1, x), b1) 
#A1 = tf.nn.relu(Z1)               
#Z2 = tf.add(tf.matmul(W2, A1), b2)
#A2 = tf.nn.relu(Z2)               
#Z3 = tf.add(tf.matmul(W3, A2), b3)
#A3 = tf.nn.relu(Z3)               
#Z4 = tf.add(tf.matmul(W4, A3), b4)
#
#p = tf.argmax(Z4)
#sess = tf.Session()
#
#for index in range(X_test.shape[1]):
#    cur_test = X_test[:, index].reshape(X_test.shape[0], -1)
#    prediction = sess.run(p, feed_dict = { x : cur_test })
#    
#    if index % 4000 == 0:
#        print(index)
#    #print("My algorithm predicts: y = " + str(np.squeeze(prediction))) 
#
#    predictions = predictions.append({'ImageId' : index+1, 'Label' : str(np.squeeze(prediction))}, ignore_index=True)
#    
#sess.close()
#predictions = predictions.set_index('ImageId')
#predictions.to_csv('MNIST_SUBMISSION.csv')
def create_placeholders_cnn(n_h, n_w, n_c, n_y):
    X = tf.placeholder(shape=[None, n_h, n_w, n_c], dtype='float', name='X_cnn')
    Y = tf.placeholder(shape=[None, n_y], dtype='float', name='Y_cnn')
    
    return X, Y
def initialize_parameters_cnn():
    W1 = tf.get_variable('W1', [3, 3, 1,   6]     , initializer=tf.contrib.layers.xavier_initializer())
    W2 = tf.get_variable('W2', [3, 3, 6,  16]     , initializer=tf.contrib.layers.xavier_initializer())
    
    parameters = {'W1' : W1,
                  'W2' : W2 }
    
    return parameters
def forward_propagation_cnn(X, parameters, keep_prob):
    W1 = parameters['W1']
    W2 = parameters['W2']
    
    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding="SAME")
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize = [1,2,2,1], strides = [1,2,2,1], padding="SAME")

    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding="VALID")
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize = [1,2,2,1], strides = [1,2,2,1], padding="VALID")

    Z3 = tf.contrib.layers.flatten(P2)                                               #should be size 400....
    dense1 = tf.layers.dense(inputs=Z3, units=250, activation=tf.nn.relu)
    dropout1 = tf.layers.dropout(inputs=dense1, rate=0.4, training=keep_prob)
    
    dense2 = tf.layers.dense(inputs=dropout1, units=120, activation=tf.nn.relu)
    dropout2 = tf.layers.dropout(inputs=dense2, rate=0.35, training=keep_prob)

    Z4 = tf.contrib.layers.fully_connected(dropout2, 10, activation_fn=None)
    
    #print('X Shape: ' , X.shape)
    #print('Z1 Shape: ', Z1.shape)
    #print('A1 Shape: ', A1.shape)
    #print('P1 Shape: ', P1.shape)
    #print('Z2 Shape: ', Z2.shape)
    #print('A2 Shape: ', A2.shape)
    #print('P2 Shape: ', P2.shape)
    #print('Z3 Shape: ', Z3.shape)
    #print('DO1 Shape: ', dropout1.shape)
    #print('DO2 Shape: ', dropout2.shape)
    #print('Z4 Shape: ', Z4.shape)
    
    return Z4
def compute_cost_cnn(parameters, Z5, Y):
    W1 = parameters['W1']
    W2 = parameters['W2']

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z5, labels = Y)) + 0.001*tf.nn.l2_loss(W1) + 0.001*tf.nn.l2_loss(W2)
    
    return cost
def random_mini_batches_cnn(X, Y, mini_batch_size = 32, seed = 0):
    m = X.shape[0]
    mini_batches = []
    np.random.seed(seed)

    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches
def model_cnn(X_train, Y_train, X_test, learning_rate = 0.0001, num_epochs = 1500, minibatch_size = 32, print_cost = True):
    ops.reset_default_graph()
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []
    
    kp = tf.placeholder(dtype='bool')
    
    X, Y = create_placeholders_cnn(n_H0, n_W0, n_C0, n_y) 
    parameters = initialize_parameters_cnn()
    Z5 = forward_propagation_cnn(X, parameters, keep_prob=kp)
    cost = compute_cost_cnn(parameters, Z5, Y) 
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            epoch_cost = 0
            num_minibatches = int(m / minibatch_size)
            minibatches = random_mini_batches_cnn(X_train, Y_train, minibatch_size)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                
                _, mini_batch_cost = sess.run([optimizer, cost], feed_dict = { X : minibatch_X, Y : minibatch_Y, kp : True})
                epoch_cost += mini_batch_cost / num_minibatches
                
            if print_cost == True and epoch % 10 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                plt.plot(np.squeeze(costs))
    
        correct_prediction = tf.equal(tf.argmax(Z5,1), tf.argmax(Y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        train_acc = accuracy.eval({X: X_train, Y: Y_train, kp : False})
        test_acc  = accuracy.eval({X: X_dev  , Y: Y_dev  , kp : False})
        print ("Train Accuracy:", train_acc)
        print ("Test Accuracy:", test_acc)
        print ("-------------------------------------------")
        
        test_pred = sess.run(Z5, feed_dict={X : X_test, kp : False})
        
        return test_pred
X_train_orig, Y_train_orig, X_test_orig = load_dataset()
X_train, X_dev, Y_train, Y_dev = train_test_split(X_train_orig, Y_train_orig, test_size = 0.1, random_state=42)
Y_train = convert_to_one_hot(Y_train, 10).T
X_train = X_train/255

Y_dev = convert_to_one_hot(Y_dev, 10).T
X_dev = X_dev/255

X_test = X_test_orig/255
width = height = np.ceil(np.sqrt(X_train.shape[1])).astype(np.uint8)

X_train = np.reshape(np.array(X_train) , (-1, width, height, 1))
X_dev   = np.reshape(np.array(X_dev) , (-1, width, height, 1))
X_test  = np.reshape(np.array(X_test), (-1, width, height, 1))
print(X_train.shape)
print(Y_train.shape)
print(X_dev.shape)
print(Y_dev.shape)
print(X_test .shape)
test_pred = model_cnn(X_train, Y_train, X_test, num_epochs = 150)
test_pred = np.argmax(test_pred, axis=1)
k = 27000
print("Label Prediction: %i"%test_pred[k])
fig = plt.figure(figsize=(2,2)); plt.axis('off')
plt.imshow(X_test[k,:,:,0]); plt.show()
submission = pd.DataFrame(data={'ImageId':(np.arange(test_pred.shape[0])+1), 'Label':test_pred})
submission.to_csv('MNIST_SUBMISSION.csv', index=False)