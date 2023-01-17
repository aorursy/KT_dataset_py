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
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math
from tensorflow.python.framework import ops

training_data = h5py.File('../input/train_catvnoncat.h5','r')
testing_data = h5py.File('../input/test_catvnoncat.h5','r')
train_data = np.array(training_data)
test_data = np.array(testing_data)
print("Training data files ",train_data)
print("Testing data files " ,test_data)
training_data['train_set_x']
X_train_orig = np.array(training_data['train_set_x'][:])
Y_train_orig = np.array(training_data['train_set_y'][:])
X_test_orig = np.array(testing_data['test_set_x'][:])
Y_test_orig = np.array(testing_data['test_set_y'][:])
classes = np.array(testing_data['list_classes'])
X_train = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test = X_test_orig.reshape(X_test_orig.shape[0],-1).T
Y_train = Y_train_orig.reshape((1,Y_train_orig.shape[0]))
Y_test = Y_test_orig.reshape((1,Y_test_orig.shape[0]))
print("shape of training images data : ", X_train.shape)
print("shape of testing images data : ", X_test.shape)
print("shape of training labels images data : ", Y_train.shape)
print("shape of testing  labels images data : ", Y_test.shape)
print("Total  Classes = ", len(classes), " = ", classes)
index = 11
plt.imshow(X_train_orig[index])
print("This is ", classes[Y_train_orig[index]])
# INITIALIZING THE PARAMETERS
regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
def initialize_parameters():
    W1 = tf.get_variable(name='W1',shape=[25,12288],initializer=tf.contrib.layers.xavier_initializer(seed =1),regularizer=regularizer)
    b1 = tf.get_variable(name='b1',shape=[25,1],initializer=tf.zeros_initializer())
    W2 = tf.get_variable(name='W2',shape=[12,25],initializer=tf.contrib.layers.xavier_initializer(seed =1),regularizer=regularizer)
    b2 = tf.get_variable(name='b2',shape=[12,1],initializer=tf.zeros_initializer())
    W3 = tf.get_variable(name='W3',shape=[1,12],initializer=tf.contrib.layers.xavier_initializer(seed =1),regularizer=regularizer)
    b3 = tf.get_variable(name='b3',shape=[1,1],initializer=tf.zeros_initializer())
    parameters = {'W1':W1,
                  'b1':b1,
                  'W2':W2,
                  'b2':b2,
                  'W3':W3,
                  'b3':b3}
    return parameters
    
#SIGMOID FUNCTION
def sigmoid(Z):
    x = tf.placeholder(tf.float32,name='x')
    sigmoid = tf.sigmoid(x)
    with tf.Session() as sess:
        result = sess.run(sigmoid,feed_dict = {x:Z})
    return result
#FORWARD PROPOGATION
def forward_propagation(X,parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    Z1 = tf.add(tf.matmul(W1,X),b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2,A1),b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3,A2),b3)
    
    return Z3
    
def create_placeholders(n_x,n_h):
    X = tf.placeholder(tf.float32,shape=[n_x,None],name='X')
    Y = tf.placeholder(tf.float32,shape = [n_h,None],name= 'Y')
    return X,Y
def compute_cost(Z3, Y):
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
    
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
   
    cost_1 = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
    reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
    cost = tf.reduce_mean(cost_1+reg_term)
    
    return cost
##BACKPROPOGATION + WHOLE MODEL
def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
          num_epochs = 2000, print_cost = True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.
    
    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 209)
    Y_train -- test set, of shape (output size = 1, number of training examples = 209)
    X_test -- training set, of shape (input size = 12288, number of training examples = 50)
    Y_test -- test set, of shape (output size = 1, number of test examples = 50)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost

    X, Y = create_placeholders(n_x, n_y)

    parameters = initialize_parameters()
    
    Z3 = forward_propagation(X, parameters)
    
    cost = compute_cost(Z3, Y)
   
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0. # Defines a cost related to an epoch
            
            seed = seed + 1
            #minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            
           
            _ , Cost = sess.run([optimizer, cost], feed_dict={X: X_train, Y: Y_train})
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, Cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(Cost)
                
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")

    
        return parameters

parameters = model(X_train, Y_train, X_test, Y_test)
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
    A3 = tf.nn.sigmoid(Z3)
    return A3
    
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
    
    x = tf.placeholder("float", [12288, X.shape[1]])
    
    A3 = forward_propagation_for_predict(x, params)
    
    
    sess = tf.Session()
    prediction = sess.run(A3, feed_dict = {x: X})
        
    return prediction

    
def accuracy(label,pred):
    for i in range(0, label.shape[1]):
        if prediction[0,i] > 0.5:
            prediction[0,i] = 1
        else:
            prediction[0,i] = 0

    print("Accuracy: "  + str(np.sum((prediction == label)/(label.shape[1]))))
    
        
prediction = predict(X_test, parameters)
for i in range(0, prediction.shape[1]):
    if prediction[0,i] > 0.5:
        prediction[0,i] = 1
    else:
        prediction[0,i] = 0
accuracy(Y_test,prediction)
training_prediction = predict(X_train, parameters)
for i in range(0, training_prediction.shape[1]):
    if training_prediction[0,i] > 0.5:
        training_prediction[0,i] = 1
    else:
        training_prediction[0,i] = 0
for i in range(0, 209):
        if training_prediction[0,i] > 0.5:
            training_prediction[0,i] = 1
        else:
            training_prediction[0,i] = 0
print(" Training Accuracy: "  + str(np.sum((training_prediction == Y_train)/209)))