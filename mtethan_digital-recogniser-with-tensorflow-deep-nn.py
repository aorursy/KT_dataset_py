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
import tensorflow as tf
import matplotlib.pyplot as plt
# Import data
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
#train_data.describe()
#test_data.describe()
X_train_orig = train_data.drop("label", axis = 1) /255
Y_train_orig = train_data.label
X_test = test_data / 255
from sklearn.model_selection import train_test_split
# Split train and dev set
num_test = 0.3
X_train, X_dev, Y_train, Y_dev = train_test_split(X_train_orig , Y_train_orig, test_size = num_test, shuffle = False)
X_train.shape, Y_train.shape, X_dev.shape, Y_dev.shape
# See a training sample
index = 2000
sample = X_train.iloc[index].reshape(28, 28)
plt.imshow(sample)
print("y =" + str(Y_train[index]))
tf.reset_default_graph
def get_one_hot(z):
    a = tf.placeholder(tf.int64, name = "y")
    
    one_hot = tf.one_hot(a, depth = 10)
    
    Y = tf.transpose(one_hot)
    with tf.Session() as sess:
        result = sess.run(Y, feed_dict = {a : z})
    return result
Y_train = get_one_hot(Y_train)
Y_dev = get_one_hot(Y_dev)
Y_train.shape, Y_dev.shape
def create_placeholder():
    X = tf.placeholder(tf.float32, name = "X")
    Y = tf.placeholder(tf.float32, name = "Y")
    return X, Y
def initialise_parameters():
    W1 = tf.get_variable("W1", [150, 784], initializer = tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", [150, 1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [50, 150], initializer = tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", [50, 1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [10, 50], initializer = tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", [10, 1], initializer = tf.zeros_initializer())
    
    parameters = {"W1" : W1,
                  "b1" : b1,
                  "W2" : W2,
                  "b2" : b2,
                  "W3" : W3,
                  "b3" : b3}
    return parameters
def forward_prop(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    Z1 = tf.add(tf.matmul(W1, tf.cast(tf.transpose(X),"float32")), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)
    return Z3
def compute_cost(Z3, Y, parameters, lambd):
    # important !!
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    m = Y.get_shape()[1]
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= logits, labels = labels))
    # add regularization term
    W1 = parameters['W1']
    
    W2 = parameters['W2']
    
    W3 = parameters['W3']
    
    cost_reg1 = tf.reduce_sum(tf.square(W1))
    cost_reg2 = tf.reduce_sum(tf.square(W2))
    cost_reg3 = tf.reduce_sum(tf.square(W3))
    costs_regular = tf.add(tf.add(cost_reg1, cost_reg2), cost_reg3)
    costs_regularization = tf.divide(tf.multiply(lambd , costs_regular), 29400) *0.5
    cost = cost + costs_regularization
    return cost
def NNmodel(X_train, Y_train, X_dev, Y_dev, learning_rate, num_interation, lambd, print_cost = True):
    tf.reset_default_graph()
    X, Y = create_placeholder()
    parameters = initialise_parameters()
    Z3 = forward_prop(X, parameters)
    cost = compute_cost(Z3, Y, parameters, lambd)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    costs = []
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(num_interation):
            _, c = sess.run([optimizer, cost], feed_dict = {X : X_train, Y : Y_train})
            if i % 2 == 0:
                costs.append(c)
            if i % 10 == 0 and print_cost:
                print("Cost after {} interation = {}".format(i, c))
                
        plt.plot(costs)
        plt.xlabel("interation per tens")
        plt.ylabel("cost")
        plt.title("learning rate =" + str(learning_rate))
        plt.show
        
        parameters = sess.run(parameters)
        
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Train Accuracy :", sess.run(accuracy, feed_dict = {X : X_train, Y : Y_train}))
        print("Dev Accuracy :", sess.run(accuracy, feed_dict = {X : X_dev, Y : Y_dev}))
        
    return parameters
modelparameters = NNmodel(X_train, Y_train, X_dev, Y_dev, 0.01, 200, 100., print_cost = True)
def predict(X, parameters):
    Z = tf.placeholder(tf.float32, name = "Z")
    Z3 = forward_prop(X, parameters)
    Y_pred = tf.argmax(Z3)
    with tf.Session() as sess:
        result = sess.run(Y_pred, feed_dict = {Z : X})
    return result
Y_pred = predict(X_test, modelparameters)
Y_pred.shape
ImageId = np.arange(1,28001)
Submission = pd.DataFrame({"ImageId" : ImageId, "Label" : Y_pred})
Submission.to_csv("Submission.csv", index = False)
Submission.head()
