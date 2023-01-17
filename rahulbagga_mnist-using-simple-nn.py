import pandas as pd
import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
print(os.listdir("../input"))
train_df = pd.read_csv('../input/train.csv')
train = train_df.values
y = train[:,0]
X = train[:,1:]
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X, y, test_size=0.2)
def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y
X_train_norm = X_train_orig/255
X_test_norm = X_test_orig/255
X_train = X_train_norm.T
X_test = X_test_norm.T
y_train = convert_to_one_hot(y_train_orig, 10)
y_test = convert_to_one_hot(y_test_orig, 10)
print ("number of training examples = " + str(X_train.shape[1]))
print ("number of test examples = " + str(X_test.shape[1]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(y_test.shape))
def random_mini_batches(X, Y, mini_batch_size = 64):
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches
# Create Placeholders
def create_placeholders():
    with tf.device('/gpu:0'):
        X = tf.placeholder(tf.float32, [n_feature,None], name="X")
        Y = tf.placeholder(tf.float32, [n_classes,None], name="Y")
    
    return X, Y
def compute_cost(Z, Y):
    logits = tf.transpose(Z)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
    return cost
n_feature = X_train.shape[0]
n_classes = y_train.shape[0]
def initialize_parameters():
    with tf.device('/gpu:0'):
        tf.set_random_seed(1)
        W1 = tf.get_variable("W1", [10, 784], initializer = tf.contrib.layers.xavier_initializer(seed=1))
        b1 = tf.get_variable("b1", [10, 1], initializer = tf.zeros_initializer())
    
        parameters = {"W1": W1,
                  "b1": b1}
    
    return parameters
# Implements the forward propagation for the model: LINEAR -> SIGMOID ->LINEAR -> SOFTMAX
def forward_propagation(X, parameters):
    # Retrieve the parameters from the dictionary "parameters" 
    with tf.device('/gpu:0'):
        W1 = parameters['W1']
        b1 = parameters['b1']

        Z = tf.add(tf.matmul(W1, X), b1)
    
    return Z
def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001, num_epochs = 100, minibatch_size = 8, print_cost = True):
    with tf.device('/gpu:0'):
        X, Y = create_placeholders()
        costs = []
        m = X_train.shape[1]
        parameters = initialize_parameters()
        Z = forward_propagation(X, parameters)
        cross_entropy = compute_cost(Z, Y)
        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            epoch_cost = 0.
            num_minibatches = int(m / minibatch_size)
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _ , minibatch_cost = sess.run([train_step, cross_entropy], feed_dict={X: minibatch_X, Y: minibatch_Y})
                epoch_cost += minibatch_cost / num_minibatches
            if epoch % 10 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if epoch % 5 == 0:
                costs.append(epoch_cost)    
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        parameters = sess.run(parameters)
        correct_prediction = tf.equal(tf.argmax(Z), tf.argmax(Y))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
        print("Test_Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        
        # lets save the parameters in a variable
        print("Parameters have been trained!")
    
        return parameters
parameters = model(X_train, y_train, X_test, y_test)


