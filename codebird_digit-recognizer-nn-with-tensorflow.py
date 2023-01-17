import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt

import csv

import matplotlib.cm as cm

from tensorflow.python.framework import ops

import math

from random import randint

%matplotlib inline
def dense_to_1hot(labels, depth):

    C = tf.constant(depth, dtype=tf.int32)

    one_hot_matrix = tf.one_hot(labels, C, axis=0)

    one_hot = sess.run(one_hot_matrix)

    return one_hot
def random_mini_batches(X, Y, mini_batch_size = 128, seed = 0):

    np.random.seed(seed)

    m = X.shape[1]                  # number of training examples

    mini_batches = []

        

    # Step 1: Shuffle (X, Y)

    permutation = list(np.random.permutation(m))

    shuffled_X = X[:, permutation]

    shuffled_Y = Y[:, permutation].reshape((10,m))



    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.

    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning

    for k in range(0, num_complete_minibatches):

        mini_batch_X = shuffled_X[:, k * mini_batch_size: (k+1) * mini_batch_size]

        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: (k+1) * mini_batch_size]

        mini_batch = (mini_batch_X, mini_batch_Y)

        mini_batches.append(mini_batch)

    

    # Handling the end case (last mini-batch < mini_batch_size)

    if m % mini_batch_size != 0:

        mini_batch_X = shuffled_X[:, (k+1) * mini_batch_size: ]

        mini_batch_Y = shuffled_Y[:, (k+1) * mini_batch_size: ]

        mini_batch = (mini_batch_X, mini_batch_Y)

        mini_batches.append(mini_batch)

    

    return mini_batches
def create_placeholders(n_x, n_y):

    X = tf.placeholder(dtype=tf.float32, shape=(n_x, None))

    Y = tf.placeholder(dtype=tf.float32, shape=(n_y, None))

    keep_prob = tf.placeholder(dtype=tf.float32)

    return X, Y, keep_prob
def initialize_parameters():

    W1 = tf.get_variable("W1", [50,784], initializer = tf.contrib.layers.xavier_initializer(seed=1))

    b1 = tf.get_variable("b1", [50,1], initializer = tf.zeros_initializer())

    W2 = tf.get_variable("W2", [25,50], initializer = tf.contrib.layers.xavier_initializer(seed=1))

    b2 = tf.get_variable("b2", [25,1], initializer = tf.zeros_initializer())

    W3 = tf.get_variable("W3", [10, 25], initializer = tf.contrib.layers.xavier_initializer(seed=1))

    b3 = tf.get_variable("b3", [10,1], initializer = tf.zeros_initializer())

    parameters = {"W1": W1,

                  "b1": b1,

                  "W2": W2,

                  "b2": b2,

                  "W3": W3,

                  "b3": b3}

    

    return parameters
def forward_propagation(X, parameters, keep_prob):

    W1 = parameters['W1']

    b1 = parameters['b1']

    W2 = parameters['W2']

    b2 = parameters['b2']

    W3 = parameters['W3']

    b3 = parameters['b3']

    

    Z1 = tf.add(tf.matmul(W1, X), b1)                                            

    Z1_drop = tf.nn.dropout(Z1, keep_prob)

    A1 = tf.nn.relu(Z1_drop)                                             

    Z2 = tf.add(tf.matmul(W2, A1), b2)     

    Z2_drop = tf.nn.dropout(Z2, keep_prob)

    A2 = tf.nn.relu(Z2_drop)                                              

    Z3 = tf.add(tf.matmul(W3, A2), b3)  

    Z3_drop = tf.nn.dropout(Z3, keep_prob)

    

    return Z3_drop
def compute_cost(Z3, Y):

    #transpose Z3 and Y for the softmax_cross_entropy function

    logits = tf.transpose(Z3)

    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))    

    return cost
def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,

          num_epochs = 200, minibatch_size = 32, print_cost = True, keep_prob=0.5):

    

    ops.reset_default_graph()                         

    seed = 1                                          

    (n_x, m) = X_train.shape                          

    n_y = Y_train.shape[0]                            

    costs = []                                        

    

    # Create Placeholders of shape (n_x, n_y)

    X, Y, keep_prob_ph = create_placeholders(X_train.shape[0], Y_train.shape[0])

    

    # Initialize parameters

    parameters = initialize_parameters()

    

    # Forward propagation: Build the forward propagation in the tensorflow graph

    Z3 = forward_propagation(X, parameters, keep_prob_ph)

    

    # Cost function: Add cost function to tensorflow graph

    cost = compute_cost(Z3, Y)

    

    # Backpropagation: Define the tensorflow optimizer.

    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    

    # Initialize all the variables

    init = tf.global_variables_initializer()



    # Start the session to compute the tensorflow graph

    with tf.Session() as sess:

        

        # Run the initialization

        sess.run(init)

        

        # Do the training loop

        for epoch in range(num_epochs+1):



            epoch_cost = 0.                       

            num_minibatches = int(m / minibatch_size) 

            seed = seed + 1

            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)



            for minibatch in minibatches:



                # Select a minibatch

                (minibatch_X, minibatch_Y) = minibatch

                

                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y, keep_prob_ph: keep_prob})

                

                epoch_cost += minibatch_cost / num_minibatches



            # Print the cost every 100 epoch

            if print_cost == True and epoch % 100 == 0:

                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))

            if print_cost == True and epoch % 5 == 0:

                costs.append(epoch_cost)

        #Print the last epoch cost if it is not a multiple of 100

        if epoch % 100 != 0: 

            print ("Cost after epoch %i: %f" % (epoch, epoch_cost))    

            

        # plot the cost

        plt.plot(np.squeeze(costs))

        plt.ylabel('cost')

        plt.xlabel('iterations (per tens)')

        plt.title("Learning rate =" + str(learning_rate))

        plt.show()



        # save the parameters

        parameters = sess.run(parameters)

        print ("Parameters have been trained!")



        # Calculate the correct predictions

        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))



        # Calculate accuracy on the test set

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))



        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train, keep_prob_ph: 1}))

        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test, keep_prob_ph: 1}))

        

        return parameters
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

    

    x = tf.placeholder("float", [784, None])

    

    z3 = forward_propagation(x, params, 1)

    p = tf.argmax(z3)

    

    sess = tf.Session()

    prediction = sess.run(p, feed_dict = {x: X})

        

    return prediction
def write_predictions(predictions):

    with open('kaggle/submission_nn_tf.csv', 'w') as subs:

        subs.write("ImageId,Label\n")

        for i, pred in enumerate(predictions):

            subs.write(str(i+1)+','+str(pred)+'\n')
totalTrain = 42000

tainPart = 40000

testPart = 2000

data = []

sess = tf.Session()

with open('../input/train.csv', 'r') as train:

    pixelReader = csv.reader(train, delimiter=',')

    next(pixelReader, None)

    for row in pixelReader:

        data.append(row[0:])

        

data = np.array(data).astype(int)

np.random.shuffle(data)

data_train, data_test = data[:tainPart,:], data[tainPart:,:]

labels_train_dense = data_train[:, 0]

labels_test_dense = data_test[:, 0]



data_train = np.multiply(data_train[:, 1:].T, 1/255)

data_test = np.multiply(data_test[:, 1:].T, 1/255)

assert(data_train.shape == (784, tainPart))

assert(data_test.shape == (784, testPart))



labels_train = dense_to_1hot(labels_train_dense, 10)

labels_test = dense_to_1hot(labels_test_dense, 10)

assert(labels_train.shape == (10, tainPart))

assert(labels_test.shape == (10, testPart))



image_to_show = 10

plt.axis('off')

plt.imshow(data_train[:, image_to_show].reshape(28, 28),  cmap=cm.binary)

sess.close()



#read test data

test = []

with open('../input/test.csv', 'r') as train:

    pixelReader = csv.reader(train, delimiter=',')

    next(pixelReader, None)

    for row in pixelReader:

        test.append(row[0:])



test = np.array(test).astype(int)

test = np.multiply(test, 1.0 / 255.0)

test = test.T
parameters = model(data_train, labels_train, data_test, labels_test, learning_rate = 0.0007, num_epochs = 750, keep_prob=0.99, minibatch_size = 512)
predictions = predict(test, parameters)

#write_predictions(predictions)