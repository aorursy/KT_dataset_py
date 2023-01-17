# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import matplotlib.pyplot as plt

import random

import time

from skimage import io, transform



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def load_images(path):

    """

    

    Arguments:

    path (str) -- path to images

    

    Returns:

    images (list) -- list with images;

    labels (list) -- list with labels;

    

    """

    classes = [i for i in range(43)]

    images = []

    labels = []

    for new_class in classes:

        new_path = path + str(new_class) + "/"

        file_names = [os.path.join(new_path, f)

                     for f in os.listdir(new_path)]

        

        for file in file_names:

            images.append(io.imread(file))

            labels.append(new_class)

    

    return images, labels





train_images, train_labels = load_images("../input/train/")       
test_data = pd.read_csv("../input/Test.csv")

test_labels = test_data['ClassId'].values

paths = test_data['Path'].values





def load_test_images(paths):

    """

    

    Arguments:

    paths (str) -- paths to images;

    

    Returns:

    images (list) - list with images;

    

    """

    images = []

    for f in paths:

        image = io.imread('../input/test/' + f.replace('Test/', ''))

        images.append(image)

    return images



test_images = load_test_images(paths)
NUM_CLASSES = 43

train_images = np.array(train_images)

train_labels = np.array(train_labels)

test_images = np.array(test_images)

test_labels = np.array(test_labels)



print("Shape of train images is " + str(train_images.shape))

print("Shape of train labels is " + str(train_labels.shape))

print("Shape of test images is " + str(test_images.shape))

print("Shape of test labels is " + str(test_labels.shape))

print("Amount of classes is " + str(NUM_CLASSES))
def show_images(images, labels, amount):

    """

    

    Arguments:

    images (np.array) -- list with images;

    labels (np.array) -- list with labels

    amount (int) -- amount of images to show.

    

    """

    

    for i in range(amount):

        index = int(random.random() * len(images))

        plt.axis('off')

        plt.imshow(images[index])

        plt.show()

        

        print("Size of this image is " + str(images[index].shape))

        print("Class of the image is " + str(labels[index]))



        

print("Train images")

show_images(train_images, train_labels, 3)
print("Test images")

show_images(test_images, test_labels, 3)
def change_images_size(images, size):

    """

    

    Arguments:

    images (np.array) -- list with images;

    size (tuple) -- new shape of images.

    

    Returns:

    new_images (np.array) -- np.array with new shape of images. 

    

    """

    

    print("Change shape...")

    new_images = np.array([transform.resize(image, size) for image in images])

    

    print("Done!")

    return new_images



train_images = change_images_size(train_images, (51, 51))

test_images = change_images_size(test_images, (51, 51))
train_images /= 255.

test_images /= 255.



train_images = train_images.reshape(train_images.shape[0], -1).T

test_images = test_images.reshape(test_images.shape[0], -1).T

train_labels = np.eye(NUM_CLASSES)[train_labels.reshape(-1)].T

test_labels = np.eye(NUM_CLASSES)[test_labels.reshape(-1)].T



print("Train images shape is " + str(train_images.shape))

print("Train labels shape is " + str(train_labels.shape))

print("Test images shape is " + str(test_images.shape))

print("Test labels shape is " + str(test_labels.shape))
def create_placeholder(n_x, n_y):

    """

    

    Arguments:

    n_x (int) -- amount of input neurons;

    n_y (int) -- amount of classes;

    

    Returns:

    X (tf.placeholder) -- placeholder for data imput;

    Y (tf.placeholder) -- placeholder for imput labels.

    

    """

    

    X = tf.placeholder(tf.float32, (n_x, None))

    Y = tf.placeholder(tf.float32, (n_y, None))

    

    return X, Y
def initialize_parameters():

    """

    

    Initialize parameters for neural network.

    

    """

    

    W1 = tf.get_variable('W1', [625, 7803], initializer=tf.contrib.layers.xavier_initializer())

    b1 = tf.get_variable('b1', [625, 1], initializer=tf.zeros_initializer())

    W2 = tf.get_variable('W2', [125, 625], initializer=tf.contrib.layers.xavier_initializer())

    b2 = tf.get_variable('b2', [125, 1], initializer=tf.zeros_initializer())

    W3 = tf.get_variable('W3', [75, 125], initializer=tf.contrib.layers.xavier_initializer())

    b3 = tf.get_variable('b3', [75, 1], initializer=tf.zeros_initializer())

    W4 = tf.get_variable('W4', [43, 75], initializer=tf.contrib.layers.xavier_initializer())

    b4 = tf.get_variable('b4', [43, 1], initializer=tf.zeros_initializer())

    

    parameters = dict()

    

    parameters['W1'] = W1

    parameters['b1'] = b1

    parameters['W2'] = W2

    parameters['b2'] = b2

    parameters['W3'] = W3

    parameters['b3'] = b3

    parameters['W4'] = W4

    parameters['b4'] = b4

    

    return parameters
def forward_propogation(X, parameters, training):

    """

    

    Arguments:

    X (tf.placeholder) -- input images;

    parameters (dict) -- model parameters;

    

    Returns:

    

    Z3 -- output of NN.

    

    """

    

    W1 = parameters['W1']

    b1 = parameters['b1']

    W2 = parameters['W2']

    b2 = parameters['b2']

    W3 = parameters['W3']

    b3 = parameters['b3']

    W4 = parameters['W4']

    b4 = parameters['b4']

    

    Z1 = tf.add(tf.matmul(W1, X), b1)                                              

    A1 = tf.nn.leaky_relu(Z1)

    A1_dropout = tf.layers.dropout(A1, training=training)

    Z2 = tf.add(tf.matmul(W2, A1_dropout), b2)                                            

    A2 = tf.nn.leaky_relu(Z2)

    A2_dropout = tf.layers.dropout(A2, training=training)

    Z3 = tf.add(tf.matmul(W3, A2_dropout), b3)

    A3 = tf.nn.leaky_relu(Z3)

    A3_dropout = tf.layers.dropout(A3, training=training)

    Z4 = tf.add(tf.matmul(W4, A3_dropout), b4)

    

    return Z4
def compute_cost(Z3, Y):

    """

    

    Arguments:

    Z3 - output of NN;

    Y - input labels.

    

    Returns:

    cost (tf.tensor) -- tensor of cost function.

    

    """

    logits = tf.transpose(Z3)

    labels = tf.transpose(Y)

    

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    

    return cost
def model(X_train, Y_train, X_test, Y_test, num_epochs, learning_rate=0.001):

    

    """

    

    Arguments:

    X_train (np.array) -- train images for NN;

    Y_train (np.array) -- train labels for NN;

    X_test (np.array) -- test_images for NN;

    Y_test (np,array) -- test_labels for NN;

    num_epochs (int) -- amount of epochs.

    

    """

    tf.reset_default_graph()

    tf.set_random_seed(1)

    

    n_x, m = X_train.shape[0], X_train.shape[1]

    n_y = Y_train.shape[0]

    

    X, Y = create_placeholder(n_x, n_y)

    

    parameters = initialize_parameters()

    

    Z3 = forward_propogation(X, parameters, True)

    

    Z3_test = forward_propogation(X, parameters, False)

    

    cost = compute_cost(Z3, Y)

    

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    

    costs = []

    

    init = tf.global_variables_initializer()

    

    max_accuracy = 0

    

    with tf.Session() as sess:

        

        sess.run(init)

        

        print("I am learning")

        

        start_time = time.time()

        

        for i in range(num_epochs):

            

            _, t_cost  = sess.run([optimizer, cost], feed_dict={X: X_train, Y: Y_train})

            

            if i % 10 == 0 or i == num_epochs - 1:

                print("Cost after " + str(i) + " epoch is " + str(t_cost) + " |", end=" ")

                costs.append(t_cost)

                

                correct_pred = tf.equal(tf.argmax(Z3), tf.argmax(Y))

                accuracy = tf.reduce_mean(tf.cast(correct_pred, 'float'))

                

                test_pred = tf.equal(tf.argmax(Z3_test), tf.argmax(Y))

                accuracy_2 = tf.reduce_mean(tf.cast(test_pred, 'float'))

        

                train_accuracy = accuracy.eval({X: X_train, Y: Y_train})

                test_accuracy = accuracy_2.eval({X: X_test, Y: Y_test})

                

                if test_accuracy > max_accuracy:

                    max_accuracy = test_accuracy

                    epoch = i

                

                print("Train accuracy is " + str(train_accuracy) + " |", end=" ")

                print("Test accuracy is " + str(test_accuracy))

        

        print("Done!")

        print()

        

        end_time = time.time()

                

        parameters = sess.run(parameters)

        

        print("The best test accuracy is " + str(max_accuracy))

        print("Amount of epochs is " + str(epoch))

        print("The speed of the algorithm is " + str(end_time - start_time) + " seconds")

        

    plt.plot(np.squeeze(costs))

    plt.xlabel('iterations')

    plt.ylabel('cost')

    plt.title("Learning rate is " + str(learning_rate))

    plt.show()
model(train_images, train_labels, test_images, test_labels, 1000)