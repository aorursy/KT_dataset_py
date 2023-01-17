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
### NORMALIZATION

train_images /= 255.0

test_images /= 255.0



train_labels = np.eye(NUM_CLASSES)[train_labels.reshape(-1)]

test_labels = np.eye(NUM_CLASSES)[test_labels.reshape(-1)]



print("Train images shape is " + str(train_images.shape))

print("Train labels shape is " + str(train_labels.shape))

print("Test images shape is " + str(test_images.shape))

print("Test labels shape is " + str(test_labels.shape))
def create_placeholder(n_h, n_w, n_c, n_y):

    """

    

    Arguments:

    n_h (int) -- height of image;

    n_w (int) -- width of image;

    n_c (int) -- amount of image channels;

    n_y (int) -- amount of classes;

    

    Returns:

    X (tf.placeholder) -- placeholder for data imput;

    Y (tf.placeholder) -- placeholder for imput labels.

    

    """

    

    X = tf.placeholder(tf.float32, (None, n_h, n_w, n_c))

    Y = tf.placeholder(tf.float32, (None, n_y))

    

    return X, Y
def initialize_parameters():

    """

    

    Initialize parameters for neural network.

    

    """

    

    W1 = tf.get_variable('W1', [3, 3, 3, 8], initializer=tf.contrib.layers.xavier_initializer())

    W2 = tf.get_variable('W2', [5, 5, 8, 64], initializer=tf.contrib.layers.xavier_initializer()) # 8 в конце

    W3 = tf.get_variable('W3', [3, 3, 64, 128], initializer=tf.contrib.layers.xavier_initializer())

    

    parameters = dict()

    

    parameters['W1'] = W1

    parameters['W2'] = W2

    parameters['W3'] = W3

    

    return parameters
def forward_propogation(X, parameters):

    """

    

    Arguments:

    X (tf.placeholder) -- input images;

    parameters (dict) -- model parameters;

    

    Returns:

    

    Z3 -- output of NN.

    

    """

    

    W1 = parameters['W1']

    W2 = parameters['W2']

    W3 = parameters['W3']

    

    Z1 = tf.nn.conv2d(X, W1, strides=[1, 2, 2, 1], padding='VALID') # result shape: 25x25x8

    A1 = tf.nn.leaky_relu(Z1)

    A1_dropout = tf.nn.dropout(A1, rate=0.2)

    P1 = tf.nn.max_pool(A1_dropout, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID') # result shape: 12x12x8

    

    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 2, 2, 1], padding='SAME')

    A2 = tf.nn.leaky_relu(Z2)

    A2_dropout = tf.nn.dropout(A2, rate=0.2)

    P2 = tf.nn.max_pool(A2_dropout, ksize=[1, 5, 5, 1], strides=[1, 2, 2, 1], padding='SAME')

    

    Z3 = tf.nn.conv2d(P2, W3, strides=[1, 2, 2, 1], padding='SAME')

    A3 = tf.nn.leaky_relu(Z3)

    A3_dropout = tf.nn.dropout(A3, rate=0.2)

    P3 = tf.nn.max_pool(A3_dropout, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    

    P3 = tf.contrib.layers.flatten(P3)

    Z4 = tf.contrib.layers.fully_connected(P3, NUM_CLASSES, activation_fn=None)

    

    return Z4
def compute_cost(Z4, Y):

    """

    

    Arguments:

    Z3 - output of NN;

    Y - input labels.

    

    Returns:

    cost (tf.tensor) -- tensor of cost function.

    

    """

    

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z4, labels=Y))

    

    return cost
def model(X_train, Y_train, X_test, Y_test, num_epochs, learning_rate=0.1):

    

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

    

    m, n_h, n_w, n_c = X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3]

    n_y = Y_train.shape[1]

    

    X, Y = create_placeholder(n_h, n_w, n_c, n_y)

    

    parameters = initialize_parameters()

    

    Z4 = forward_propogation(X, parameters)

    

    cost = compute_cost(Z4, Y)

    

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    

    costs = []

    

    init = tf.global_variables_initializer()

    

    with tf.Session() as sess:

        

        sess.run(init)

        

        print("I am learning")

        

        start_time = time.time()

        

        for i in range(num_epochs):

            

            _, t_cost  = sess.run([optimizer, cost], feed_dict={X: X_train, Y: Y_train})

            

            if i % 10 == 0 or i == num_epochs - 1:

                print("Cost after " + str(i) + " epoch is " + str(t_cost))

                costs.append(t_cost)

        

        print("Done!")

        

        end_time = time.time()

                

        parameters = sess.run(parameters)

        

        correct_pred = tf.equal(tf.argmax(Z4, 1), tf.argmax(Y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_pred, 'float'))

        

        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})

        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})

        

        print("Train accuracy is " + str(train_accuracy))

        print("Test accuracy is " + str(test_accuracy))

        print("The speed of the algorithm is " + str(end_time - start_time) + " seconds")

        

    plt.plot(np.squeeze(costs))

    plt.xlabel('iterations')

    plt.ylabel('cost')

    plt.title("Learning rate is " + str(learning_rate))

    plt.show()
model(train_images, train_labels, test_images, test_labels, 700)