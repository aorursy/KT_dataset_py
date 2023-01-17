# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
import datetime
import random
import math
import cv2
import tensorflow as tf
import numpy as np # linear algebra
import matplotlib.pyplot as plt
import matplotlib.cm as cm
## constants
TRAIN_DIR = "../input/train/"
TEST_DIR = "../input/test/"
TRAIN_SIZE = 22500
TEST_SIZE = 2500
DEV_RATIO = 0.1
IMAGE_HEIGHT = IMAGE_WIDTH = 128

LEARNING_RATE = 0.0001
MINIBATCH_SIZE = 32
INPUT_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH * 3
OUTPUT_SIZE = 2
## tool functions
def ex_time(func):
    start_time = datetime.datetime.now()
    
    def wrapper(*args, **kwargs):
        print("start time: {}".format(start_time))
        res = func(*args, **kwargs)
        
        end_time = datetime.datetime.now()
        ex_time = end_time - start_time
        print("end time: {}".format(end_time))
        print("excute time: {} seconds".format(ex_time.seconds))

        return res
       
    return wrapper

def display(image, image_width=IMAGE_HEIGHT, image_height=IMAGE_HEIGHT, interpolation=3):
    # (784) => (28,28)
    one_image = image.reshape(image_width,image_height, interpolation)
    
    new_f = plt.figure()
    plt.axis('off')
    plt.imshow(one_image, cmap=cm.binary)
    plt.show()
    plt.close()
## data utility functions
def dense_to_one_hot(labels_dense, num_classes):
    """
    # convert class labels from scalars to one-hot vectors
    # 0 => [1 0 0 0 0 0 0 0 0 0]
    # 1 => [0 1 0 0 0 0 0 0 0 0]
    # ...
    # 9 => [0 0 0 0 0 0 0 0 0 1]
    """
    num_labels = labels_dense.shape[0]
    #print("num_labels:", num_labels)
    index_offset = np.arange(num_labels) * num_classes
    #print("index_offset:", index_offset)
    labels_one_hot = np.zeros((num_labels, num_classes))
    #print("labels_one_hot:", labels_one_hot)
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    #print(index_offset + labels_dense.ravel())
    #print("labels_one_hot2:", labels_one_hot)
    return labels_one_hot

def split_data(images, labels, dev_ratio=DEV_RATIO):
    dev_count = int(labels.shape[1] * DEV_RATIO)
    dev_images = images[:, :dev_count]
    train_images = images[:, dev_count:]
    dev_labels = labels[:, :dev_count]
    train_labels = labels[:, dev_count:]
    print("train images shape: {}, train labels shape:{}, \
    dev images shape: {}, dev labels shape: {}".format(train_images.shape, train_labels.shape, dev_images.shape, dev_labels.shape))
    return train_images, train_labels, dev_images, dev_labels
#@ex_time
def pre_data(dirname=TRAIN_DIR, file_count=1000):
    all_filenames = os.listdir(dirname)
    random.shuffle(all_filenames)
    filenames = all_filenames[:file_count]
    
    ## images
    images = np.zeros((file_count, IMAGE_HEIGHT*IMAGE_WIDTH*3))
    for i in range(file_count):
        imgnd_origin = cv2.imread(dirname+filenames[i])
        imgnd_resized = cv2.resize(imgnd_origin, (IMAGE_HEIGHT, IMAGE_WIDTH), interpolation=cv2.INTER_CUBIC)
        imgnd_flatten = imgnd_resized.reshape(1,-1)
        images[i] = imgnd_flatten
    
    ## labels from filenames
    labels_list = ["dog" in filename for filename in filenames]
    labels = np.array(labels_list, dtype='int8').reshape(file_count, 1)
    
    ## shuffle
    permutation = list(np.random.permutation(labels.shape[0]))
    shuffled_labels = labels[permutation, :]
    shuffled_images = images[permutation, :]
    
    ## dense to one hot
    labels = dense_to_one_hot(shuffled_labels, OUTPUT_SIZE)
    ## normalization
    images = shuffled_images/255.0
    
    return images.T, labels.T

images, labels = pre_data(file_count=100)

train_images, train_labels, dev_images, dev_labels = split_data(images, labels)
print(train_images.shape, train_labels.shape, dev_images.shape, dev_labels.shape)
def init_params(layers_dims):
    '''
    Initializes parameters to build a neural network with tensorflow.
    
    Arguments:
        layers_dims: python array (list) containing the size of each layer.
                     e.g.:[n_x=n_l0, n_l1, n_l2, ..., n_lL=n_Y].n_l2 is size of second hidden layer.
    
    Returns:
        params: a dictionary of tensors containing W1, b1, W2, b2, ..., WL, bL. e.g.:
                {
                    "W1": W1,
                    "b1": b1,
                    "W2": W2,
                    "b2": b2
                }
        
    
    '''
    L = len(layers_dims)
    params = {}
    
    for l in range(1, L):
        params['W' + str(l)] = tf.get_variable('W' + str(l), [layers_dims[l], layers_dims[l-1]], initializer = tf.contrib.layers.xavier_initializer())
        params['b' + str(l)] = tf.get_variable('b' + str(l), [layers_dims[l], 1], initializer = tf.zeros_initializer())
    return params

def forward_propagation_with_dropout(X, params, keep_prob=0.1):
    """
    Implements the forward propagation: LINEAR -> RELU + DROPOUT -> LINEAR -> RELU + DROPOUT -> LINEAR -> SIGMOID.
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    params -- python dictionary containing your parameters(tf.Variable) "W1", "b1", "W2", "b2", ..., "WL", "bL":
                    W1 -- weight matrix of shape (20, 2)
                    b1 -- bias vector of shape (20, 1)
                    W2 -- weight matrix of shape (3, 20)
                    b2 -- bias vector of shape (3, 1)
                    W3 -- weight matrix of shape (1, 3)
                    b3 -- bias vector of shape (1, 1)
    keep_prob - probability of keeping a neuron active during drop-out, scalar
    
    Returns:
    ZL -- the output of the last LINEAR unit
    """
    keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob")
    L = int(len(params)/2)
    cache = {"A0": X}
    for l in range(1, L+1):
        cache["Z"+str(l)] = tf.matmul(params["W"+str(l)], cache["A"+str(l-1)]) + params["b"+str(l)]
        cache["Droped_Z"+str(l)] = tf.nn.dropout(cache["Z"+str(l)], keep_prob)
        cache["A"+str(l)] = tf.nn.relu(cache["Z"+str(l)])
    return cache["Z"+str(L)]

def compute_cost(Z, Y):
    """
    Computes the cost
    
    Arguments:
    Z -- output of forward propagation (output of the last LINEAR unit), of shape (n_Y, number of examples)
    Y -- labels vector placeholder, same shape as Z
    
    Returns:
    cost - Tensor of the cost function
    """
    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z)
    labels = tf.transpose(Y)
    
    # compute cost
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return cost
    
def random_mini_batches(X, Y, mini_batch_size = 64):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[1]                  # number of training examples
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(math.floor(m/mini_batch_size)) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[:, k*mini_batch_size:(k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k*mini_batch_size:(k+1)*mini_batch_size]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[:, num_complete_minibatches*mini_batch_size:]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches*mini_batch_size:]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches
def model(X_train, Y_train, X_test, Y_test, learning_rate=LEARNING_RATE, decay_rate=0,
          num_epochs=2500, minibatch_size=MINIBATCH_SIZE, print_cost=True,
          layers_dims=[784, 3,3,10], optimizer="GradientDecent"):
    '''
    Implements a tensorflow neural network: e.g. LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    Arguments:
    X_train -- training set, of shape (input size, number of training examples)
    Y_train -- test set, of shape (output size, number of training examples)
    X_test -- training set, of shape (input size, number of training examples)
    Y_test -- test set, of shape (output size, number of test examples)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 10 epochs
    layers_dims: python array (list) containing the size of each layer.
                 e.g.:[n_x=n_l0, n_l1, n_l2, ..., n_lL=n_Y].n_l2 is size of second hidden layer.
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    '''
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs_log = []
    
    X = tf.placeholder(dtype=tf.float32, shape=(n_x, None), name="X")
    Y = tf.placeholder(dtype=tf.float32, shape=(n_y, None), name="Y")
    epoch_p = tf.placeholder(dtype=tf.float32, name="epoch_p")
    #### tool init_params
    params = init_params(layers_dims)
    #### tool foward_propa
    Z = forward_propagation_with_dropout(X, params)
    #### tool compute_cost
    cost = compute_cost(Z, Y)
    #### learning_rate decay
    learning_rate = learning_rate * np.power((10/(epoch_p+1)), decay_rate)

    
    if optimizer == "GradientDescent":
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    elif optimizer == "Adam":
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    ## let's go
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            epoch_cost = 0
            n_minibatches = int(m/minibatch_size)
            #### tool random_mini_batches
            minibatches = random_mini_batches(X_train, Y_train, mini_batch_size=minibatch_size)
            
            for minibatch in minibatches:
                mini_X, mini_Y = minibatch
                o, minibatch_cost = sess.run((optimizer, cost), feed_dict={X: mini_X, Y: mini_Y, epoch_p: epoch})
                epoch_cost += minibatch_cost / n_minibatches
                
            if print_cost and (epoch%10 == 0):
                print("Cost after epoch {} is {}".format(epoch, epoch_cost))

            if print_cost and (epoch%2 == 0):
                costs_log.append(epoch_cost)
        plt.plot(np.squeeze(costs_log))
        plt.ylabel('cost')
        plt.xlabel('iterations (per 5)')
        plt.title("Learning Rate = {}".format(learning_rate))
        plt.show()
        # lets save the parameters in a variable
        params = sess.run(params)
        print ("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        return params, costs_log
    
#@ex_time
def train():
    tf.reset_default_graph()
    params, costs_log = model(train_images, train_labels, dev_images, dev_labels,
                              num_epochs =101, learning_rate=LEARNING_RATE, optimizer="Adam",
                             layers_dims=[INPUT_SIZE, 3, OUTPUT_SIZE], decay_rate=100)
    return params, costs_log

params, costs_log = train()
def predict(X, params):
    """
    Implements a tensorflow neural network prediction using given params.
    
    Arguments:
    X -- Images to predict, ndarry set of shape (input size, number of images)
    params -- parameters learnt by some model. They can then be used to predict.
    
    Returns:
    result -- list of prediction shape of (1, number of input images)
    """
    # conver X to tf Placeholder
    X_placeholder = tf.placeholder(tf.float32, shape=X.shape, name="X_placeholder")
    # conver params to tensors
    L = int(len(params)/2)
    params_tensor = {}
    for l in range(1, L+1):
        params_tensor["W"+str(l)] = tf.convert_to_tensor(params["W"+str(l)])
        params_tensor["b"+str(l)] = tf.convert_to_tensor(params["b"+str(l)])
    # foward propagation
    Z = forward_propagation_with_dropout(X_placeholder, params_tensor, keep_prob=1.0)
    prediction = tf.nn.softmax(Z)
    
    #run tf Session
    with tf.Session() as sess:
        result = sess.run(prediction, feed_dict={X_placeholder: X})
    return result
res = predict(dev_images, params)
res
def why_time(func):
    """
    其中，输出每列的具体解释如下：
    ncalls：表示函数调用的次数；
    tottime：表示指定函数的总的运行时间，除掉函数中调用子函数的运行时间；
    percall：（第一个percall）等于 tottime/ncalls；
    cumtime：表示该函数及其所有子函数的调用运行的时间，即函数开始调用到返回的时间；
    percall：（第二个percall）即函数运行一次的平均时间，等于 cumtime/ncalls；
    filename:lineno(function)：每个函数调用的具体信息；
    """
    
    import cProfile
    cmd = "{}()".format(func.__name__)
    
    # 直接把分析结果打印到控制台
    cProfile.run(cmd, sort="cumulative")
    
    # print out to file
    #cProfile.run("test()", filename="result.out")
    
    # sort by excute time
    #cProfile.run("test()", filename="result.out", sort="cumulative")
why_time(train)
images, labels = pre_data(file_count=1000)
train_images, train_labels, dev_images, dev_labels = split_data(images, labels)

params, costs_log = train()
tf.reset_default_graph()
params, costs_log = model(train_images, train_labels, dev_images, dev_labels,
                              num_epochs =101, learning_rate=LEARNING_RATE, optimizer="Adam",
                             layers_dims=[INPUT_SIZE, 30, 20, OUTPUT_SIZE], decay_rate=100)
tf.reset_default_graph()
params, costs_log = model(train_images, train_labels, dev_images, dev_labels,
                              num_epochs =101, learning_rate=LEARNING_RATE, optimizer="Adam",
                             layers_dims=[INPUT_SIZE, 30, 20, OUTPUT_SIZE], decay_rate=100)
tf.reset_default_graph()
params, costs_log = model(train_images, train_labels, dev_images, dev_labels,
                              num_epochs =101, learning_rate=LEARNING_RATE, optimizer="Adam",
                             layers_dims=[INPUT_SIZE, 3, 20, OUTPUT_SIZE], decay_rate=100)
tf.reset_default_graph()
params, costs_log = model(train_images, train_labels, dev_images, dev_labels,
                              num_epochs =1001, learning_rate=LEARNING_RATE, optimizer="Adam",
                             layers_dims=[INPUT_SIZE, 30, 20, OUTPUT_SIZE], decay_rate=100)
