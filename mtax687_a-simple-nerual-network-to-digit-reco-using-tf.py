import math

import datetime

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.cm as cm

import tensorflow as tf
TRAIN_SIZE = 38000

DEV_SIZE = 2000

TEST_SIZE = 2000

MINIBATCH_SIZE = 64

LEARNING_RATE = 0.0001
reader = pd.read_csv('../input/train.csv')



data = reader.values

images = data[:, 1:]

labels_dense = data[:, :1]

print("images shape:{}, labels shape:{}".format(images.shape, labels_dense.shape))


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
## tools

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



def display(image, image_width=28, image_height=28):

    # (784) => (28,28)

    one_image = image.reshape(image_width,image_height)

    

    new_f = plt.figure()

    plt.axis('off')

    plt.imshow(one_image, cmap=cm.binary)

    plt.show()

    plt.close()



# output image     

display(images[0])
# convert class labels_dense from scalars to one-hot vectors

labels = dense_to_one_hot(labels_dense, 10)

labels = labels.astype(np.uint8)

print("images shape:{}, labels shape:{}".format(images.shape, labels.shape))
train_images = images[:TRAIN_SIZE, :]

train_labels = labels[:TRAIN_SIZE, :]

dev_images = images[TRAIN_SIZE:TRAIN_SIZE+DEV_SIZE, :]

dev_labels = labels[TRAIN_SIZE:TRAIN_SIZE+DEV_SIZE, :]

test_images = images[TRAIN_SIZE+DEV_SIZE:, :]

test_labels = labels[TRAIN_SIZE+DEV_SIZE:, :]



print("train_images shape:{}".format(train_images.shape))

print("dev_images shape:{}".format(dev_images.shape))

print("test_images shape:{}".format(test_images.shape))



print("train_labels shape:{}".format(train_labels.shape))

print("dev_labels shape:{}".format(dev_labels.shape))

print("test_labels shape:{}".format(test_labels.shape))
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



def forward_propagation_with_dropout(X, params, keep_prob=0.5):

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
def model(X_train, Y_train, X_test, Y_test, learning_rate=LEARNING_RATE, decay_rate=0.9,

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

    learning_rate = learning_rate * (decay_rate**epoch_p)

    

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

    print(Z)

    prediction = tf.argmax(Z)

    

    #run tf Session

    with tf.Session() as sess:

        result = sess.run(prediction, feed_dict={X_placeholder: X})

    return result
@ex_time

def train():

    tf.reset_default_graph()

    params, costs_log = model(train_images.T, train_labels.T, dev_images.T, dev_labels.T,

                              num_epochs =11, learning_rate=LEARNING_RATE, optimizer="Adam",

                             layers_dims=[784, 50, 50, 30, 10], decay_rate=0.99)

    return params, costs_log



params, costs_log = train()
IMAGE_INDEX = 1028

res = predict(test_images[IMAGE_INDEX:IMAGE_INDEX+1].T, params)

display(test_images[IMAGE_INDEX])

print("{} predicted as {}".format(test_labels[IMAGE_INDEX], res))

def predict_test_images(filename='../input/test.csv'):

    reader = pd.read_csv(filename)

    data = reader.values

    images = data

    ids = np.arange(1, data.shape[0]+1).reshape(data.shape[0], 1)

    print("images shape:{}, test ids shape:{}".format(images.shape, ids.shape))

    p = predict(images.T, params).reshape(data.shape[0], 1)

    print("predicted as {}, shape {}".format(p, p.shape))

    res = np.concatenate((ids, p), axis=1)

    print(res.shape)

    np.savetxt("result.csv", res, delimiter=",", header="ImageId,Label", fmt="%d")

    return res
res = predict_test_images()
@ex_time

def test_minibatch_size(minibatch_size):

    tf.reset_default_graph()

    params, costs_log = model(train_images.T, train_labels.T, dev_images.T, dev_labels.T,

                              num_epochs =11, learning_rate=LEARNING_RATE, optimizer="Adam",

                             layers_dims=[784, 50, 50, 30, 10], decay_rate=0.99, minibatch_size=minibatch_size)

    return params, costs_log
params, costs_log = test_minibatch_size(32)
params, costs_log = test_minibatch_size(64)
params, costs_log = test_minibatch_size(128)
params, costs_log = test_minibatch_size(256)
params, costs_log = test_minibatch_size(256)
@ex_time

def test_layers_dims(layers_dims):

    tf.reset_default_graph()

    params, costs_log = model(train_images.T, train_labels.T, dev_images.T, dev_labels.T,

                              num_epochs =11, learning_rate=LEARNING_RATE, optimizer="Adam",

                             layers_dims=layers_dims, decay_rate=0.99, minibatch_size=32)

    return params, costs_log
params, costs_log = test_layers_dims([784, 3, 3, 10])
params, costs_log = test_layers_dims([784, 30, 30, 10])
params, costs_log = test_layers_dims([784, 30, 30, 30, 30, 10])