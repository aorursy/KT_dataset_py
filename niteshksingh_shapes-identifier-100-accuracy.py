import os

import numpy as np

import cv2

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.python.framework import ops

from sklearn.model_selection import train_test_split

from math import floor

IMG_S1,IMG_S2 = 80,80

X=[]

Y=[]

def imgread_scale_resize(PATH,img):

    img_array = cv2.imread(os.path.join(PATH,img),cv2.IMREAD_GRAYSCALE)

    img_array = cv2.resize(img_array , (IMG_S1,IMG_S2))

    return img_array



def createTrainingData():

    datadir = "/kaggle/input/four-shapes/shapes"

    CATEGORIES = ["circle","square","star","triangle"]

    a = 0

    for ele in CATEGORIES:

        PATH = os.path.join(datadir,ele)

        class_num = CATEGORIES.index(ele)

        for img in os.listdir(PATH):

            try:

                img_array = imgread_scale_resize(PATH,img)

                X.append(list(img_array))

                Y.append(class_num)

            except Exception as e:

                pass

            a+=1

createTrainingData()
#Reshape and flatten X and Y

X = np.array(X)

Y = np.array(Y)



X = X.reshape(X.shape[0],-1).T

Y = Y.T

X = X/255 #normalized



#Output config C: DEPth

def one_hottie(labels,C):

    sess = tf.Session()

    C = tf.constant(C) #four shapes

    One_hot_matrix = tf.one_hot(labels,C,axis=0)

    Y = sess.run(One_hot_matrix)

    sess.close()

    return Y

Y = one_hottie(Y,4)



#splitting

X_train, X_test, y_train, y_test = train_test_split(X.T, Y.T, test_size = 0.13, random_state = 0)

X_train, X_test, y_train, y_test = X_train.T, X_test.T, y_train.T, y_test.T
#creating placeholders

def create_placeholders(n_x,n_y):

    """

    Creates two placeholder of shape (n_x,none) and (n_y,none)

    n_x : shape of 1 image vector - height * width * colors(RGB or grayscale etc)

    n_y : no. of classes

    """

    X = tf.placeholder(dtype = tf.float32, shape = (n_x,None),name = "X")

    Y = tf.placeholder(dtype = tf.float32, shape = (n_y,None),name = "Y")

    return X,Y



# SHAPE OF NET : LINEAR(Z1) => RELU(A1) => LINEAR(z2) => RELU(a2) => LINEAR(z3) => SOFTMAX(y)



#model functions

def initialize_parameters():

    """

        W1 : [25, shape of 1 image vector]

        b1 : [25, 1]

        W2 : [12, 25]

        b2 : [12, 1]

        W3 : [6, 12]

        b3 : [6, 1]

    Returns:

    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3

    """ 

    W1 = tf.get_variable(name= "W1", shape= [25,6400], initializer = tf.contrib.layers.xavier_initializer())

    b1 = tf.get_variable(name= "b1", shape= [25,1], initializer = tf.zeros_initializer())

    W2 = tf.get_variable(name= "W2", shape= [12,25], initializer = tf.contrib.layers.xavier_initializer())

    b2 = tf.get_variable(name= "b2", shape= [12,1], initializer = tf.zeros_initializer())

    W3 = tf.get_variable(name= "W3", shape= [4,12], initializer = tf.contrib.layers.xavier_initializer())

    b3 = tf.get_variable(name= "b3", shape= [4,1], initializer = tf.zeros_initializer())



    parameters = {"W1": W1,

                  "b1": b1,

                  "W2": W2,

                  "b2": b2,

                  "W3": W3,

                  "b3": b3}

    return parameters



#forward prop 

def forward_propagation(X, parameters):

    """

    Arguments:

    X : placeholder of shape (input size, number of examples)

    parameters : dict of W and b



    Returns:

    Z3 -- the output of the last LINEAR unit

    """

     

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

    

    return Z3  #Linear Z3



def compute_cost(Z3, Y):

    """

    Computes the cost

    

    Arguments:

    Z3 : Linear unit output of forward prop ...of shape (6, number of examples)

    Y : True vales placeholder, same shape as Z3

    

    Returns:

    cost - Tensor of the cost function

    """

    

    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)

    logits = tf.transpose(Z3)

    labels = tf.transpose(Y)



    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits ,labels = labels))

    

    return cost



def random_mini_batches(X_train, y_train, minibatch_size, seed):

    m = X_train.shape[1]                  # number of training examples

    mini_batches = []

    np.random.seed(seed)

    num_complete_minibatches = floor(m/minibatch_size) # number of mini batches of size minibatch_size in your partitionning

    for k in range(0, num_complete_minibatches):

        mini_batch_X = X_train[:, k * minibatch_size : k * minibatch_size + minibatch_size]

        mini_batch_Y = y_train[:, k * minibatch_size : k * minibatch_size + minibatch_size]

        mini_batch = (mini_batch_X, mini_batch_Y)

        mini_batches.append(mini_batch)

    

    # Handling the end case (last mini-batch < minibatch_size)

    if m % minibatch_size != 0:

        mini_batch_X = X_train[:, num_complete_minibatches * minibatch_size : m]

        mini_batch_Y = y_train[:, num_complete_minibatches * minibatch_size : m]

        mini_batch = (mini_batch_X, mini_batch_Y)

        mini_batches.append(mini_batch)

    

    return mini_batches
def model(X_train, y_train, X_test, y_test, learning_rate = 0.001,

          num_epochs = 1000, minibatch_size = 50, print_cost = True):

    """

    Arguments:

    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)

    y_train -- test set, of shape (output size = 6, number of training examples = 1080)

    X_test -- training set, of shape (input size = 12288, number of training examples = 120)

    y_test -- test set, of shape (output size = 6, number of test examples = 120)

    learning_rate -- learning rate of the optimization

    num_epochs -- number of epochs of the optimization loop

    minibatch_size -- size of a minibatch

    print_cost -- True to print the cost every 100 epochs

    

    Returns:

    parameters -- parameters learnt by the model. They can then be used to predict.

    """

    

    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables

    tf.set_random_seed(1)                             # to keep consistent results

    seed = 3                                          # to keep consistent results

    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)

    n_y = y_train.shape[0]                            # n_y : output size

    costs = []                                        # To keep track of the cost

    

    X, Y = create_placeholders(n_x,n_y)

    parameters = initialize_parameters()

    

    Z3 = forward_propagation(X,parameters)



    cost = compute_cost(Z3,Y)

    

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)



    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        

        sess.run(init)

        for epoch in range(num_epochs):



            epoch_cost = 0.                       # Defines a cost related to an epoch

            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set

            seed = seed + 1

            minibatches = random_mini_batches(X_train, y_train, minibatch_size, seed)



            for minibatch in minibatches:



                (minibatch_X, minibatch_Y) = minibatch

                _ , minibatch_cost = sess.run([optimizer,cost],feed_dict={X:minibatch_X, Y:minibatch_Y})

                

                epoch_cost += minibatch_cost / num_minibatches

                

            if print_cost == True and epoch % 25 == 0:

                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))

            if print_cost == True and epoch % 5 == 0:

                costs.append(epoch_cost)

                

        plt.plot(np.squeeze(costs))

        plt.ylabel('cost')

        plt.xlabel('iterations (per fives)')

        plt.title("Learning rate =" + str(learning_rate))

        plt.show()

        

        parameters = sess.run(parameters)

        print ("Parameters have been trained!")



        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))



        # Calculate accuracy on the test set

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))



        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: y_train}))

        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: y_test}))

        

        

        return parameters
parameters = model(X_train, y_train, X_test, y_test,num_epochs=100)
import pandas as pd

pd.DataFrame.from_dict(data=parameters, orient='index').to_csv('Trained_parameters.csv',header=True, index_label='id')