# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running thiimport numpy as np 

import pandas as pd

from pandas import Series, DataFrame 

from tensorflow.python.framework import ops

import tensorflow as tf

import matplotlib.pyplot as plt

import math

import sklearn.metrics as skm

%matplotlib inline

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots

plt.rcParams['image.interpolation'] = 'nearest'

plt.rcParams['image.cmap'] = 'gray'

from sklearn.linear_model import LogisticRegression



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Load the train and test csv files.

train = pd.read_csv('../input/testcsv/train.csv')

test = pd.read_csv('../input/testcsv/test.csv')
# Preprocessing Step taken by Forca

id_test = test['id'].values

target_train = train['target'].values

train = train.drop(['target','id'], axis = 1)

test = test.drop(['id'], axis = 1)

col_to_drop = train.columns[train.columns.str.startswith('ps_calc_')]

train = train.drop(col_to_drop, axis=1)

test = test.drop(col_to_drop, axis=1)  

train = train.replace(-1, np.nan)

test = test.replace(-1, np.nan)

cat_features = [a for a in train.columns if a.endswith('cat')]

for column in cat_features:

    temp = pd.get_dummies(pd.Series(train[column]))

    train = pd.concat([train,temp],axis=1)

    train = train.drop([column],axis=1)

for column in cat_features:

    temp = pd.get_dummies(pd.Series(test[column]))

    test = pd.concat([test,temp],axis=1)

    test = test.drop([column],axis=1)

test=test.fillna(method='pad')

train=train.fillna(method='pad')

print(train.values.shape, test.values.shape)

print(target_train.shape)
#Function to evaluate the Gini Coefficients

def eval_gini(y_true, y_prob):

    y_true = np.asarray(y_true)

    y_true = y_true[np.argsort(y_prob)]

    ntrue = 0

    gini = 0

    delta = 0

    n = len(y_true)

    for i in range(n-1, -1, -1):

        y_i = y_true[i]

        ntrue += y_i

        gini += y_i * delta

        delta += 1 - y_i

    gini = 1 - 2 * gini / (ntrue * (n - ntrue))

    return gini





#Applying vanila logistic regression.

from sklearn.cross_validation import train_test_split

x_train ,x_test, y_train, y_test = train_test_split(train, target_train,test_size=0.3)



log_model=LogisticRegression()

log_model.fit(x_train,y_train)

pred_log = log_model.predict_proba(x_test)[:,1]

print( "Gini log  Test= ", eval_gini(y_test, pred_log))
#Mini Batches Generation for TensorFlow. Lots of code ahead, don't worry, its just helper functions.

def random_mini_batches(X, Y, mini_batch_size = 1024, seed = 0):

    np.random.seed(seed)            

    m = X.shape[1]                  

    mini_batches = []

    

    permutation = list(np.random.permutation(m))

    shuffled_X = X[:, permutation]

    shuffled_Y = Y[:, permutation]

    

    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning

    for k in range(0, num_complete_minibatches):

        mini_batch_X = shuffled_X[:, k*mini_batch_size : mini_batch_size*(k+1)]

        mini_batch_Y = shuffled_Y[:,mini_batch_size*k : mini_batch_size*(k+1)]

        mini_batch = (mini_batch_X, mini_batch_Y)

        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:

        mini_batch_X = shuffled_X[:, mini_batch_size*num_complete_minibatches : m]

        mini_batch_Y = shuffled_Y[:, mini_batch_size*num_complete_minibatches : m]

        mini_batch = (mini_batch_X, mini_batch_Y)

        mini_batches.append(mini_batch)

    return mini_batches
# creating placeholders for TensorFlow

def create_placeholders(n_x, n_y):

    X = tf.placeholder(dtype="float", shape=(n_x, None), name='X')

    Y = tf.placeholder(dtype="float", shape=(n_y, None), name='Y')

    return X, Y
# initialize_parameters

tf.reset_default_graph()

def initialize_parameters(f1=198, f2=100, f3=50):

    tf.set_random_seed(1)  

    W1 = tf.get_variable("W1", [f2,f1], initializer = tf.contrib.layers.xavier_initializer(seed=1))

    b1 = tf.get_variable('b1', [f2,1], initializer= tf.zeros_initializer())

    W2 = tf.get_variable("W2", [f3,f2], initializer = tf.contrib.layers.xavier_initializer(seed=1))

    b2 = tf.get_variable('b2', [f3,1], initializer= tf.zeros_initializer())

    W3 = tf.get_variable('W3', [f2,f3], initializer=tf.contrib.layers.xavier_initializer(seed=1))

    b3 = tf.get_variable('b3', [f2,1], initializer= tf.zeros_initializer())

    W4 = tf.get_variable('W4', [f1,f2], initializer=tf.contrib.layers.xavier_initializer(seed=1))

    b4 = tf.get_variable('b4', [f1,1], initializer= tf.zeros_initializer())

    parameters = {"W1": W1,

                  "b1": b1,

                  "W2": W2,

                  "b2": b2,

                  "W4": W4,

                  "b4": b4,

                  "W3": W3,

                  "b3": b3}

    

    return parameters
#Forward Prop steps for tensorflow

def forward_propagation(X, parameters):

    W1 = parameters['W1']

    b1 = parameters['b1']

    W2 = parameters['W2']

    b2 = parameters['b2']

    W3 = parameters['W3']

    b3 = parameters['b3']

    W4 = parameters['W4']

    b4 = parameters['b4']

    

    Z1 = tf.add(tf.matmul(W1, X), b1)                                  # Z1 = np.dot(W1, X) + b1

    A1 = tf.nn.tanh(Z1)

    Z2 = tf.add(tf.matmul(W2, A1), b2)                                  # Z1 = np.dot(W1, X) + b1

    A2 = tf.nn.relu(Z2)# A1 = relu(Z1)                                              # A2 = relu(Z2)

    Z3 = tf.add(tf.matmul(W3, A2), b3)                           # Z3 = np.dot(W3,Z2) + b3

    A3=  tf.nn.tanh(Z3)

    Z4 = tf.add(tf.matmul(W4, A3), b4)                           # Z3 = np.dot(W3,Z2) + b3

    A4=  tf.nn.relu(Z4)

    return A4, A2

#Cost computation for tensorflow

def compute_cost(A4, Y, parameters):

    m = Y.shape[1]

    W1 = parameters['W1']

    W2 = parameters['W2']

    W3 = parameters['W3']

    W4 = parameters['W4']

    #Cost is the the difference of output with input.

    cost = tf.reduce_mean(tf.pow(Y - A4, 2))

    return cost
#Final AE Model

def model(X_train, Y_train, f_2, f_3, learning_rate,

          num_epochs , minibatch_size , print_cost):

    ops.reset_default_graph()                         

    tf.set_random_seed(1)                             

    seed = 3                                          

    (n_x, m) = X_train.shape 

    n_y = Y_train.shape[0]                            

    costs = []                                        

    X, Y = create_placeholders(n_x, n_y)

    parameters = initialize_parameters(f2=f_2,f3=f_3)

    A4, A2 = forward_propagation(X, parameters)

    cost = compute_cost(A4, Y, parameters)

    #Tensorflow optimizer

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init) 

        # Do the training loop

        for epoch in range(num_epochs):

            epoch_cost = 0.                      

            num_minibatches = int(m / minibatch_size)

            seed = seed + 1

            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                # Select a minibatch

                (minibatch_X, minibatch_Y) = minibatch

                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                epoch_cost += minibatch_cost / num_minibatches



            # Print the cost every epoch

            if print_cost == True:

                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))

            if print_cost == True:

                costs.append(epoch_cost)

                

        plt.plot(np.squeeze(costs))

        plt.ylabel('cost')

        plt.xlabel('iterations (per tens)')

        plt.title("Learning rate =" + str(learning_rate))

        plt.show()

        parameters = sess.run(parameters)

        return parameters, A2
train=train.as_matrix()

test=test.as_matrix()
#Run the MODEL

parameters, _ = model(train.T,train.T,160,100,minibatch_size=512,num_epochs=10, learning_rate=0.001, print_cost=True)
#Helper functions to calculate the features using parameters



def forward_propagationout(X, parameters):

    # retrieve parameters

    W1 = parameters['W1']

    b1 = parameters['b1']

    W2 = parameters['W2']

    b2 = parameters['b2']

    W3 = parameters['W3']

    b3 = parameters['b3']

    W4 = parameters['W4']

    b4 = parameters['b4']

    

    Z1 = np.dot(W1, X) + b1                                  # Z1 = np.dot(W1, X) + b1

    A1 = np.tanh(Z1)

    Z2 = np.dot(W2, A1) + b2                                   # Z1 = np.dot(W1, X) + b1

    A2 = relu(Z2)                                             # A2 = relu(Z2)

    Z3 = np.dot(W3, A2) + b3                           # Z3 = np.dot(W3,Z2) + b3

    A3=  np.tanh(Z3)

    Z4 = np.dot(W4, A3) + b4                            # Z3 = np.dot(W3,Z2) + b3

    A4=  relu(Z4)

    return A2

def relu(x):

    s = np.maximum(0,x)

    return s



#A2 featues from the middle layer of AE

A2=forward_propagationout(train.T.astype('float32'), parameters)

x_train ,x_test, y_train, y_test = train_test_split(A2.T, target_train,test_size=0.3)

log_model=LogisticRegression()

log_model.fit(x_train,y_train)

pred_log = log_model.predict_proba(x_test)[:,1]

print( "Gini log  Test= ", eval_gini(y_test, pred_log))