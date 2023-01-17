# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import math

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from pandas import Series

import matplotlib.pyplot as plt

from tensorflow.python.framework import ops

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
def dataProcess(data):

    data1 = data

    del data1['Name']

    del data1['Ticket']

    del data1['Cabin']

    Sex = data1['Sex']

    Age = data1['Age']

    Embarked = data1['Embarked']

    Sex = Sex.replace(['female','male'],[1,0])

    Age = Age.replace(np.nan,0)

    Embarked = Embarked.replace([np.nan,'S','C','Q'],[0,1,2,3])

    data1['Sex'] = Sex

    data1['Age'] = Age

    data1['Embarked'] = Embarked

    return data1
data1  = pd.read_csv('../input/test.csv',header = 0, index_col = 'PassengerId')

data2  = pd.read_csv('../input/train.csv',header = 0, index_col = 'PassengerId')

Survived = data2.pop('Survived')

_,n = data2.shape

data2.insert(n,'Survived',Survived)
data1  = dataProcess(data1)

data2 = dataProcess(data2)
train = data2.values

test = data1.values
np.random.seed(1)

random_key = np.random.permutation(891)

X_train= train[random_key[0:700],0:7]

Y_train = train[random_key[0:700],7].reshape(700,1)

X_valid= train[random_key[700:800],0:7]

Y_valid= train[random_key[700:800],7].reshape(100,1)

X_test = train[random_key[800:],0:7]

Y_test = train[random_key[800:],7].reshape(91,1)
print("X_train"+str(X_train.shape))

print("Y_train"+str(Y_train.shape))

print("X_valid"+str(X_valid.shape))

print("Y_valid"+str(Y_valid.shape))

print("X_test"+str(X_test.shape))

print("Y_test"+str(Y_test.shape))

def X_normal(X):

    mean1 = np.mean(X[:,2])

    mean2 = np.mean(X[:,5])

    std1 = np.std(X[:,2])

    std2 = np.std(X[:,5])

    X[:,2] = (X[:,2]- mean1)/std1

    X[:,5] = (X[:,5]- mean2)/std2

    return X
X_train = X_normal(X_train)

X_valid = X_normal(X_valid)

X_test = X_normal(X_test)
def y_process(y):

    m,_ = y.shape

    Y = np.zeros((m,2))

    one= np.argwhere(y==1)

    zero = np.argwhere(y==0)

    Y[zero[:,0],0] = 1

    Y[one[:,0],1] = 1

    return Y
Y_trainP = y_process(Y_train)

Y_validP = y_process(Y_valid)

Y_testP = y_process(Y_test)
def initial_parameters():

    tf.set_random_seed(1234)

    W1 = tf.get_variable("W1",[32,7],initializer = tf.contrib.layers.xavier_initializer())

    W2 = tf.get_variable("W2",[32,32],initializer = tf.contrib.layers.xavier_initializer())

    W3 = tf.get_variable("W3",[64,32],initializer = tf.contrib.layers.xavier_initializer())

    W4 = tf.get_variable("W4",[64,64],initializer = tf.contrib.layers.xavier_initializer())

    W5 = tf.get_variable("W5",[128,64],initializer = tf.contrib.layers.xavier_initializer())

    W6 = tf.get_variable("W6",[128,128],initializer = tf.contrib.layers.xavier_initializer())

    W7 = tf.get_variable("W7",[64,128],initializer = tf.contrib.layers.xavier_initializer())

    W8 = tf.get_variable("W8",[64,64],initializer = tf.contrib.layers.xavier_initializer())

    W9 = tf.get_variable("W9",[32,64],initializer = tf.contrib.layers.xavier_initializer())

    W10 = tf.get_variable("W10",[32,32],initializer = tf.contrib.layers.xavier_initializer())

    b1 = tf.get_variable("b1",[32,1],initializer = tf.initializers.zeros())

    b2 = tf.get_variable("b2",[32,1],initializer = tf.initializers.zeros())

    b3 = tf.get_variable("b3",[64,1],initializer = tf.initializers.zeros())

    b4 = tf.get_variable("b4",[64,1],initializer = tf.initializers.zeros())

    b5 = tf.get_variable("b5",[128,1],initializer = tf.initializers.zeros())

    b6 = tf.get_variable("b6",[128,1],initializer = tf.initializers.zeros())

    b7 = tf.get_variable("b7",[64,1],initializer = tf.initializers.zeros())

    b8 = tf.get_variable("b8",[64,1],initializer = tf.initializers.zeros())

    b9 = tf.get_variable("b9",[32,1],initializer = tf.initializers.zeros())

    b10 = tf.get_variable("b10",[32,1],initializer = tf.initializers.zeros())

    parameters = {"W1":W1,"W2":W2,"W3":W3,"W4":W4,"W5":W5,"W6":W6,"W7":W7,"W8":W8,"W9":W9,"W10":W10,

                  "b1":b1,"b2":b2,"b3":b3,"b4":b4,"b5":b5,"b6":b6,"b7":b7,"b8":b8,"b9":b9,"b10":b10}

    return parameters
def forward_propagation(X,parameters,keep_prob):

    W1 = parameters["W1"]

    W2 = parameters["W2"]

    W3 = parameters["W3"]

    W4 = parameters["W4"]

    W5 = parameters["W5"]

    W6 = parameters["W6"]

    W7 = parameters["W7"]

    W8 = parameters["W8"]

    W9 = parameters["W9"]

    W10 = parameters["W10"]

    b1 = parameters["b1"]

    b2 = parameters["b2"]

    b3 = parameters["b3"]

    b4 = parameters["b4"]

    b5 = parameters["b5"]

    b6 = parameters["b6"]

    b7 = parameters["b7"]

    b8 = parameters["b8"]

    b9 = parameters["b9"]

    b10 = parameters["b10"]

    Z1 = tf.add(tf.matmul(W1,X,transpose_b = True),b1)

    A1 = tf.nn.dropout(tf.nn.relu(Z1), keep_prob)

    Z2 = tf.add(tf.matmul(W2,A1),b2)

    A2 = tf.nn.dropout(tf.nn.relu(Z2), keep_prob)

    Z3 = tf.add(tf.matmul(W3,A2),b3)

    A3 = tf.nn.dropout(tf.nn.relu(Z3), keep_prob)

    Z4 = tf.add(tf.matmul(W4,A3),b4)

    A4 = tf.nn.dropout(tf.nn.relu(Z4), keep_prob)

    Z5 = (tf.add(tf.matmul(W5,A4),b5))

    A5 = tf.nn.dropout(tf.nn.relu(Z5), keep_prob)

    Z6 = (tf.add(tf.matmul(W6,A5),b6))

    A6 = tf.nn.dropout(tf.nn.relu(Z6), keep_prob)

    Z7 = (tf.add(tf.matmul(W7,A6),b7))

    A7 = tf.nn.dropout(tf.nn.relu(Z7), keep_prob)

    Z8 = (tf.add(tf.matmul(W8,A7),b8))

    A8 = tf.nn.dropout(tf.nn.relu(Z8), keep_prob)

    Z9 = (tf.add(tf.matmul(W9,A8),b9))

    A9 = tf.nn.dropout(tf.nn.relu(Z9), keep_prob)

    Z10 = (tf.add(tf.matmul(W10,A9),b10))

    A10 = tf.nn.dropout(tf.transpose(tf.nn.relu(Z10)), keep_prob)

    Z11 = tf.contrib.layers.fully_connected(A10,2,activation_fn=None)

    return Z11
def compute_cost(Z,Y):

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = Z, labels = Y))

    return cost
def model(X_train,Y_train,X_valid,Y_valid,X_test,Y_test,learning_rate=0.009 ,

          num_epochs =500,minibatch_size = 64,print_cost = True):

    ops.reset_default_graph()    

    m,n = X_train.shape

    costs = []

    X = tf.placeholder(tf.float32,shape = (None,7))

    Y = tf.placeholder(tf.float32,shape = (None,2))

    parameters = initial_parameters()

    Z = forward_propagation(X,parameters,0.5)

    cost = compute_cost(Z,Y)

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:

            sess.run(init)

            for epoch in range(num_epochs):

                minibatch_cost = 0.

                num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set

                minibatches = random_mini_batches(X_train, Y_train, minibatch_size)

                

                for minibatch in minibatches:



                # Select a minibatch

                    (minibatch_X, minibatch_Y) = minibatch

                # IMPORTANT: The line that runs the graph on a minibatch.

                # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).

                ### START CODE HERE ### (1 line)

                    _ , temp_cost = sess.run([optimizer,cost],feed_dict = {X:minibatch_X,Y:minibatch_Y})

                ### END CODE HERE ###

                

                    minibatch_cost += temp_cost/num_minibatches

                

                

                #Print the cost every epoch

                if print_cost == True and epoch % 5 == 0:

                    print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))

                if print_cost == True and epoch % 1 == 0:

                    costs.append(minibatch_cost)

       # plot the cost

            plt.plot(np.squeeze(costs))

            plt.ylabel('cost')

            plt.xlabel('iterations (per tens)')

            plt.title("Learning rate =" + str(learning_rate))

            plt.show()

                

            predict_op = tf.argmax(Z, 1)

            correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        #test the Y_test and predict outputs

            

        # Calculate accuracy on the test set

            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            train_accuracy = accuracy.eval({X: X_train, Y: Y_train})

            valid_accuracy = accuracy.eval({X: X_valid, Y: Y_valid})

            test_accuracy = accuracy.eval({X: X_test, Y: Y_test})

            print("Train Accuracy:", train_accuracy)

            print("valid Accuracy:", valid_accuracy)

            print("test Accuracy:",test_accuracy)

    return train_accuracy,valid_accuracy,test_accuracy,parameters
def random_mini_batches(X, Y, mini_batch_size = 64):

    """

    Creates a list of random minibatches from (X, Y)

    

    Arguments:

    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)

    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)

    mini_batch_size - size of the mini-batches, integer

    

    Returns:

    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)

    """

    

    m = X.shape[0]                  # number of training examples

    mini_batches = []

    

    # Step 1: Shuffle (X, Y)

    permutation = list(np.random.permutation(m))

    shuffled_X = X[permutation,:]

    shuffled_Y = Y[permutation,:]



    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.

    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning

    for k in range(0, num_complete_minibatches):

        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]

        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]

        mini_batch = (mini_batch_X, mini_batch_Y)

        mini_batches.append(mini_batch)

    

    # Handling the end case (last mini-batch < mini_batch_size)

    if m % mini_batch_size != 0:

        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:]

        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]

        mini_batch = (mini_batch_X, mini_batch_Y)

        mini_batches.append(mini_batch)

    

    return mini_batches



learning_rate = [0.0001,0.0003,0.0009,0.001,0.003,0.009,0.01,0.03,0.09]

minibatch_size = [32,64,128]

max_accuracy = 0.0

max_batch= 0

max_rate = 0

accuracy = {}

accuracy_train = []

accuracy_valid = []

accuracy_test = []

parameters_train = []

for i in range(len(learning_rate)):

    for j in range(len(minibatch_size)):

        print("learning_rate: %f; minibatch_size: %f"%(learning_rate[i],minibatch_size[j]))

        train_accuracy,valid_accuracy,test_accuracy,parameters= model(X_train,Y_trainP,X_valid,Y_validP,X_test,Y_testP,

                                                        learning_rate = learning_rate[i],

                                                        minibatch_size = minibatch_size[j])

        accuracy_train.append(train_accuracy)

        accuracy_valid.append(valid_accuracy)

        accuracy_test.append(test_accuracy)

        parameters_train.append(parameters)

        if(valid_accuracy>max_accuracy):

            max_accuracy = valid_accuracy

            accuracy = {'train':train_accuracy,'valid':valid_accuracy,'test':test_accuracy}

            max_batch = j

            max_rate = i



                
print(max_accuracy)

print(accuracy)

print(max_batch)

print(max_rate)
train_accuracy,valid_accuracy,test_accuracy,parameters= model(X_train,Y_trainP,X_valid,Y_validP,X_test,Y_testP,

                                                        learning_rate = 0.003,

                                                        minibatch_size = 32)