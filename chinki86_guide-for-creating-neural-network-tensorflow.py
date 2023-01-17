# This Python 3 environment comes with many helpful anal,ytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# Any results you write to the current directory are saved as output.




# Read training and test data files
train = pd.read_csv("../input/fashion-mnist_train.csv").values
test  = pd.read_csv("../input/fashion-mnist_test.csv").values




# Reshape and normalize training data
trainX = train[:, 1:].reshape(train.shape[0],1,28, 28).astype( 'float32' )
X_train = trainX / 255.0

y_train = train[:,0]


# Reshape and normalize test data
testX = test[:,1:].reshape(test.shape[0],1, 28, 28).astype( 'float32' )
X_test = testX / 255.0

y_test = test[:,0]
X_train.shape,X_test.shape,y_train.shape,y_test.shape

X_train = X_train.reshape(X_train.shape[0],-1).T

X_test = X_test.reshape(X_test.shape[0],-1).T
def one_hot_encoding(Y,C):
    Y = np.eye(C)[Y.reshape(-1)].T
    
    return Y

y_train = one_hot_encoding(y_train,10)
y_train.shape
y_test = one_hot_encoding(y_test,10)

X_train.shape,X_test.shape,y_train.shape,y_test.shape
def create_placeholder(n_x,n_y):
    X = tf.placeholder(tf.float32,[n_x,None],name="X")
    Y = tf.placeholder(tf.float32,[n_y,None],name="Y")
    return X,Y

def initialize_parameters():
    tf.set_random_seed(1)
    
    W1 = tf.get_variable("W1",[100,784],initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1",[100,1],initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2",[200,100],initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2",[200,1],initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3",[10,200],initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3",[10,1],initializer=tf.zeros_initializer())
    
    parameters = {"W1":W1,"b1":b1,"W2":W2,"b2":b2,"W3":W3,"b3":b3}
    
    return parameters

def compute_cost(Z3,Y):
    
    
    logits = tf.transpose(Z3)
    
    labels = tf.transpose(Y)
    
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=labels))
    
    return cost


def forward_prop(X,parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    Z1 = tf.add(tf.matmul(W1, X),b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1),b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2),b3)
    
    return Z3


def model(X_train,Y_train,X_test,Y_test,num_epochs=1500,learning_rate=0.0001,print_cost=True):
    tf.set_random_seed(1)
    seed=3
    (n_x,m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []
    tf.reset_default_graph()
    
    X,Y = create_placeholder(n_x,n_y)
    parameters = initialize_parameters()
    Z3 = forward_prop(X,parameters)
    cost = compute_cost(Z3,Y)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as session:
        session.run(init)
        
        for epoch in range(num_epochs):
            epoch_cost = 0
            seed = seed + 1
            _ , new_cost = session.run([optimizer,cost],feed_dict={X:X_train,Y:Y_train})
            
            epoch_cost += new_cost
            
            if print_cost == True and epoch % 100 == 0:
                print("Cost after epoch %i %f" % (epoch,epoch_cost))
            if print_cost == True and epoch % 100 == 0:
                costs.append(epoch_cost)
                
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
    
        parameters = session.run(parameters)
        print("Parameters have been trained")
    
        correct_prediction = tf.equal(tf.argmax(Z3),tf.argmax(Y))
    
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
    
        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
    
    return parameters


parameters = model(X_train,y_train,X_test,y_test)
