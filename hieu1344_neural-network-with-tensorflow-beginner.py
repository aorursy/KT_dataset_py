# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import tensorflow as tf
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

print(train_df.shape, test_df.shape)
train_df.head()
X_train = train_df.iloc[:, 1:].values.T

y_train = train_df.iloc[:, 0].values

X_test = test_df.values.T
index = 5

plt.imshow(X_train[:, index].reshape(28, 28))

print(y_train[index])
y_train[:10]
#convert labels to one hot vector

def to_onehot(labels):

    ones = tf.one_hot(labels, 10, axis=0)

    with tf.Session() as sess:

        ones = sess.run(ones)

    return ones





y_train = to_onehot(y_train)

y_train[:, :10]
X_train = X_train / 255

X_test = X_test / 255
def initialize_parameters():

    tf.set_random_seed(1)       

    

    W1 = tf.get_variable('W1', shape=[25, 784], initializer=tf.contrib.layers.xavier_initializer(seed=1),

                          regularizer=tf.contrib.layers.l2_regularizer(0.8))

    b1 = tf.get_variable('b1', shape=[25, 1], initializer=tf.contrib.layers.xavier_initializer(seed=1))

    W2 = tf.get_variable('W2', shape=[15, 25], initializer=tf.contrib.layers.xavier_initializer(seed=1),

                         regularizer=tf.contrib.layers.l2_regularizer(0.8))

    b2 = tf.get_variable('b2', shape=[15, 1], initializer=tf.contrib.layers.xavier_initializer(seed=1))

    W3 = tf.get_variable('W3', shape=[10, 15], initializer=tf.contrib.layers.xavier_initializer(seed=1),

                         regularizer=tf.contrib.layers.l2_regularizer(0.8))

    b3 = tf.get_variable('b3', shape=[10, 1], initializer=tf.contrib.layers.xavier_initializer(seed=1))



    parameters = {"W1": W1,

                  "b1": b1,

                  "W2": W2,

                  "b2": b2,

                  "W3": W3,

                  "b3": b3}

    

    return parameters

def forward_propagation(X, parameters):

    W1 = parameters['W1']

    b1 = parameters['b1']

    W2 = parameters['W2']

    b2 = parameters['b2']

    W3 = parameters['W3']

    b3 = parameters['b3']



    Z1 = tf.add(tf.matmul(W1, X) , b1)                                          

    A1 = tf.nn.relu(Z1)                                              

    Z2 = tf.add(tf.matmul(W2, A1), b2)                                            

    A2 = tf.nn.relu(Z2)                                              

    Z3 = tf.add(tf.matmul(W3, A2), b3)                          

    

    return Z3
def compute_cost(Z3, Y):

    logits = tf.transpose(Z3)

    labels = tf.transpose(Y)



    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    return cost

def model(X_train, Y_train, learning_rate = 0.0005,

          n_iterations = 2500, print_cost = True):

    

    tf.reset_default_graph()                         

    tf.set_random_seed(1)                             

    seed = 3                                          

    (n_x, m) = X_train.shape                          

    n_y = y_train.shape[0]                            

    costs = []                                        

    

    X = tf.placeholder(tf.float32, shape=(n_x, None), name='X')

    y = tf.placeholder(tf.float32, shape=(n_y, None), name='y')

    parameters = initialize_parameters()

 

    Z3 = forward_propagation(X, parameters)



    cost = compute_cost(Z3, y)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)



    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)

        

        # Do the training loop

        for i in range(n_iterations):

            _ , c = sess.run([optimizer, cost], feed_dict={X:X_train, y:y_train})

            

            # Print the cost every epoch

            if print_cost == True and i % 100 == 0:

                print ("Cost after %i iterations: %f" % (i, c))

                costs.append(c)

            

                

        # plot the cost

        plt.plot(np.squeeze(costs))

        plt.ylabel('cost')

        plt.xlabel('iterations')

        plt.title("Learning rate =" + str(learning_rate))

        plt.show()



        # lets save the parameters in a variable

        parameters = sess.run(parameters)

        print ("Parameters have been trained!")



        # Calculate the correct predictions

        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(y))



        # Calculate accuracy on the test set

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))



        print ("Train Accuracy:", accuracy.eval({X: X_train, y: y_train}))

        return parameters
parameters = model(X_train, y_train)
parameters
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

    

    z3 = forward_propagation(x, params)

    p = tf.argmax(z3)

    

    with tf.Session() as sess:

        prediction = sess.run(p, feed_dict = {x: X})

        

    return prediction
p = predict(X_test, parameters)
submission = pd.DataFrame({'ImageId':np.arange(1, X_test.shape[1]+1), 

                          'Label': p})
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "forest.csv"):  

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



# create a random sample dataframe

# create a link to download the dataframe

create_download_link(submission)



# ↓ ↓ ↓  Yay, download link! ↓ ↓ ↓ 