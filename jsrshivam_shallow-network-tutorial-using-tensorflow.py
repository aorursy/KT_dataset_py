import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))

from sklearn.datasets import make_classification







X, y = make_classification(n_samples=50000,n_features=8,



                           n_informative=8,n_redundant=0,



                           n_clusters_per_class=2, random_state = 26)
print(type(X), type(y))

print(len(X[1]), X[1], y[1])

print("Shape : {}, {}".format(X.shape, y.shape))
import matplotlib

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap



colors=['black','yellow']



cmap = matplotlib.colors.ListedColormap(colors)



#Plot the figure



plt.figure()



plt.title('Non-linearly separable classes')



plt.scatter(X[:,0], X[:,1], c=y,



            marker= '*', s=50, cmap=cmap, alpha = 0.5 )



plt.savefig('fig1.png', bbox_inches='tight')
from sklearn.model_selection import train_test_split



X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.25,random_state = 25)



print("shape of X_train:{} shape 0f Y_train:{}".format(X_train.shape, Y_train.shape))  

print("shape of X_test:{} shape 0f Y_test:{}".format(X_test.shape, Y_test.shape))  
X_train = X_train.T



Y_train = Y_train.reshape(1, len(Y_train))



X_test = X_test.T



Y_test = Y_test.reshape(1, len(Y_test))



print("shape of X_train:{} shape 0f Y_train:{} after transformation".format(X_train.shape, Y_train.shape))

import tensorflow as tf



def placeholders(num_features):



  A_0 = tf.placeholder(dtype = tf.float64, shape = ([num_features,None]))



  Y = tf.placeholder(dtype = tf.float64, shape = ([1,None]))



  return A_0,Y
def initialiseParameters(num_features, num_nodes):



  W1 = tf.Variable(initial_value=tf.random_normal([num_nodes,num_features], dtype = tf.float64) * 0.01)



  b1 = tf.Variable(initial_value=tf.zeros([num_nodes,1], dtype=tf.float64))



  W2 = tf.Variable(initial_value=tf.random_normal([1,num_nodes], dtype=tf.float64) * 0.01)



  b2 = tf.Variable(initial_value=tf.zeros([1,1], dtype=tf.float64))



  return {"W1":W1,"b1":b1,"W2":W2,"b2":b2}  
def forward_propagation(A_0,parameters):



  Z1 = tf.matmul(parameters["W1"],A_0) + parameters["b1"]



  A1 = tf.nn.relu(Z1)



  Z2 = tf.matmul(parameters["W2"],A1) + parameters["b2"]



  return Z2
def shallow_model(X_train,Y_train,X_test,Y_test, num_nodes, learning_rate, num_iter):



  num_features = X_train.shape[0]



  A_0, Y = placeholders(num_features)



  parameters = initialiseParameters(num_features, num_nodes)



  Z2 = forward_propagation(A_0, parameters)



  cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Z2,labels=Y))



  train_net = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)



  init = tf.global_variables_initializer()



  



  with tf.Session() as sess:



    sess.run(init)



    for i in range(num_iter):



      _,c = sess.run([train_net, cost], feed_dict={A_0: X_train, Y: Y_train})



      if i % 1000 == 0:



        print(c)



    correct_prediction = tf.equal(tf.round(tf.sigmoid(Z2)), Y)



    # Calculate accuracy



    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))



    print("Accuracy on test set:", accuracy.eval({A_0: X_test, Y: Y_test}))
shallow_model(X_train, Y_train, X_test, Y_test, 8, 0.2, 10000)
