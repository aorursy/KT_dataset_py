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
train_data = pd.read_csv("../input/sign_mnist_train.csv")
test_data =  pd.read_csv("../input/sign_mnist_test.csv")
train_data.shape
train_data.head()
labels = train_data["label"].values
label_class = np.unique(np.array(labels))
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import math
plt.figure(figsize =(18,8))
sns.countplot(x=labels)
train_data.drop("label" , axis=1 , inplace=True)
one_hot = pd.get_dummies(labels)
one_hot = np.array(one_hot)
one_hot.shape
image_data = train_data.values
image_data
#print(image_data[0].reshape(28,28))
plt.imshow(image_data[0].reshape(28,28))
images = np.array([image_data[i].reshape(28,28) for i in range(image_data.shape[0])])
flatten_images = np.array([i.flatten() for i in images])
def create_placeholders(n_x, n_y):
    
    X = tf.placeholder(tf.float32, shape= [n_x , None] , name="X")
    Y = tf.placeholder(tf.float32, shape= [n_y , None] , name="Y")
    
    return X,Y
def initialise_parameters():
    
    W1 = tf.get_variable("W1", [25,784], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", [25,1], initializer = tf.zeros_initializer()) 
    W2 = tf.get_variable("W2", [12,25], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2", [12,1], initializer = tf.zeros_initializer()) 
    W3 = tf.get_variable("W3", [24,12], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable("b3", [24,1], initializer = tf.zeros_initializer()) 
    
    parameters = {
                    "W1":W1,
                    "W2":W2,
                    "W3":W3,
                    "b1":b1,
                    "b2":b2,
                    "b3":b3
    }
    
    return parameters    
def forward_propogation(X,parameters):
    #getting all the parameters 
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]
    b3 = parameters["b3"]
    
    Z1 = tf.add(tf.matmul(W1,X),b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2,A1),b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3,A2),b3)   #It is important to note that the forward propagation stops at z3. 
                                        #The reason is that in tensorflow the last linear layer output is given as input to the function computing the loss. 
                                        #Therefore, you don't need a3
    return Z3
    
def compute_cost(Z3,Y):
    
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits = logits , lables=labels))
    return cost
def BackPropogation(cost, learning_rate):
    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(cost)
    return optimizer
def mini_batches(X,Y, minibatch_size):
    
    m = X.shape[0]
    minibatch = []
    
    permutation = list(np.random.permutation(m))
    
    #shuffling the data
    shuffled_X = X[permutation,:]
    shuffled_Y = Y[permutation,:]
    
    number_of_batches = math.floor(m/minibatch_size)
    
    for i in range(0, number_of_batches):
        
        minibatch_X = shuffled_X[ i*minibatch_size: (i+1)*minibatch_size , :]
        minibatch_Y = shuffled_Y[ i*minibatch_size: (i+1)*minibatch_size , :]
        
        minibatch_tuple = (minibatch_X , minibatch_Y)
        minibatch.append(minibatch_tuple)
        
    # handling last batch of the set
    
    if m% minibatch_size !=0:
        
        minibatch_X = shuffled_X[:(m-(number_of_batches*minibatch_size)) , :]
        minibatch_Y = shuffled_Y[:(m-(number_of_batches*minibatch_size)) , :]
        
        minibatch_tuple = (minibatch_X , minibatch_Y)
        minibatch.append(minibatch_tuple)
        
        
    return minibatch
