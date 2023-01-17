#Includes L2 and dropout regularization.
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf #For tenserflow neuralnetwork
import sklearn.preprocessing #to make one hot encoding matrix

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")

#print("Sample Train data\n",train_data.head())
#print("Sample Test data\n", test_data.head())
print("Train shape",train_data.shape)
print("Test shape", test_data.shape)

train_data  = train_data.sample(train_data.shape[0],random_state=1).reset_index(drop=True)
#print("Sample Train data\n",train_data.head())
print("Train shape",train_data.shape)

nrow = 38500
train_train = train_data.iloc[0:nrow,]
train_val = train_data.iloc[(nrow):,]
print("Train Train shape",train_train.shape)
print("Train validation shape",train_val.shape)
train_Y = train_train.iloc[:,0]
train_X = train_train.iloc[:,1:]

val_Y = train_val.iloc[:,0]
val_X = train_val.iloc[:,1:]

#print(train_Y.shape, train_X.shape,val_Y.shape, val_X.shape)

train_Y = np.array(train_Y).reshape(train_Y.shape[0],1)
val_Y = np.array(val_Y).reshape(val_Y.shape[0],1)
train_X = np.array(train_X)
val_X = np.array(val_X)
test_X = np.array(test_data)

#print(train_Y.shape, train_X.shape,val_Y.shape, val_X.shape)

train_X = np.transpose(train_X)
train_X = train_X.astype(np.float32)
train_Y = np.transpose(train_Y)

val_X = np.transpose(val_X)
val_X = val_X.astype(np.float32)
val_Y = np.transpose(val_Y)

test_X = np.transpose(test_X)
test_X = test_X.astype(np.float32)

print("train_y_shape:",train_Y.shape, "train_x_shape:",train_X.shape,"val_y_shape:",val_Y.shape, "val_x_shape:",val_X.shape,
     "Test_X_shape",test_X.shape)
#del train_data, test_data
#Create placeholders for X and Y
def create_placeholder (n_x, n_y):
    X = tf.placeholder(tf.float32, shape=[n_x,None])
    Y = tf.placeholder(tf.float32, shape=[n_y,None])
    return X,Y
print("Done...")
def initialize_parameters (n_x, n_y, n_layer1, n_layer2):
    W1 = tf.get_variable("W1",[n_layer1,n_x],initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1",[n_layer1,1],initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2",[n_layer2,n_layer1],initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2",[n_layer2,1],initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3",[n_y,n_layer2],initializer = tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3",[n_y,1],initializer = tf.zeros_initializer())
    
    parameters = {"W1": W1,"b1": b1,"W2": W2,"b2": b2,"W3": W3,"b3": b3}
    return parameters
print("Done...")
def forward_prop(X, parameters,keep_prob=0.9, dropout=False):
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    b1 = parameters['b1']
    b2 = parameters['b2']
    b3 = parameters['b3']
    
    Z1 = tf.add(tf.matmul(W1,X),b1, name="Z1")
    A1 = tf.nn.tanh(Z1, name="A1")
    if dropout==True:
        A1 = tf.nn.dropout(A1, keep_prob)
    Z2 = tf.add(tf.matmul(W2,A1),b2,name="Z2")
    A2 = tf.nn.tanh(Z2,name="A2")
    if dropout==True:
        A2 = tf.nn.dropout(A2, keep_prob)
    Z3 = tf.add(tf.matmul(W3,A2),b3,name="Z3")
    
    return Z3
print("Done...")
def get_cost (Z3, Y, regularizer, reg_param=0.001, l2=False):
    prediction = tf.transpose(Z3)
    actual = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=actual))
    if l2==True:
        cost = tf.reduce_mean(cost+reg_param * regularizer)
    return cost
print("Done...")
def get_regularizer(parameters):
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    regularizer = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3)
    return regularizer
print('Done...')
#https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-1-hot-encoded-numpy-array
def convertToOneHot(vector, num_classes=None):
    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0

    if num_classes is None:
        num_classes = np.max(vector)+1
    else:
        assert num_classes > 0
        assert num_classes >= np.max(vector)

    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1
    return result.astype(int)
print("Done...")
n_x = train_X.shape[0]
m = train_X.shape[1]
n_y = np.unique(train_Y).shape[0] #Total softmax unit entries
n_layer1 = 30
n_layer2 = 25
learning_rate = 0.001
minibatch_size = 1024
n_epoch = 200
reg_coef = 0.002
keep_prob = 0.9

#Create one hot encoding for Y
Y_train_oh = np.transpose(convertToOneHot(np.transpose(train_Y).reshape(train_Y.shape[1]),10))
#print(Y_train_oh.shape)
Y_val_oh = np.transpose(convertToOneHot(np.transpose(val_Y).reshape(val_Y.shape[1]),10))
#print(Y_val_oh.shape)

tf.reset_default_graph()
X,Y = create_placeholder(n_x,n_y)
param = initialize_parameters(n_x, n_y, n_layer1, n_layer2)
#Set the last parameter to true to build NN with dropout regularization.
Z3 =  forward_prop(X, param, keep_prob, True)
regularizer = get_regularizer(param)
#Set the last parameter to true to build NN with L2 regularization.
f_cost = get_cost(Z3, Y, regularizer, reg_coef,False)

#Backward propogation
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(f_cost)
print('Done...')
#Start running tensorflow
init = tf.global_variables_initializer()
seed = 3

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epoch):
        epoch_cost = 0
        num_minibatches = int(m / minibatch_size)
        
        for nb in range(num_minibatches):
            seed = seed+1
            randcol = np.random.randint(0,m,minibatch_size)
            minibatch_x = train_X[:,randcol]
            minibatch_y = Y_train_oh[:,randcol]
            _ , minibatch_cost = sess.run([optimizer, f_cost], feed_dict={X: minibatch_x, Y: minibatch_y})
            epoch_cost += minibatch_cost / num_minibatches
        
        # Print the cost every epoch
        if epoch % 25 == 0:
            print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
        
    parameters = sess.run(param)
    correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        
    print ("Train Accuracy:", accuracy.eval({X: train_X, Y: Y_train_oh}))
    print ("Validation Accuracy:", accuracy.eval({X: val_X, Y: Y_val_oh}))
with tf.Session() as sess:
    sess.run(init)
    predictions_test = tf.argmax(forward_prop(test_X,parameters))
    predictions = sess.run(predictions_test)
    print(predictions.shape)
    print(predictions[1:50])
sample_submission = pd.read_csv("../input/sample_submission.csv")
print(sample_submission.head())
sample_submission.Label = predictions
print(sample_submission.head())
sample_submission.to_csv('submission.csv',sep=",",index=False)