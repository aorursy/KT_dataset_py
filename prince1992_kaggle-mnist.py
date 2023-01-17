# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
print(os.listdir("./"))

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/train.csv").as_matrix()


Y_train = train_data[:,0]
X_train = train_data[:,1:]
Y_train = Y_train.reshape([-1,1])
print(X_train.shape,Y_train.shape)
print((len(X_train)))
## Shuffle and Normalization

indices = list(range(len(X_train)))
np.random.shuffle(indices)
X_train = X_train[indices]
Y_train = Y_train[indices]
print("Shuffle done")


## Spliting data into train and test set
n_train = int(0.7*len(Y_train))

x_train,y_train,x_test,y_test = X_train[:n_train,:],Y_train[:n_train],X_train[n_train:,:],Y_train[n_train:]

x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

print("Length of training set: ",len(x_train))
## Normalization
x_train-=127
x_train/=127

x_test-=127
x_test/=127

print("normalization done")
## Building Tensorflow graph

def get_next_batch(batch_size,X,Y):
    for i in range(0,len(Y),batch_size):
        yield X[i:i+batch_size,:],Y[i:i+batch_size]
input_ = tf.placeholder(dtype=tf.float32, shape=(None,X_train.shape[1]))
label = tf.placeholder(dtype=tf.int32, shape=(None,1))
learning_rate = tf.placeholder(dtype=tf.float32)
keep_prob = tf.placeholder(dtype=tf.float32)
def neural_network():
    input_2d = tf.reshape(input_,[-1,28,28,1])
    Layer_1 = tf.layers.conv2d(input_2d,8,5,1,activation=tf.nn.relu)
    Layer_2 = tf.layers.max_pooling2d(Layer_1,2,1)
    Layer_3 = tf.layers.conv2d(Layer_2,8,3,1,activation=tf.nn.relu)
    Layer_4 = tf.layers.max_pooling2d(Layer_3,2,1)
    Layer_5 = tf.layers.flatten(Layer_4)
    Layer_6 = tf.layers.dense(Layer_5, 32, activation=tf.nn.relu)
    out = tf.layers.dense(Layer_6,10)
    
    #Define cost
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=tf.one_hot(label,depth=10)))
    
    ## Apply optimizer
    optimzer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    return cost, optimzer, out
cost, optimizer, output = neural_network()
epochs = 150
batch_size = 512
num_batches = int(x_train.shape[0]/batch_size)
current_placeholder = 0.001
train_accuracy = []
val_accuracy = []
test_accuracy = []
cost_curr = []
shuffle_indices = int(0.7*len(X_train))
y_predicted_final = []
def eval_cost(X, y):
    total_cost = 0
    nb_batches = 0
    for X,y in get_next_batch(256,X,y):
        feed_dict={input_: X, label: y, learning_rate:current_placeholder, keep_prob:1.0}
        total_cost += cost.eval(feed_dict=feed_dict)
        nb_batches += 1
    return total_cost / nb_batches

def eval_accuracy(X,y):
    nb_batches = 0
    total_acc = 0
    for X,y in get_next_batch(256,X,y):
        feed_dict={input_: X, label: y, learning_rate:current_placeholder, keep_prob:1.0}
        y_predicted = np.argmax(output.eval(feed_dict=feed_dict),1)
        total_acc += accuracy_score(y,y_predicted)
        nb_batches += 1
    return total_acc/nb_batches

def eval_test(X):
    feed_dict={input_: X, learning_rate:current_placeholder, keep_prob:1.0}
    y_predicted = np.argmax(output.eval(feed_dict=feed_dict),1)
    y_predicted_final = y_predicted
    return y_predicted

test_data = pd.read_csv("../input/test.csv").as_matrix()
print(test_data.shape)
test_data = test_data.astype(np.float32)
test_data-=127
test_data/=127
## Launching the graph

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    current_placeholder = 0.001
    
    for epoch in tqdm(range(epochs)):
        indices = list(range(len(X_train)))
        np.random.shuffle(indices)
        X_train = X_train[indices]
        Y_train = Y_train[indices]
        x_train,y_train,x_test,y_test = X_train[:shuffle_indices, :],Y_train[:shuffle_indices],X_train[shuffle_indices:,:],Y_train[shuffle_indices:]
        
        for x,y in get_next_batch(batch_size, x_train,y_train):
            sess.run(optimizer,feed_dict={input_:x, label:y, learning_rate:current_placeholder, keep_prob:0.70})
            
        if (epoch+1) % 1 == 0:
            # Find training cost.
            c = eval_cost(x_train, y_train)
            cost_curr.append(c)
            # Find train accuracy
            current_train_acc = eval_accuracy(x_train,y_train)
            train_accuracy.append(current_train_acc)
            # Find test accuracy
            test_acc = eval_accuracy(x_test, y_test)
            test_accuracy.append(test_acc)
            
            
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))
            print("Train Accuracy:", current_train_acc)
            #print("Validation Accuracy:", current_val_acc)
            print("Test Accuracy:",test_acc)
            print()
    print("Optimization Finished!")
    # Find test accuracy
    print("Test Accuracy:",eval_accuracy(x_test, y_test) )
    
     # Find test output
    submission = []
    submission = eval_test(test_data)
    print("Test Accuracy output :",submission)
    saver = tf.train.Saver()
    saver.save(sess, './')
    print("Model saved")
    
        
print("Predicted Value: ",set(submission))
test_data = pd.read_csv("../input/test.csv")
test_data.index
Submission = pd.DataFrame()
Submission['ImageId'] = test_data.index+1
Submission['Label'] = submission

Submission.head()
Submission.to_csv('submission.csv', index=False)
