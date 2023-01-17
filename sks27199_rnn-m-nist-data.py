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
import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

from IPython.display import clear_output

import pandas as pd

from sklearn.utils import shuffle

from sklearn.metrics import accuracy_score
train = pd.read_csv('../input/mnist_train.csv')

test = pd.read_csv('../input/mnist_test.csv')
train = np.array(train)

test = np.array(test)
train_X = train[:,1:]

train_Y = pd.get_dummies(train[:,0])

print(train_X.shape)

test_X = test[:,1:]

test_Y = pd.get_dummies(test[:,0])

print(test_Y.shape)
train_X = train_X.reshape(-1, 28, 28)

test_X = test_X.reshape(-1, 28, 28)
train_X.shape
tf.reset_default_graph()
# Training Parameters

learning_rate = 0.001

training_iters = 250

batch_size = 128
# Network Parameters

num_input = 28 # MNIST data input (img shape: 28*28)

timesteps = 28 # timesteps

num_hidden = 64 # hidden layer num of features: n_neurons

num_classes = 10 # MNIST total classes (0-9 digits)
# tf Graph input

X = tf.placeholder("float", [None, timesteps, num_input])

Y = tf.placeholder("float", [None, num_classes])
def RNN(x):



    # Prepare data shape to match `rnn` function requirements

    # Current data input shape: (batch_size, timesteps, n_input)

    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)

    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow

    RNN_cell = tf.nn.rnn_cell.BasicRNNCell(num_hidden)



    # Get lstm cell output

    outputs, states = tf.nn.static_rnn(RNN_cell, x, dtype=tf.float32)



#     out = tf.matmul(outputs[-1], weights['out']) + biases['out']



    out = tf.layers.dense(states, num_classes)



    # Softmax activation, using rnn inner loop last output

    

    out = tf.nn.softmax(out)

    

    return out
prediction = RNN(X)
# Define loss and optimizer

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

train_op = optimizer.minimize(loss_op)
# Evaluate model (with test logits, for dropout to be disabled)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



# Initialize the variables (i.e. assign their default value)

init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init) 



for i in range(training_iters):

    for batch in range(len(train_X)//batch_size):

        batch_x = train_X[batch*batch_size:min((batch+1)*batch_size,len(train_X))]

        batch_y = train_Y[batch*batch_size:min((batch+1)*batch_size,len(train_Y))]    



        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

            # Calculate batch loss and accuracy

        loss = sess.run([loss_op], feed_dict={X: batch_x, Y: batch_y})

            

#     acc = sess.run([accuracy], feed_dict={X: batch_x, Y: batch_y})

    predTest = sess.run(prediction , feed_dict={X:test_X})



    p = np.argmax(predTest,1)

    t = np.argmax(np.array(test_Y),1)



    acc = accuracy_score(p,t)

    print("Iter "+str(i)+" Out of",training_iters , " Loss= ",loss, "acc=",acc )

print("Optimization Finished!")

        

#     print("Step " + str(i) + ",        Batch Loss= ",loss, ",       Training Accuracy= ",acc)

    

# print("Optimization Finished!")

while(True):

    r = np.random.randint(9000)

    test_img = np.reshape(test_X[r], (28,28))

    plt.imshow(test_img, cmap="gray")

    test_pred = sess.run(prediction, feed_dict = {X:[test_X[r]]})

    print("Model : I think it is :    ",np.argmax(test_pred))

    plt.show()

    

    if input("Enter n to exit")=='n':

        break

clear_output();
wrong = test_X[t!=p]

wrong.shape
a,b,c = wrong.shape
while(True):

    r=np.random.randint(a)

    plt.imshow(wrong[r].reshape((28,28)),cmap="gray")

    test_pred_1=sess.run(prediction, feed_dict = {X:[wrong[r]]})

    print("Model : I think it is :    ",np.argmax(test_pred_1))

    plt.show()

    if input("Enter n to exit")=='n':

        break

clear_output();
p = np.argmax(predTest,1)

print(p)

t = np.argmax(np.array(test_Y),1)

print(t)

acc = accuracy_score(p,t)

print(acc*100)
print("Saving Weights")

saver = tf.train.Saver()

saver.save(sess,"weights_"+str(i)+"/weights.ckpt")

print("Weights Saved")