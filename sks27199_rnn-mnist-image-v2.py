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

import matplotlib.pyplot as plt

from IPython.display import clear_output

import pandas as pd

from sklearn.utils import shuffle

from sklearn.metrics import accuracy_score
train = pd.read_csv("../input/mnist_train.csv")

test = pd.read_csv("../input/mnist_test.csv")
print(train.shape)

print(test.shape)

print(train.columns)

print(train[:2])

print(test[:2])

print(train[:10]["label"])

print(test[:10]["label"])
train = np.array(train)

test = np.array(test)
print(train[10:15])

print(test[10:15])
train_x = train[:,1:]

train_y = pd.get_dummies(train[:,0])

test_x = test[:,1:]

test_y = pd.get_dummies(test[:,0])
print(train_x.shape)

print(train_y.shape)

print(test_x.shape)

print(test_y.shape)

print(train_x[10:12])

print(train_y[10:12])

print(test_x[10:12])

print(test_y[10:12])
#NETWORK PARAMETERS

n_steps = 28

n_inputs = 28

n_neurons = 150

n_outputs = 10
train_x = train_x.reshape(-1,n_steps,n_inputs)

test_x = test_x.reshape(-1,n_steps,n_inputs)
print(train_x.shape)

print(test_x.shape)

print(train_x[0:1])

print(test_x[0:1])
train_X , train_y = shuffle(train_x , train_y)

test_X , test_y = shuffle(test_x , test_y)
tf.reset_default_graph()
# Training Parameters

learning_rate = 0.001

training_iters = 500

batch_size = 150

display_step = 200
X = tf.placeholder(tf.float32,[None,n_steps,n_inputs])

y = tf.placeholder(tf.int32,[None,n_outputs])
def RNN(X):

    basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)

    outputs , states = tf.nn.dynamic_rnn(basic_cell,X,dtype=tf.float32)

    out = tf.layers.dense(states, n_outputs)

    out = tf.nn.softmax(out)

    return out

    
prediction = RNN(X)
# Define loss and optimizer

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

train_op = optimizer.minimize(loss_op)
# correct_pred = tf.nn.in_top_k(prediction, y, 1)
# Evaluate model (with test logits, for dropout to be disabled)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



# Initialize the variables (i.e. assign their default value)

init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init) 



for i in range(training_iters):

    for batch in range(len(train_X)//batch_size):

        batch_x = train_X[batch*batch_size:min((batch+1)*batch_size,len(train_X))]

        batch_y = train_y[batch*batch_size:min((batch+1)*batch_size,len(train_y))]

        batch_x = batch_x.reshape((-1, n_steps, n_inputs))



        sess.run(train_op, feed_dict={X: batch_x, y: batch_y})

        loss = sess.run([loss_op], feed_dict={X: batch_x, y: batch_y})

    predTest = sess.run(prediction , feed_dict={X:test_X})

    acc_train = accuracy.eval(session=sess, feed_dict={X: batch_x, y: batch_y})

    acc_test = accuracy.eval(session=sess, feed_dict={X: test_X, y: test_y})

    print("Iter "+str(i)+" Out of",training_iters , " Loss= ",loss,"Train accuracy:", acc_train, "Test accuracy:", acc_test)

        

        

            # Calculate batch loss and accuracy

#         loss = sess.run([loss_op], feed_dict={X: batch_x, y: batch_y})

    

#     predTest = sess.run(prediction , feed_dict={X:test_X})



#     p = np.argmax(predTest,1)

#     t = np.argmax(np.array(test_y),1)



#     acc = accuracy_score(p,t)

#     print("Iter "+str(i)+" Out of",training_iters , " Loss= ",loss, "acc=",acc )

            

#     acc = sess.run([accuracy], feed_dict={X: batch_x, y: batch_y})

        

#     print("Step " + str(i) + ",        Batch Loss= ",loss, ",       Training Accuracy= ",acc)

    

print("Optimization Finished!")
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
wrong = test_X[tf.argmax(prediction, 1)!=tf.argmax(y, 1)]

wrong.shape
a,b,c, d = wrong.shape
wrong = np.reshape(wrong, (b,c,d))   

wrong.shape
while(True):

    r=np.random.randint(b)

    plt.imshow(wrong[r].reshape((28,28)),cmap="gray")

    test_pred_1=sess.run(prediction, feed_dict = {X:[wrong[r]]})

    print("Model : I think it is :    ",np.argmax(test_pred_1))

    plt.show()

    

    if input("Enter n to exit")=='n':

        break

clear_output();
p = np.argmax(predTest,1)

print(p)

t = np.argmax(np.array(test_y),1)

print(t)

acc = accuracy_score(p,t)

print(acc*100)
print("Saving Weights")

saver = tf.train.Saver()

saver.save(sess,"weights_"+str(i)+"/weights.ckpt")

print("Weights Saved")