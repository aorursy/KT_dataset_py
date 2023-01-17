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
from sklearn.model_selection import train_test_split
from IPython.display import clear_output
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
train = pd.read_csv('../input/mnist_train.csv')
test = pd.read_csv('../input/mnist_test.csv')
print(train.shape)
print(test.shape)
print(train.label)
print(test)
train = np.array(train)
print(train)
test = np.array(test)
print(test)

train_x = train[:,1:]
print(train_x)
train_y = pd.get_dummies(train[:,0])
print(train_y)
test_x = test[:,1:]
print(test_x)
test_y = pd.get_dummies(test[:,0])
print(test_y)
train_x = train_x.reshape(-1, 28, 28, 1)
test_x = test_x.reshape(-1, 28, 28, 1)
train_X,train_y = shuffle(train_x, train_y)
test_X,test_y = shuffle(test_x, test_y)
training_iters = 10
learning_rate = 0.001 
batch_size = 100

n_input = 28
n_classes = 10


x = tf.placeholder("float", [None, 28,28,1])
y = tf.placeholder("float", [None, n_classes])
def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME') # first 1 is for batch last 1 is for channels
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x) 

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')
weights={}
biases={}
weights = {
    'wc1': tf.Variable(tf.truncated_normal([3,3,1,32])),
    'wc2': tf.Variable(tf.truncated_normal([3,3,32,64])), 
    'wc3': tf.Variable(tf.truncated_normal([3,3,64,128]))
}
biases = {
    'bc1': tf.Variable(tf.truncated_normal([32])),
    'bc2': tf.Variable(tf.truncated_normal([64])),
    'bc3': tf.Variable(tf.truncated_normal([128]))
    }
def conv_net(x, weights, biases):  

    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)

    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)

    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    conv3 = maxpool2d(conv3, k=2)


    # Fully connected layer
    fc1 = tf.reshape(conv3, [-1, 4*4*128])
    fc1 = tf.layers.dense(fc1, 100)
    
    out = tf.layers.dense(fc1, 10, activation=None)
    return out
pred = conv_net(x, weights, biases)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) 

for i in range(training_iters):
    for batch in range(len(train_X)//batch_size):
        batch_x = train_X[batch*batch_size:min((batch+1)*batch_size,len(train_X))]
        batch_y = train_y[batch*batch_size:min((batch+1)*batch_size,len(train_y))]    

        
        opt = sess.run(optimizer, feed_dict={x: batch_x,
                                             y: batch_y})
        
        loss = sess.run(cost, feed_dict={x: batch_x,
                                         y: batch_y})
    predTest = sess.run(pred , feed_dict={x:test_X})

    p = np.argmax(predTest,1)
    t = np.argmax(np.array(test_y),1)

    acc = accuracy_score(p,t)
    print("Iter "+str(i)+" Out of",training_iters , " Loss= ",loss, "acc=",acc )
    print("Optimization Finished!")
    

while(True):
    r = np.random.randint(9000)
    test_img = np.reshape(test_X[r], (28,28))
    plt.imshow(test_img, cmap="gray")
    test_pred = sess.run(pred, feed_dict = {x:[test_X[r]]})
    print("Model : I think it is :    ",np.argmax(test_pred))
    plt.show()
    
    if input("Enter n to exit")=='n':
        break
    clear_output();
wrong = test_X[t!=p]
wrong.shape
r=np.random.randint(200)
plt.imshow(wrong[r].reshape((28,28)),cmap="gray")
test_pred_1=sess.run(pred, feed_dict = {x:[wrong[r]]})
print("Model : I think it is :    ",np.argmax(test_pred_1))

p = np.argmax(predTest,1)
print(p)
t = np.argmax(np.array(test_y),1)
print(t)
acc = accuracy_score(p,t)
print(acc*100)
    
