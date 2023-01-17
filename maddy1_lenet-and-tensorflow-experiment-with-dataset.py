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
seed = 128
rng = np.random.RandomState(seed)
train_data = pd.read_csv('/kaggle/input/fashion-mnist_train.csv')
test_data = pd.read_csv('/kaggle/input/fashion-mnist_test.csv')
train=np.array(train_data).astype('float32')
test=np.array(test_data)
x_train = train[:,1:]/255

y_train = train[:,0]


x_test= test[:,1:]/255

y_test=test[:,0]
image = x_train.loc[1:4,:].reshape((28,28))
plt.imshow(image, cmap='gray')


from sklearn.model_selection import train_test_split
x_train,x_validate,y_train,y_validate = train_test_split(x_train,y_train,test_size = 0.2,random_state = 12345)
y_train = np.array(y_train,dtype='int32')
y_train_onehot = np.eye(10)[y_train]
y_train_onehot=np.array(y_train_onehot,dtype='float32')
print(y_train_onehot[25],y_train_onehot[25].dtype)
# architecture hyper-parameter
learning_rate = 0.01
training_iters = 100000
batch_size = 64
display_step = 5

n_input = 784 # 28x28 image
n_classes = 10 # 1 for each digit [0-9]
dropout = 0.75 
image_shape = (28,28,1)
noofepochs = 100


x = tf.placeholder(tf.float32, [None, n_input])
y= tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)
print(x.shape, y.shape,y.dtype)
def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)
def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
def conv_net(x, weights, biases, dropout):
    # reshape input to 28x28 size
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution layer 1
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max pooling
    conv1 = maxpool2d(conv1, k=2)

    # Convolution layer 2
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max pooling
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)

    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out
weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
# Create the model
model = conv_net(x, weights, biases, keep_prob)
print(model)
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_model = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_model, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
print(y_train[8])
import random
import datetime


def getbatch(all_data, all_label, batchsize=16):
    count = 0
    arraylength=len(all_data)
    while count < arraylength/batchsize:
        random.seed(datetime.datetime.now())
        randstart = random.randint(0, arraylength-batchsize-1)
        count += 1
        batch_x=np.reshape(all_data[randstart:randstart+batchsize],(batchsize,784))
        yield (batch_x, all_label[randstart:randstart+batchsize])

a = getbatch(x_train,y_train_onehot, batchsize = batch_size)
x, y = next(a)
print(x.shape, y.shape, x.min(), x.max())



# Launch the graph


with tf.Session() as sess:
    sess.run(init)
    for i in range(50):
        for epoch in range(noofepochs):
            batch_x = x_train.iloc[epoch*100:(epoch+1)*100-1, :]
            batch_y = y_train_onehot[epoch*100:(epoch+1)*100-1, :]
            sess.run(optimizer, feed_dict={x:batch_x, y:batch_y})
       
    
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        for batch_x, batch_y in getbatch(x_train,y_train_onehot,batch_size):  
            print(batch_x.shape, batch_y.shape,batch_x.dtype,batch_y.dtype)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
            if step % display_step == 0:
                    loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                                  y: batch_y,
                                                                  keep_prob: 1.})
                    print("Iter " + str(step*batch_size) + ", Loss= " + \
                          "{:.3f}".format(loss) + ", Training Accuracy= " + \
                          "{:.3f}".format(acc))
            step += 1