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
train_data = pd.read_csv('../input/fashion-mnist_train.csv')
train_data.sample(5)
train_data.groupby('label').head(1)
train_data.groupby('label').head(2).label.value_counts()
#train_X , only pixels data

train_X = train_data.drop('label', axis=1)



#train_y , only labels , dummies , shape = samples x 10

train_y = train_data.label

train_y = pd.get_dummies(train_y)



print(train_X.shape, train_y.shape)
#loading of test_data in the same way as train data

test_data = pd.read_csv('../input/fashion-mnist_test.csv')



test_X = test_data.drop('label', axis=1)

test_y = test_data.label

test_y = pd.get_dummies(test_y)



print(test_X.shape, test_y.shape)
#take 1 sample from every label and sort by label's number

samples = train_data.groupby('label').head(1).sort_values('label').index
import matplotlib.pyplot as plt



f = plt.figure(figsize=(10,6))



for i, index in enumerate(samples):

    a = f.add_subplot(2, 5, i + 1)

    a.axis('Off')  

    image = np.reshape(train_X.loc[index, :].values, (28, 28))

    label = train_data.label[index]

    plt.title('Label: {}'.format(label))

    plt.imshow(image, cmap='gray_r')

plt.show()
#function for batch training

def next_batch(train_X, train_y, batch_size):

    global i

    #print(i)

    train_X = train_X.loc[i*batch_size:(i+1)*batch_size-1, :]

    train_y = train_y.loc[train_X.index, :]

    i += 1

    #print(train_X.index, train_y.index)

    return train_X.values, train_y.values
next_batch(train_X, train_y, 2)
next_batch(train_X, train_y, 100)
next_batch(train_X, train_y, 100)
import tensorflow as tf



learning_rate = 0.01

training_iters = 60000

batch_size = 128

display_step = 10



#Network Parameters

n_input = 784 #28x28

n_classes = 10 #0-9

dropout = 0.75 #to reduce an overfitting, term refers to dropping out units in a neural network



#tf graph input , define placeholders(symbol zastepczy) foor the input raph

x = tf.placeholder(tf.float32, [None, n_input])

y = tf.placeholder(tf.float32, [None, n_classes]) #output tensor - it will contain the ouput probability for each class



#dropout (keep probability)

keep_prob = tf.placeholder(tf.float32)
#Create model

def conv2d(img, w, b):

    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], padding='SAME'), b)) #padding 'SAME' means the output tensor will have the same size of input tensor



def max_pool(img, k):

    return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')



#Store layers weights & bias

# 5x5 conv, 1 input, 32 outputs

wc1 = tf.Variable(tf.random_normal([5, 5, 1, 32])) #A variable maintains state in the graph across calls to run(), tf.random_normal outputs random values from a normal distribution

bc1 = tf.Variable(tf.random_normal([32]))



# 5x5 conv, 1 input, 32 outputs

#wc2 = tf.Variable(tf.random_normal([5, 5, 32, 64]))

#bc2 = tf.Variable(tf.random_normal([64]))



#Fully connected, 7*7*64 inputs, 1024 outputs

wd1 = tf.Variable(tf.random_normal([14*14*32, 4096]))



# 1024 inputs, 10 outputs (class prediction)

wout = tf.Variable(tf.random_normal([4096, n_classes]))

bd1 = tf.Variable(tf.random_normal([4096]))

bout = tf.Variable(tf.random_normal([n_classes]))
i = 0



#Construct model

_X = tf.reshape(x, shape=[-1, 28, 28, 1]) #Changing the form of 4D input iages to a tensor



#Convolution layer

conv1 = conv2d(_X, wc1, bc1)



#Max pooling

conv1 = max_pool(conv1, k=2)



#Apply dropout

conv1 = tf.nn.dropout(conv1, keep_prob)



#Convolution layer

#conv2 = conv2d(conv1, wc2, bc2)



#Max pooling

#conv2 = max_pool(conv2, k=2)



#Apply dropout

#conv2 = tf.nn.dropout(conv2, keep_prob)



#Fully connected layer

#Reshape conv2 output to fit dense layer input

dense1 = tf.reshape(conv1, [-1, wd1.get_shape().as_list()[0]])



#Relu activation

dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, wd1), bd1))



#Apply dropout

dense1 = tf.nn.dropout(dense1, keep_prob)



#Output, class prediction

pred = tf.add(tf.matmul(dense1, wout), bout)



#Define loss and optimizer

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)



#Evaluate model

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



#Initializing the variables

init = tf.initialize_all_variables()



#Launch the graph

with tf.Session() as sess:

    sess.run(init)

    step = 1

    #Keep training until reach max iterations

    while step * batch_size < training_iters:

        batch_xs, batch_ys = next_batch(train_X, train_y, batch_size)

        #print(batch_xs, batch_ys)

        #Fit training using batch data

        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})

        if step % display_step == 0:

            #Calculate batch accuracy

            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})

            #Calculate batch loss

            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})

            

            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))

        step += 1

    print("Optimization finished")

    #Calculate accuracy for 256 mnist test images

    print("Testing accuracy:", sess.run(accuracy, feed_dict={x: test_X.values[:512], y: test_y.values[:512], keep_prob: 1.}))