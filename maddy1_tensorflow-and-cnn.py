# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
from PIL import Image
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
import cv2
%matplotlib inline
import tensorflow as tf
import time
import math

from sklearn.model_selection import train_test_split
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
all_data1 = np.load('../input/Sign-language-digits-dataset/X.npy')
all_labels1 = np.load('../input/Sign-language-digits-dataset/Y.npy')
test_size = 0.15
X_train, X_test, Y_train, Y_test = train_test_split(all_data1, all_labels1, test_size=test_size, random_state=42)
img_size = 64
channel_size = 1
num_classes = 10
img_shape = (img_size, img_size)

print("Training Size:", X_train.shape)
print(X_train.shape[0],"samples - ", X_train.shape[1],"x",X_train.shape[2],"grayscale image")

print("\n")

print("Test Size:",X_test.shape)
print(X_test.shape[0],"samples - ", X_test.shape[1],"x",X_test.shape[2],"grayscale image")
Y_test_cls = np.argmax(Y_test, axis=1)
Y_train_cls = np.argmax(Y_train, axis=1)
#test some images
print('Test Images:')
n = 10
plt.figure(figsize=(20,20))
for i in range(1, n+1):
    ax = plt.subplot(1, n, i)
    plt.imshow(X_train[i].reshape(img_size, img_size))
    plt.gray()
    plt.axis('off')
train_X = X_train
train_Y = Y_train
train_X_new = train_X.reshape(X_train.shape[0],(img_size * img_size))
test_X_new = X_test.reshape(X_test.shape[0],(img_size * img_size))
print("Training set :",train_X_new.shape,train_Y.shape)

# Shapes of test set
print("Test set (images)",test_X_new.shape)
print("Test set (labels)",Y_test.shape)
learning_rate = 0.001
training_iters = 50000
batch_size = 32
display_step = 20

n_input = img_size * img_size # 64x64 image
dropout = 0.75 
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)
print('Shape of placeholder',X.shape, Y.shape)
def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)
def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
def conv_net(x, weights, biases, dropout):
    # reshape input to 64x64 size
    x = tf.reshape(x, shape=[-1, 64, 64, 1])
    

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
    fc1 = tf.nn.relu(fc1) # layer
    fc1 = tf.nn.dropout(fc1, dropout)

    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32]),name='wc1'),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64]),name='wc2'),
    'wd1': tf.Variable(tf.random_normal([64 * 64 * 4, 1024]),name='wd1'),
    'out': tf.Variable(tf.random_normal([1024, num_classes]),name='wout')
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32]),name='bc1'),
    'bc2': tf.Variable(tf.random_normal([64]),name='bc2'),
    'bd1': tf.Variable(tf.random_normal([1024]),name='bd1'),
    'out': tf.Variable(tf.random_normal([num_classes]),name='bout')
}
import random
import datetime


def getbatch(alldata, alllabels, batch_size = 16):
    nlabels = alllabels.shape[0]
    #print(nlabels)
    number_of_batches = nlabels//batch_size # change to nlabels
    for batch_number in range(number_of_batches):
        rand_index = np.array([random.randrange(0, nlabels) for i in range(batch_size)])        
        batch_x = alldata[rand_index]
        batch_x = (batch_x-batch_x.min())/(batch_x.max()-batch_x.min())
        batch_y = alllabels[rand_index]
        yield (batch_x, batch_y)

# Create the model
model = conv_net(x, weights, biases, keep_prob)
print(model)
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
y_true_cls =  tf.argmax(y, 1)
y_pred_cls = tf.argmax(model, 1)
# This is a vector of booleans whether the predicted 
#class equals the true class of each image.
correct_model = tf.equal(y_pred_cls,y_true_cls)
# This calculates the classification accuracy by first type-casting 
#the vector of booleans to floats, so that False becomes 0 and True becomes 1,
#and then calculating the average of these numbers.
accuracy = tf.reduce_mean(tf.cast(correct_model, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
loss_train = []
steps_train = []
acc_train = []

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1   
#     Keep training until reach max iterations
    while step * batch_size < training_iters:
        a = getbatch(train_X_new,train_Y, batch_size)
        batch_x, batch_y = next(a)
        #print(batch_x.shape, batch_y.shape)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
        if step % display_step == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Loss= " + \
                  "{:.3f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
            loss_train.append(loss)
            steps_train.append(step*batch_size)
            acc_train.append(acc)
        step += 1
    
   #
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_X_new,
                                      y: Y_test,
                                      keep_prob: 1.}))
    
    cls_pred = sess.run(y_pred_cls, feed_dict={x: test_X_new,
                                      y: Y_test,
                                      keep_prob: 1.})
#get the predictions for the test data

predicted_classes = cls_pred

#get the indices to be plotted

y_true = Y_test_cls

correct = np.nonzero(cls_pred==y_true)[0]


incorrect = (correct == False)

from sklearn.metrics import classification_report

target_names = ["Class {}".format(i) for i in range(num_classes)]

print(classification_report(y_true, predicted_classes, target_names=target_names))