from __future__ import print_function

import matplotlib.pyplot as plt

import numpy as np

import cv2

import pandas as pd

from tqdm import tqdm

from sklearn.model_selection import train_test_split

import tensorflow as tf

# Config the matplotlib backend as plotting inline in IPython

%matplotlib inline
sess = tf.InteractiveSession()
train_set = pd.read_csv('../input/train_labels.csv')

test_set = pd.read_csv('../input/sample_submission.csv')



def read_img(img_path):

    img = cv2.imread(img_path)

    img = cv2.resize(img, (128, 128))

    return img



train_img, test_img = [], []

for img_path in tqdm(train_set['name'].iloc[: ]):

    train_img.append(read_img('../input/train/' + str(img_path) + '.jpg'))

for img_path in tqdm(test_set['name'].iloc[: ]):

    test_img.append(read_img('../input/test/' + str(img_path) + '.jpg'))



train_img = np.array(train_img, np.float32) / 255

train_label = np.array(train_set['invasive'].iloc[: ])

test_img = np.array(test_img, np.float32) / 255
dataset = np.array(train_img, dtype=np.float32)

dataset.shape
X_train, X_test, y_train, y_test = train_test_split(train_img, train_label, test_size=0.3, random_state=324)
#y_train = np.reshape(y_train, [-1,1])

#y_test = np.reshape(y_test, [-1,1])

y_train = pd.get_dummies(y_train)

y_test =  pd.get_dummies(y_test)



y_train
batch_size, height, width, channels = dataset.shape

class_output = 2 # Not sure about this but there is only one class

x  = tf.placeholder(tf.float32, shape=[None, height, width, channels])

y_ = tf.placeholder(tf.float32, shape=[None, class_output])
W_conv1 = tf.Variable(tf.truncated_normal([5, 5, channels, 16], stddev=0.1))

b_conv1 = tf.Variable(tf.constant(0.1, shape=[16])) # need 16 biases for 16 outputs
convolve1= tf.nn.conv2d(x, W_conv1, strides=[1, 2, 2, 1], padding='SAME') + b_conv1
h_conv1 = tf.nn.relu(convolve1)
conv1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #max_pool_2x2

conv1
W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 16, 32], stddev=0.1))

b_conv2 = tf.Variable(tf.constant(0.1, shape=[32])) #need 32 biases for 32 outputs

convolve2= tf.nn.conv2d(conv1, W_conv2, strides=[1, 2, 2, 1], padding='SAME')+ b_conv2

h_conv2 = tf.nn.relu(convolve2)

conv2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #max_pool_2x2

conv2
layer2_matrix = tf.reshape(conv2, [-1, 8*8*32])
W_fc1 = tf.Variable(tf.truncated_normal([8 * 8 * 32, 512], stddev=0.1))

b_fc1 = tf.Variable(tf.constant(0.1, shape=[512])) # need 512 biases for 512 outputs
fcl=tf.matmul(layer2_matrix, W_fc1) + b_fc1
h_fc1 = tf.nn.relu(fcl)

h_fc1
keep_prob = tf.placeholder(tf.float32)

layer_drop = tf.nn.dropout(h_fc1, keep_prob)

layer_drop
W_fc2 = tf.Variable(tf.truncated_normal([512, 1], stddev=0.1)) #512 neurons

b_fc2 = tf.Variable(tf.constant(0.1, shape=[2])) # 1 possibility
fc=tf.matmul(layer_drop, W_fc2) + b_fc2
y_CNN= tf.nn.softmax(fc)

y_CNN
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_CNN), reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_CNN,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())
X_train_tmp = X_train[:200]

y_train_tmp = y_train[:200]
for i in range(500):

    if i%50 == 0:

        train_accuracy = accuracy.eval(feed_dict={x:X_train_tmp, y_: y_train_tmp, keep_prob: 1.0})

        print("step %d, training accuracy %g"%(i, float(train_accuracy)))

    train_step.run(feed_dict={x: X_train_tmp, y_: y_train_tmp, keep_prob: 0.5})
X_test_tmp = X_test[:100]

y_test_tmp = y_test[:100]
print("test accuracy %g"%accuracy.eval(feed_dict={x: X_test_tmp, y_: y_test_tmp, keep_prob: 1.0}))