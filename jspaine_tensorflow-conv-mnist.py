# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



def submit(data, predictions, name):

    submission = pd.DataFrame({

        "ImageId": range(1,28001),

        "Label": predictions

    })

    submission.to_csv("%s.csv" % name, index=False)



class Flags:

    def __init__(self):

        self.train = True

        self.train_iter = 8001



FLAGS = Flags()



def weight_var(shape):

    initial = tf.truncated_normal(shape, stddev=0.1)

    return tf.Variable(initial, name="weights")



def bias_var(shape):

    initial = tf.constant(0.1, shape=shape)

    return tf.Variable(initial, name="bias")



def conv2d(x, W):

    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")



def max_pool_2x2(x):

    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],

                          strides=[1, 2, 2, 1], padding="SAME")



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

x_test = np.array(test)
x_train = train.iloc[:, 1:]

y_train = np.array(pd.get_dummies(train["label"]))



print("train shape: %s, y_train: %s, x_train: %s" % 

      (train.shape, y_train.shape, x_train.shape))

#print("test shape: %s" % x_test.shape)



class Data:

    def __init__(self, train, test):

        self.train = train

        self.train_size = train["y"].shape[0]

        self.train_idx = 0

        self.test = test

        self.test_size = test.shape[0]

        self.test_idx = 0

        self.num_test_batches = 56

        

    def next_train_batch(self, num):

        next_idx = self.train_idx + num

        if next_idx > self.train_size:

            result = np.array([

                self.train["x"].iloc[self.train_idx : self.train_size],

                self.train["y"][self.train_idx : self.train_size]

            ])

            self.train_idx = 0

        else:

            result = np.array([

                self.train["x"].iloc[self.train_idx : next_idx],

                self.train["y"][self.train_idx : next_idx]

            ])

            self.train_idx = next_idx

        return result

    

    def next_test_batch(self):

        if self.test_idx <= self.num_test_batches:

            num_per_batch = int(self.test_size / self.num_test_batches)

            start = self.test_idx * num_per_batch

            self.test_idx += 1

            end = self.test_idx * num_per_batch

            result = self.test[start : end]

        else:

            print("test data exhausted")

            result = np.empty([0])

        return result

    

data = Data(train={"x": x_train, "y": y_train}, test=x_test)

batch = data.next_train_batch(100)

print(batch[0].shape, batch[1].shape)
x = tf.placeholder(tf.float32, shape=[None, 784], name="x")

y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y_")

x_image  = tf.reshape(x, [-1, 28, 28, 1], name="x_image")

keep_prob = tf.placeholder(tf.float32, name="keep_prob")



with tf.name_scope("conv_1"):

    W_conv1 = weight_var([5, 5, 1, 16])

    b_conv1 = bias_var([16])



    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    h_pool1 = max_pool_2x2(h_conv1)



with tf.name_scope("conv_2"):

    W_conv2 = weight_var([5, 5, 16, 32])

    b_conv2 = bias_var([32])



    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    h_pool2 = max_pool_2x2(h_conv2)



with tf.name_scope("fc_1"):

    W_fc1 = weight_var([7 * 7 * 32, 256])

    b_fc1 = bias_var([256])



    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 32])

    h_fc1 = tf.nn.relu(tf.nn.xw_plus_b(h_pool2_flat, W_fc1, b_fc1))



with tf.name_scope("fc_2"):

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)



    W_fc2 = weight_var([256, 10])

    b_fc2 = bias_var([10])



with tf.name_scope('Model'):

    y = tf.nn.xw_plus_b(h_fc1_drop, W_fc2, b_fc2)

    

with tf.name_scope('Loss'):

    cross_entropy = tf.reduce_mean(

        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    

with tf.name_scope('Optimizer'):

    global_step = tf.Variable(0, trainable=False)

    initial_rate = 0.005

    rate = tf.train.exponential_decay(initial_rate, global_step, 300, 0.9)

    train_step = (tf.train.AdamOptimizer(rate)

                  .minimize(cross_entropy, global_step=global_step))

    

with tf.name_scope('Accuracy'):

    correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
sess = tf.Session()

init = tf.global_variables_initializer()

sess.run(init)



for i in range(FLAGS.train_iter):

    train_batch = data.next_train_batch(50)

    train_dict = {x: train_batch[0], y_: train_batch[1],

                  keep_prob: 1.0}

    train_dict_drop = {x: train_batch[0], y_: train_batch[1],

                      keep_prob: 0.8}    



    if i % 100 == 0:

        train_accuracy = accuracy.eval(feed_dict=train_dict, session=sess)

        print("step %d, train %g" % (i, train_accuracy))

        

    sess.run([train_step], feed_dict=train_dict_drop)



prediction = tf.argmax(y, 1)

predictions = prediction.eval(

        feed_dict={x: data.next_test_batch(), keep_prob: 1.0}, 

        session=sess)

for i in range(data.num_test_batches - 1):

    batch = data.next_test_batch()

    results = prediction.eval(

        feed_dict={x: batch, keep_prob: 1.0}, 

        session=sess)



    predictions = np.concatenate((predictions, results))

    

print(predictions.shape)

submit(test, predictions, 'out')