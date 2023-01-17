import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



%matplotlib inline
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
img = train.loc[0][1:].values.reshape((28,28))

plt.imshow(img, cmap=plt.get_cmap('gray'))
def handle_labels(num_array):

    length = len(num_array)

    patten = np.zeros([length, 10])

    for i in range(length):

        patten[i][num_array[i]] = 1

    return patten
batch_num = 0

examples_num = train.shape[0]

def next_batch(batch_size):

    global batch_num

    global train



    start = batch_num

    end = batch_num + batch_size - 1



    if end > examples_num:

        train = train.sample(frac=1).reset_index(drop=True)

        batch_num = 0

        start = batch_num

        end = batch_num + batch_size - 1



    data = train.loc[start:end]

    labels = data.pop('label')

    batch_num = end + 1

    return data.values, handle_labels(labels.values)
import tensorflow as tf



x = tf.placeholder(tf.float32, shape=[None, 784])

y_ = tf.placeholder(tf.float32, shape=[None, 10])



def weight_variable(shape):

  initial = tf.truncated_normal(shape, stddev=0.1)

  return tf.Variable(initial)



def bias_variable(shape):

  initial = tf.constant(0.1, shape=shape)

  return tf.Variable(initial)



def conv2d(x, W):

  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')



def max_pool_2x2(x):

  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')



W_conv1 = weight_variable([5, 5, 1, 32])

b_conv1 = bias_variable([32])



x_image = tf.reshape(x, [-1, 28, 28, 1])



h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

h_pool1 = max_pool_2x2(h_conv1)



W_conv2 = weight_variable([5, 5, 32, 64])

b_conv2 = bias_variable([64])



h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

h_pool2 = max_pool_2x2(h_conv2)



W_fc1 = weight_variable([7 * 7 * 64, 1024])

b_fc1 = bias_variable([1024])



h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)



keep_prob = tf.placeholder(tf.float32)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)



W_fc2 = weight_variable([1024, 10])

b_fc2 = bias_variable([10])



y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2



cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

predict = tf.argmax(y_conv, 1)



saver = tf.train.Saver()



with tf.Session() as sess:

  sess.run(tf.global_variables_initializer())

  for i in range(10000):

    batch = next_batch(100)

    if i % 100 == 0:

      train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})

      print('step %d, training accuracy %g' % (i, train_accuracy))

    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})



  save_path = saver.save(sess, "model/model.ckpt")

  print("Model saved in file: %s" % save_path)
predicted_lables = np.zeros(test.shape[0])



with tf.Session() as sess:

  # Restore variables from disk.

  saver.restore(sess, "model/model.ckpt")

  # Check the values of the variables

  for i in range(0, test.shape[0]//100):

    predicted_lables[i*100 : (i+1)*100] = predict.eval(feed_dict={x: test.values[i*100 : (i+1)*100], keep_prob: 1.0})
predicted_lables[0:10]
plt.figure(num='astronaut',figsize=(28,28))

for i in range(0,10):  

    plt.subplot(1,10,i + 1)

    img = test.values[i].reshape((28,28))

    plt.imshow(img, cmap=plt.get_cmap('gray'))
np.savetxt('submission_cnn.csv', 

           np.c_[range(1,len(test)+1), predicted_lables], 

           delimiter=',', 

           header = 'ImageId,Label', 

           comments = '',

           newline='\r\n',

           fmt='%d')
sub = pd.read_csv('submission_cnn.csv')

sub.head(10)