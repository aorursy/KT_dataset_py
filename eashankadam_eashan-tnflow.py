import pandas as pd

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
% matplotlib inline
train_labels = train.iloc[:, 0].values
train_images = train.iloc[:, 1:].values
train_images = np.multiply(train_images, 1./255)

test_images = test.values
test_images = np.multiply(test_images, 1./255)
n_input = 784
n_class = 10
n_hidden1 = 256
n_hidden2 = 256
x = tf.placeholder("float", shape=[None, n_input]) 
y = tf.placeholder("float", shape=[None, n_class])
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
def cnn(X):
    
 W_conv1 = weight_variable([5, 5, 1, 32])
 b_conv1 = bias_variable([32])

 x_image = tf.reshape(x, [-1,28,28,1])

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

 keep_prob = tf.placeholder("float")
 h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
 W_fc2 = weight_variable([1024, 10])
 b_fc2 = bias_variable([10])

 y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

 return y_conv



cost = tf.nn.softmax_cross_entropy_with_logits(nn_2_layers, y)
train = tf.train.AdamOptimizer(0.001).minimize(cost)
def dense_to_one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot
train_labels = dense_to_one_hot(train_labels)
init = tf.initialize_all_variables()
correct_prediction = tf.equal(tf.argmax(nn_2_layers,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step * 100 < 42000:
        batch_x = train_images[step*100:(step+1)*100]
        batch_y = train_labels[step*100:(step+1)*100]
        cost = sess.run(train, feed_dict={x: batch_x, y: batch_y})
        if step % 10 == 0:
            print("Iter:", step*100, "Accuracy:", sess.run(accuracy, feed_dict={x: batch_x, y: batch_y}))
        step += 1