from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np # matrix tools
import pandas as pd
import matplotlib.pyplot as plt # for basic plots
import time
import os
import math

import tensorflow as tf
overview_df = pd.read_csv('../input/overview.csv')
overview_df.columns = ['idx']+list(overview_df.columns[1:])
overview_df['Contrast'] = overview_df['Contrast'].map(lambda x: 1 if x else 0) #1 for contrast, 0 for no contrast
overview_df.index
overview_df.sample(3)
overview_df.shape
im_data = np.load('../input/full_archive.npz')
# make a dictionary of the data vs idx
full_image_dict = dict(zip(im_data['idx'], im_data['image']))
full_image_dict[0]
full_image_dict[0].shape
for x in full_image_dict.keys():
    full_image_dict[x] = (full_image_dict[x] - full_image_dict[x].min()) \
    / (full_image_dict[x].max() - full_image_dict[x].min()) * 255
    full_image_dict[x] = full_image_dict[x][::2,::2]
full_image_dict[0]
full_image_dict[0].shape
labels = dict(zip(overview_df['idx'],overview_df['Contrast']))
labels
len(full_image_dict.keys())
train_data = np.asarray([full_image_dict[x].flatten() for x in list(full_image_dict.keys())[:80] if len(full_image_dict[x].flatten()) == 256*256])
train_labels = np.asarray([labels[x] for x in list(full_image_dict.keys())[:80] if len(full_image_dict[x].flatten()) == 256*256])
train_labels
tmp = np.zeros((train_labels.shape[0],2))
for i,x in enumerate(tmp):
    if train_labels[i] == 0:
        tmp[i][0] = 1
    else:
        tmp[i][1] = 1
train_labels = tmp
train_labels
test_data = np.asarray([full_image_dict[x].flatten() for x in list(full_image_dict.keys())[80:90] if len(full_image_dict[x].flatten()) == 256*256])
test_labels = np.asarray([labels[x] for x in list(full_image_dict.keys())[80:90] if len(full_image_dict[x].flatten()) == 256*256])
tmp = np.zeros((test_labels.shape[0],2))
for i,x in enumerate(tmp):
    if test_labels[i] == 0:
        tmp[i][0] = 1
    else:
        tmp[i][1] = 1
test_labels = tmp
test_labels
IMAGE_SIZE = 256
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 2
TRAIN_SIZE = train_labels.shape[0]
VALIDATION_SIZE = test_labels.shape[0]  # Size of the validation set.

FILTER_SIZE = 5
FILTER_1 = 20
FILTER_2 = 50
HIDDEN_NUM = 100
LEARNING_RATE = 0.01

SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 1
NUM_EPOCHS = 1 # set to be 1 for quick publish
EVAL_BATCH_SIZE = 1
EVAL_FREQUENCY = 20  # Number of steps between evaluations.
activation = tf.nn.relu


def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
						strides=[1, 2, 2, 1], padding='SAME')
x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE*IMAGE_SIZE])
y_ = tf.placeholder(tf.float32, shape=[None, NUM_LABELS])

# conv1
W_conv1 = weight_variable([FILTER_SIZE, FILTER_SIZE, 1, FILTER_1])
b_conv1 = bias_variable([FILTER_1])
x_image = tf.reshape(x, [-1,IMAGE_SIZE,IMAGE_SIZE,1])
h_conv1 = activation(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# conv2
W_conv2 = weight_variable([FILTER_SIZE, FILTER_SIZE, FILTER_1, FILTER_2])
b_conv2 = bias_variable([FILTER_2])
h_conv2 = activation(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#fc1
W_fc1 = weight_variable([IMAGE_SIZE //4 * IMAGE_SIZE // 4* FILTER_2, HIDDEN_NUM])
b_fc1 = bias_variable([HIDDEN_NUM])
h_pool2_flat = tf.reshape(h_pool2, [-1, IMAGE_SIZE //4 * IMAGE_SIZE // 4* FILTER_2])
h_fc1 = activation(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = h_fc1#tf.nn.dropout(h_fc1, keep_prob)

#fc2
W_fc2 = weight_variable([HIDDEN_NUM, NUM_LABELS])
b_fc2 = bias_variable([NUM_LABELS])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#output
cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_conv, labels=y_))

#training
train_step = tf.train.GradientDescentOptimizer(learning_rate = LEARNING_RATE, use_locking = True).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
# correct_prediction = tf.equal(y_conv, y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
def eval_in_batches(images, labels,sess):
    """Get all predictions for a dataset by running it in small batches."""
    size = labels.shape[0]
    if size < EVAL_BATCH_SIZE:
        raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = []
    for begin in range(1, size // EVAL_BATCH_SIZE):
        end = begin * EVAL_BATCH_SIZE
        if end <= size:
            predictions.append(sess.run(
                accuracy,
                feed_dict={x: images[(begin-1)*EVAL_BATCH_SIZE:end,], y_: labels[(begin-1)*EVAL_BATCH_SIZE:end,], keep_prob: 0.5}))
        else:
            batch_predictions = sess.run(
                accuracy,
                feed_dict={x: images[-EVAL_BATCH_SIZE:, ...],y_: labels[-EVAL_BATCH_SIZE:, ...],keep_prob:0.5})
            predictions.append(batch_predictions)
    return np.mean(predictions)

#initilization
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
#training
start_time = time.time()
for step in range(int(NUM_EPOCHS * TRAIN_SIZE) // BATCH_SIZE):
	# Compute the offset of the current minibatch in the data.
	# Note that we could use better randomization across epochs.
# 	if step % (10*EVAL_FREQUENCY) == 0:
# 		checkpoint_file = os.path.join('./', 'model.ckpt')
# 		saver.save(sess, checkpoint_file, global_step=step)
	offset = (step * BATCH_SIZE) % (TRAIN_SIZE - BATCH_SIZE)
	batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
# 	print (batch_data.shape)
	batch_labels = train_labels[offset:(offset + BATCH_SIZE), ...]
# 	print (batch_labels.shape)
	# This dictionary maps the batch data (as a numpy array) to the
	# node in the graph it should be fed to.
	feed_dict = {x: batch_data,
			   y_: batch_labels,
			   keep_prob: 1.0}
	# Run the optimizer to update weights.
	sess.run(train_step, feed_dict=feed_dict)
	# print some extra information once reach the evaluation frequency
	if step % EVAL_FREQUENCY == 0:
		# fetch some extra nodes' data
		elapsed_time = time.time() - start_time
		l, acc = sess.run([cross_entropy, accuracy], feed_dict=feed_dict)
		start_time = time.time()
		print('Step %d (epoch %.2f), %.1f ms' %
			  (step, float(step) * BATCH_SIZE / TRAIN_SIZE,
			   1000 * elapsed_time / EVAL_FREQUENCY))
		print('Minibatch loss: %.3f' % l)
		print('Minibatch acc: %.2f%%' % (acc*100))
		print('Validation acc: %.2f%%' %  (100*eval_in_batches(test_data,test_labels,sess)))
