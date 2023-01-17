import pandas as pd

import numpy as np
data = pd.read_csv('../input/train.csv')

print('data.shape: ({0[0]}, {0[1]})'.format(data.shape))
test = pd.read_csv('../input/test.csv')

print('test.shape: ({0[0]}, {0[1]})'.format(test.shape))
images = data.iloc[:, 1:].values

print('images.shape: ({0[0]}, {0[1]})'.format(images.shape))
images = images.astype(np.float32)

images = np.multiply(images, 1.0 / 255.0)
test = test.astype(np.float32)

test = np.multiply(test.as_matrix(), 1.0 / 255.0)
labels_flat = data.iloc[:, 0].values.ravel()

print('labels_flat.len: {0}'.format(len(labels_flat)))
labels_count = np.unique(labels_flat).shape[0]

print('labels_count: {0}'.format(labels_count))
def dense_to_one_hot(labels_dense, num_classes):

    num_labels = labels_dense.shape[0]

    index_offset = np.arange(num_labels) * num_classes

    labels_one_hot = np.zeros((num_labels, num_classes))

    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot
labels = dense_to_one_hot(labels_flat, labels_count)

labels = labels.astype(np.uint8)

print('labels.shape: ({0[0]}, {0[1]})'.format(labels.shape))
import tensorflow as tf



x  = tf.placeholder(tf.float32, shape=[None, 784])

y_ = tf.placeholder(tf.float32, shape=[None, 10])
def weight_variables(shape):

    initial = tf.truncated_normal(shape, stddev = 0.1)

    return tf.Variable(initial)



def bias_variable(shape):

    initial = tf.constant(0.1, shape = shape)

    return tf.Variable(initial)
def conv2d(x, W):

        return tf.nn.conv2d(x, W, strides =[1, 1, 1, 1], padding = 'SAME')

    

def max_pool_2x2(x):

    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], 

                         strides = [1, 2, 2, 1], padding = 'SAME')
W_conv1 = weight_variables([5, 5, 1, 32])

b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
print('h_conv1.shape: %s' % h_conv1.shape)
h_pool1 = max_pool_2x2(h_conv1)
print('h_pool.shape: %s' % h_pool1.shape)
W_conv2 = weight_variables([5, 5, 32, 64])

b_conv2 = bias_variable([64])



h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

h_pool2 = max_pool_2x2(h_conv2)
print('h_conv2.shape: %s' % h_conv2.shape)

print('h_pool2.shape: %s' % h_pool2.shape)
W_fc1 = weight_variables([7*7*64, 1024])

b_fc1 = bias_variable([1024])



h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
print('h_pool2_flat.shape: %s' % h_pool2_flat.shape)
keep_prob = tf.placeholder(tf.float32)

h_fcl_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = weight_variables([1024, 10])

b_fc2 = bias_variable([10])



y_conv = tf.matmul(h_fcl_drop, W_fc2) + b_fc2

prediction = tf.argmax(y_conv, 1)
VALIDATION_SET_SIZE = 4000



BATCH_SIZE = 100

TRAINING_ITERATIONS = 12000



LEARNING_RATE = 1e-3

EPSILON = 1e-8



KEEP_RATE = 0.5
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= y_, logits = y_conv))

train_step = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE, epsilon = EPSILON).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
import random

partitions = [0] * data.shape[0]

partitions[:VALIDATION_SET_SIZE] = [1] * VALIDATION_SET_SIZE

random.shuffle(partitions)
images_placeholder = tf.placeholder(images.dtype, images.shape)

labels_placeholder = tf.placeholder(labels.dtype, labels.shape)
training_images, validation_images = tf.dynamic_partition(images_placeholder, partitions = partitions, num_partitions = 2)

training_labels, validation_labels = tf.dynamic_partition(labels_placeholder, partitions = partitions, num_partitions = 2)
training_dataset = tf.data.Dataset.from_tensor_slices((training_images, training_labels))

training_dataset = training_dataset.shuffle(buffer_size = 10000)

training_dataset = training_dataset.batch(BATCH_SIZE)

training_iterator = training_dataset.make_initializable_iterator()

next_batch = training_iterator.get_next() 
validation_dataset = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))

validation_dataset = validation_dataset.batch(VALIDATION_SET_SIZE)

validation_iterator = validation_dataset.make_initializable_iterator()

validation_data = validation_iterator.get_next()
display_step = 1



with tf.Session() as sess:

    sess.run(training_iterator.initializer, feed_dict = {images_placeholder: images,

                                                         labels_placeholder: labels})

    sess.run(tf.global_variables_initializer())

    for i in range(TRAINING_ITERATIONS):

        try:

            batch = sess.run(next_batch)

            if i % display_step == 0 or (i + 1) == TRAINING_ITERATIONS:

                train_accuracy = accuracy.eval(feed_dict={x: batch[0],

                                                          y_: batch[1], 

                                                          keep_prob: 1.0 })

                sess.run(validation_iterator.initializer, feed_dict = {images_placeholder: images, 

                                                                       labels_placeholder: labels})

                validation = sess.run(validation_data)

                validation_accuracy = accuracy.eval(feed_dict = {x: validation[0], 

                                                                 y_: validation[1], 

                                                                 keep_prob: 1.0})

                print('step %d. Training accuracy %g. Validation accuracy %g.' % (i, train_accuracy, validation_accuracy))

            train_step.run(feed_dict = {x: batch[0], 

                                        y_: batch[1],

                                        keep_prob: KEEP_RATE})

        

            # increase display_step

            if i%(display_step*2) == 0 and i < 512:

                display_step *= 2

        

        except tf.errors.OutOfRangeError:

            print('End of Epoch. Initializing iterator.')

            sess.run(training_iterator.initializer, feed_dict = {images_placeholder: images,

                                                         labels_placeholder: labels})