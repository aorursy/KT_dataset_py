import pandas as pd

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

from matplotlib import pyplot as plt
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
x = tf.placeholder(tf.float32, shape=(None, 784), name='x')

y_ = tf.placeholder(tf.float32, shape=(None, 10), name='y_')



x_image = tf.reshape(x, [-1, 28, 28, 1], name='x_image')

print(x_image.shape, y_.shape)
def weight_variable(shape):

    initial = tf.truncated_normal(shape, stddev=0.1)

    return tf.Variable(initial)

def bias_variable(shape):

    initial = tf.constant(0.1, shape=shape)

    return tf.Variable(initial)

def conv2d(x, W, strides=[1,1,1,1], padding='SAME'):

    return tf.nn.conv2d(x, W, strides=strides, padding=padding)

def maxpool2d(x):

    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding='VALID')
W1 = weight_variable([5,5,1,32])

b1 = bias_variable([32]) # num of channels



conv1 = tf.nn.relu(conv2d(x_image, W1) + b1) 

maxpool1 = maxpool2d(conv1) # fig



W2 = weight_variable([5,5,32,64])

b2 = bias_variable([64])



conv2 = tf.nn.relu(conv2d(maxpool1, W2) + b2) 

maxpool2 = maxpool2d(conv2)



W3 = weight_variable([7*7*64, 1024])

b3 = weight_variable([1024])



maxpool2_flat = tf.reshape(maxpool2, [-1, 7*7*64])

fc1 = tf.nn.relu(tf.matmul(maxpool2_flat, W3) + b3)



keep_prob = tf.placeholder(tf.float32)

dropout = tf.nn.dropout(fc1, keep_prob)
W4 = weight_variable([1024,10])

b4 = bias_variable([10])



y_conv = tf.matmul(dropout, W4) + b4



sess = tf.InteractiveSession()

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)



correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



sess.run(tf.global_variables_initializer())



import time



start = time.time()

end = time.time()



num_steps = 2000

display_every = 200



for i in range(num_steps):

    batch = mnist.train.next_batch(64)

#     for j in range(len(batch)):

#         print('batch[{0}] => {1}\n'.format(j, batch[j].shape))

    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    

    if i % display_every == 0:

        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1})

        end = time.time()

        

        print('step {0}  elapsed time {1:.2f} seconds, training accuracy: {2:.3f}'.format(i, end - start, train_accuracy))

end = time.time()



print('Training time for {0} batches: {1:.2f} seconds'.format(i, end - start))

print('Test Accuracy: ', accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1}))





# sess.close()
import numpy as np

n = np.random.randint(mnist.test.images.shape[1]) # test img index

test_img = mnist.test.images[n].reshape([1, 28 * 28])

# test_img.shape

plt.imshow(test_img[0].reshape([28,28]))



classification = sess.run(tf.argmax(y_conv, 1), feed_dict={x: test_img, keep_prob:1.0})

classification
sess.close()