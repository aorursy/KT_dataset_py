from __future__ import absolute_import

from __future__ import division

from __future__ import print_function



# Import data

from tensorflow.examples.tutorials.mnist import input_data



import tensorflow as tf
flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')



mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)



sess = tf.InteractiveSession()
# Create the model

HIDDEN_1 = 25

HIDDEN_2 = 25



x = tf.placeholder(tf.float32, [None, 784])



W0 = tf.Variable(tf.zeros([784, HIDDEN_1]))

b0 = tf.Variable(tf.zeros([HIDDEN_1]))



W1 = tf.Variable(tf.zeros([HIDDEN_1, HIDDEN_2]))

b1 = tf.Variable(tf.zeros([HIDDEN_2]))



W2 = tf.Variable(tf.zeros([HIDDEN_2, 10]))

b2 = tf.Variable(tf.zeros([10]))



h1 = tf.sigmoid(tf.matmul(x, W0) + b0)

h2 = tf.sigmoid(tf.matmul(h1, W1) + b1)

y = tf.nn.softmax(tf.matmul(h2, W2) + b2)
# Define loss and optimizer

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y,1e-10,1.0)), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# Train

tf.initialize_all_variables().run()

for i in range(1000):

    batch_xs, batch_ys = mnist.train.next_batch(100)

    train_step.run({x: batch_xs, y_: batch_ys})
# Test trained model

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels})*100,'%')