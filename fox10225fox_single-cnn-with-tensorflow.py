import numpy as np

import pandas as pd

import tensorflow as tf
n_input = 784  # MNIST data input (img shape: 28*28)

n_classes = 10  # MNIST total classes (0-9 digits)



validation_size = 2000
train = pd.read_csv('../input/train.csv')

test  = pd.read_csv('../input/test.csv')
print(train.shape)

print(test.shape)
features = (train.ix[:,1:].values).astype('float32')

labels = pd.get_dummies(train.ix[:,0]).astype('float32')
print(features.shape)

print(labels.shape)
# split data into training & validation

valid_features = features[:validation_size]

valid_labels = labels[:validation_size]



train_features = features[validation_size:]

train_labels = labels[validation_size:]
print(train_features.shape)

print(train_labels.shape)

print(valid_features.shape)

print(valid_labels.shape)
test_features = (test.values).astype('float32')
print(test_features.shape)
num_filters = 16



# Features and Labels

features = tf.placeholder(tf.float32, [None, n_input])

labels = tf.placeholder(tf.float32, [None, n_classes])



x_image = tf.reshape(features, [-1,28,28,1])



W_conv = tf.Variable(tf.truncated_normal([5,5,1,num_filters], stddev=0.1))

h_conv = tf.nn.conv2d(x_image, W_conv, strides=[1,1,1,1], padding="SAME")

h_pool = tf.nn.max_pool(h_conv, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
num_units1 = 14*14*num_filters

num_units2 = 1024



h_pool_flat = tf.reshape(h_pool, [-1,num_units1])



# Weights & bias

w1 = tf.Variable(tf.random_normal([num_units1, num_units2]))

b1 = tf.Variable(tf.random_normal([num_units2]))



# Hidden Layer - a(xW1 + b1)

z1 = tf.add(tf.matmul(h_pool_flat, w1), b1)

h1 = tf.nn.relu(z1)



# Weights & bias

w0 = tf.Variable(tf.random_normal([num_units2, n_classes]))

b0 = tf.Variable(tf.random_normal([n_classes]))



# Logits - h1W0 + b0

logits = tf.add(tf.matmul(h1, w0), b0)



# Define loss and optimizer

learning_rate = tf.placeholder(tf.float32)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)



# Calculate accuracy

predict = tf.argmax(logits, 1)

correct_prediction = tf.equal(predict, tf.argmax(labels, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



init = tf.global_variables_initializer()
def print_epoch_stats(epoch_i, sess, last_features, last_labels):

    """

    Print cost and validation accuracy of an epoch

    """

    current_cost = sess.run(

        cost,

        feed_dict={features: last_features, labels: last_labels})

    valid_accuracy = sess.run(

        accuracy,

        feed_dict={features: valid_features, labels: valid_labels})

    print('Epoch: {:<4} - Cost: {:<8.3} Valid Accuracy: {:<5.3}'.format(

        epoch_i,

        current_cost,

        valid_accuracy))
import math

def batches(batch_size, features, labels):

    """

    Create batches of features and labels

    :param batch_size: The batch size

    :param features: List of features

    :param labels: List of labels

    :return: Batches of (Features, Labels)

    """

    assert len(features) == len(labels)

    outout_batches = []

    

    sample_size = len(features)

    for start_i in range(0, sample_size, batch_size):

        end_i = start_i + batch_size

        batch = [features[start_i:end_i], labels[start_i:end_i]]

        outout_batches.append(batch)

        

    return outout_batches
batch_size = 128

epochs = 10

learn_rate = 0.0005
train_batches = batches(batch_size, train_features, train_labels)
with tf.Session() as sess:

    sess.run(init)



    # Training cycle

    for epoch_i in range(epochs):



        # Loop over all batches

        for batch_features, batch_labels in train_batches:

            train_feed_dict = {

                features: batch_features,

                labels: batch_labels,

                learning_rate: learn_rate}

            sess.run(optimizer, feed_dict=train_feed_dict)



        # Print cost and validation accuracy of an epoch

        print_epoch_stats(epoch_i, sess, batch_features, batch_labels)



    predictions = sess.run(

                        predict, 

                        feed_dict={features: test_features})
submissions = pd.DataFrame({"ImageId": list(range(1, len(predictions)+1)),

                             "Label": predictions})

submissions.to_csv("output.csv", index=False, header=True)