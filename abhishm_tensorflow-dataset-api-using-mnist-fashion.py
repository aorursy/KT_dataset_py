import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
import tensorflow as tf

from tensorflow.contrib.data import Dataset

from tensorflow.contrib.data import Iterator
path = "../input/fashion-mnist_train.csv"
def get_data_gen():

    train_fraction = 0.3

    defaults_values = [[0] for _ in range(785)]

    

    def decode_line(line):

        """Convert a csv line into a (features_dict,label) pair."""

        # Decode the line to a tuple of items based on the types of

        # csv_header.values().

        items = tf.decode_csv(line, defaults_values)

        return items[1:], items[0]

    

    def in_training_set(line):

        """Returns a boolean tensor, true if the line is in the training set."""

        # If you randomly split the dataset you won't get the same split in both

        # sessions if you stop and restart training later. Also a simple

        # random split won't work with a dataset that's too big to `.cache()` as

        # we are doing here.

        num_buckets = 1000000

        bucket_id = tf.string_to_hash_bucket_fast(line, num_buckets)

        # Use the hash bucket id as a random number that's deterministic per example

        return bucket_id < int(train_fraction * num_buckets)

    

    def in_test_set(line):

        """Returns a boolean tensor, true if the line is in the training set."""

        # Items not in the training set are in the test set.

        # This line must use `~` instead of `not` beacuse `not` only works on python

        # booleans but we are dealing with symbolic tensors.

        return ~in_training_set(line)

    

    def normalize(features, labels):

        """Returns a normalized feature array between -1 and 1"""

        return 2 * features / 255 - 1, labels 

    

    base_dataset = (tf.data

              # Get the lines from the file.

              .TextLineDataset(path)

              .skip(1))

    

    train = (base_dataset

       # Take only the training-set lines.

       .filter(in_training_set)

       # Cache data so you only read the file once.

       .cache()

       # Decode each line into a (features, label) pair.

       .map(decode_line))

    

    # Do the same for the test-set.

    test = base_dataset.filter(in_test_set).cache().map(decode_line)

    

    # Bring the features between -1 and 1

    train = train.map(normalize)

    test = test.map(normalize)

    

    return train, test
def create_iterator():

    train, test = get_data_gen()

    train_batch = train.shuffle(1000).batch(32).repeat() # Repeat forever

    test_batch = test.batch(32)

    

    iterator = Iterator.from_structure(train.output_types)

    next_elem = iterator.get_next()

    training_init_op = iterator.make_initializer(train_batch)

    validation_init_op = iterator.make_initializer(test_batch)

    

    return next_elem, training_init_op, validation_init_op
def create_model(inputs, is_training):

    x_ = tf.reshape(inputs, (tf.shape(inputs)[0], 28, 28, 1))

    conv_1 = tf.layers.conv2d(x_, 32, (5, 5), name="layer_1", strides=(1, 1), activation=None)

    batch_norm_1 = tf.nn.relu(tf.layers.batch_normalization(conv_1, training=is_training))

    pool_1 = tf.layers.max_pooling2d(batch_norm_1, (2, 2), strides=(2, 2))

    conv_2 = tf.layers.conv2d(pool_1, 32, (5, 5), name="layer_2", strides=(1, 1), activation=None)

    batch_norm_2 = tf.nn.relu(tf.layers.batch_normalization(conv_2, training=is_training))

    pool_2 = tf.layers.max_pooling2d(batch_norm_2, (2, 2), strides=(2, 2))

    reshape_w2 = tf.reshape(pool_2, (tf.shape(x_)[0], 16 * 32))

    fc1 = tf.layers.dense(reshape_w2, 128, activation=tf.nn.relu)

    logits = tf.layers.dense(fc1, 10)

    return logits

    



def create_accuracy(logits, outputs):

    preds = tf.argmax(logits, axis=-1, output_type=tf.int32)

    correct_preds = tf.cast(tf.equal(preds, outputs), tf.float16)

    accuracy = tf.reduce_mean(correct_preds)

    return accuracy 





def create_loss(logits, outputs, num_labels=10):

    y_ = tf.one_hot(outputs, num_labels)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_))

    return loss





def create_optimizer(loss, learning_rate):

    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")    

    optimizer = tf.train.AdamOptimizer(learning_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):

        train_op = optimizer.minimize(loss, global_step)

    return train_op, global_step

        



def create_graph(learning_rate=0.001):

    phase = tf.placeholder(tf.bool, shape=(), name="phase")

    

    next_elem, training_init_op, validation_init_op = create_iterator()

    x, y = tf.cast(next_elem[0], tf.float32), next_elem[1]

    

    logits = create_model(x, phase)

    accuracy = create_accuracy(logits, y)

    loss = create_loss(logits, y)

    train_op, global_step = create_optimizer(loss, learning_rate)

    

    return {"train_op": train_op, 

            "accuracy": accuracy, 

            "loss":loss, 

            "training_init_op": training_init_op, 

            "validation_init_op": validation_init_op,

            "phase": phase}
def print_results(iteration, losses, accuracies):

    print("iteration: {0:5d} loss: {1:0.3f}, accuracy: {2:0.3f}"

      .format(iteration, np.mean(losses), np.mean(accuracies)))
def plot(losses, accuracies, smoothen_coef=10):

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

    ax0.plot(np.correlate(losses, np.ones(smoothen_coef) / smoothen_coef))

    ax0.set(xlabel="time", ylabel="loss")

    ax1.plot(np.correlate(accuracies, np.ones(smoothen_coef) / smoothen_coef))

    ax1.set(xlabel="time", ylabel="accuracy")
def run_a_grpah(validation_period=1000):

    tf.reset_default_graph()

    ops = create_graph()

    training_losses = []

    training_accuracies = []

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        sess.run(ops["training_init_op"])

        for i in range(10000):

            

            _, loss_, acc = sess.run([ops["train_op"], ops["loss"], ops["accuracy"]], 

                                     feed_dict={ops["phase"]: True})

            training_losses.append(loss_)

            training_accuracies.append(acc)

            if i % validation_period == 0:

                print_results(i, training_losses, training_accuracies)

                

                sess.run(ops["validation_init_op"])



                testing_losses = []

                testing_accuracies = []

                for j in range(100):

                    loss_, acc = sess.run([ops["loss"], ops["accuracy"]], 

                                         feed_dict={ops["phase"]: False})

                    testing_losses.append(loss_)

                    testing_accuracies.append(acc)

                print_results(i, testing_losses, testing_accuracies)



                sess.run(ops["training_init_op"])

            

    return training_losses, training_accuracies

                
loss, accuracy = run_a_grpah()
plot(loss, accuracy)