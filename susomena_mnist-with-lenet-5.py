from __future__ import print_function

from __future__ import division



import numpy as np

import tensorflow as tf

import tensorflow.contrib.slim as slim

import matplotlib.pyplot as plt



import pandas
train_data = np.array(pandas.read_csv('../input/train.csv'))

test_data = np.array(pandas.read_csv('../input/test.csv'))



valid_set_size = 8400



train_set = (np.reshape(train_data[:, 1:], (train_data.shape[0], 28, 28, 1)) - 128.0) / 128.0

train_labels = train_data[:, 0]

test_set = (np.reshape(test_data, (test_data.shape[0], 28, 28, 1)) - 128.0) / 128.0



valid_set = train_set[:-valid_set_size]

valid_labels = train_labels[:-valid_set_size]

train_set = train_set[-valid_set_size:]

train_labels = train_labels[-valid_set_size:]



del train_data, test_data  # We no longer need them, let's help the GC to free memory



print('Training set', train_set.shape, train_labels.shape)

print('Validation set', valid_set.shape, valid_labels.shape)

print('Test set', test_set.shape)
batch_size = 50

initial_learning_rate = 0.1



print('Building model...')



graph = tf.Graph()

with graph.as_default():

    # Input data

    tf_train_set = tf.placeholder(dtype=tf.float32, shape=[batch_size, 28, 28, 1])

    tf_train_labels = tf.placeholder(dtype=tf.int32, shape=[batch_size])

    tf_valid_set = tf.placeholder(dtype=tf.float32, shape=[batch_size, 28, 28, 1])

    tf_test_set = tf.placeholder(dtype=tf.float32, shape=[batch_size, 28, 28, 1])



    # Model (Based on LeNet-5)

    def model(x, is_training=False, reuse=None):

        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.batch_norm], reuse=reuse):

            net = slim.conv2d(x, 6, [5, 5], [1, 1], padding='SAME', activation_fn=None, scope='C1')  # 28x28x6

            net = slim.max_pool2d(net, [2, 2], [2, 2], scope='S2')  # 14x14x6

            net = slim.conv2d(net, 16, [5, 5], [1, 1], padding='VALID', activation_fn=None, scope='C3')  # 10x10x16

            net = slim.max_pool2d(net, [2, 2], [2, 2], scope='S4')  # 5x5x16

            net = slim.conv2d(net, 120, [5, 5], [1, 1], padding='VALID', scope='C5')  # 1x1x120

            net = slim.flatten(net)

            net = slim.fully_connected(net, 84, scope='F6')

            net = slim.dropout(net, keep_prob=0.5, is_training=is_training, scope='DROPOUT')

            out = slim.fully_connected(net, 10, activation_fn=None, scope='OUTPUT')



        return out



    # Loss computation

    logits = model(tf_train_set, is_training=True)

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))



    # Training op

    global_step = tf.Variable(0)

    learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, 2000, 0.5, staircase=True)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)



    # Training, validation and test predictions

    train_prediction = tf.nn.softmax(logits)

    valid_prediction = tf.nn.softmax(model(tf_valid_set, reuse=True))

    test_prediction = tf.nn.softmax(model(tf_test_set, reuse=True))
def accuracy(predictions, labels):

    return 100.0 * np.sum(np.argmax(predictions, 1) == labels) / predictions.shape[0]
num_steps = 8001



with tf.Session(graph=graph) as session:

    tf.global_variables_initializer().run()

    print('Initialized!')



    train_loss = []

    train_accuracy = []

    validation_accuracy = []



    for step in range(num_steps):

        offset = (step * batch_size) % train_labels.shape[0]

        batch_data = train_set[offset:offset + batch_size, :, :, :]

        batch_labels = train_labels[offset:offset + batch_size]

        feed_dict = {tf_train_set: batch_data, tf_train_labels: batch_labels}



        _, l, prediction = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)



        if step % 100 == 0:

            acc = accuracy(prediction, batch_labels)

            train_loss.append(l)

            train_accuracy.append(acc)

            print('Minibatch loss at step %d: %f' % (step, l))

            print('Training accuracy: %.2f%%' % acc)



            vp = []

            for k in range(valid_set.shape[0] // batch_size):

                vp.append(session.run(

                        valid_prediction, feed_dict={tf_valid_set: valid_set[k * batch_size:(k + 1) * batch_size]}))

            v_acc = accuracy(np.vstack(vp), valid_labels)

            validation_accuracy.append(v_acc)

            print('Validation accuracy: %.2f%%' % v_acc)



        if step % (train_set.shape[0] / batch_size) == 0:

            # Shuffle train set every epoch to avoid correlation bias

            permutation = np.random.permutation(train_labels.shape[0])

            train_set = train_set[permutation]

            train_labels = train_labels[permutation]



    tp = []

    for k in range(test_set.shape[0] // batch_size):

        tp.append(session.run(test_prediction, feed_dict={tf_test_set: test_set[k * batch_size:(k + 1) * batch_size]}))



    submission_predictions = np.argmax(np.vstack(tp), 1)

    np.savetxt('submission.csv', np.c_[range(1, submission_predictions.shape[0] + 1), submission_predictions], 

               delimiter=',', header='ImageId,Label', comments='', fmt='%d')



    fig, (plt1, plt2) = plt.subplots(1, 2)



    plt1.plot(range(0, num_steps, 100), train_loss)

    plt1.set_title('Training loss')

    plt1.set_xlabel('Iteration')

    plt1.set_ylabel('Loss')



    plt2.plot(range(0, num_steps, 100), train_accuracy, 'b', label='Training accuracy')

    plt2.plot(range(0, num_steps, 100), validation_accuracy, 'r', label='Validation accuracy')

    plt2.legend(loc='lower right')

    plt2.set_title('Accuracy')

    plt2.set_xlabel('Iteration')

    plt2.set_ylabel('Accuracy [%]')



    plt.show()