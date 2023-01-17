import matplotlib.pyplot as plt

from collections import namedtuple

from tensorflow.python.framework import ops

import tensorflow as tf

import numpy as np  # linear algebra

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import sys

import os



sys.path.insert(0, os.path.abspath('../input/'))

import utils



print('This is how the generator works : ')

print(help(utils.dataset_gen))
def selu(x):

    with ops.name_scope('elu') as scope:

        alpha = 1.6732632423543772848170429916717

        scale = 1.0507009873554804934193349852946

        return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))



    

def my_conv_layer(_input):

    '''This is my main conv layer

    '''

    layer = tf.layers.conv1d(

        _input,

        filters=10,

        kernel_size=3,

        strides=2,

        padding='same',

        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(

            factor=1.,

            mode='FAN_AVG',

            uniform=False),

        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),

        activation=selu)

    return layer



    

def build_neural_network():

    '''We build a 10 layer convolutional neural network ending with

    a Global average Pooling layer whose logits are submitted to

    a sigmoid function.



    Receptive fied is : U_0 = 1; U_{n+1} = U_n * 2 + 1

    Which is, for 10 layers : 2047

    '''

    tf.reset_default_graph()



    # Define placeholders (values to be fed)

    inputs = tf.placeholder(tf.float32, shape=[None, None, 1])

    labels = tf.placeholder(tf.float32, shape=[None, 1])

    learning_rate = tf.placeholder(tf.float32)

    is_training = tf.Variable(True, dtype=tf.bool)



    # Build the actual neural network here below

    nn = my_conv_layer(inputs)

    for _ in range(9):

        nn = my_conv_layer(nn)



    # Global average pooling

    nn = tf.reduce_mean(nn, [1])  # These should be the logits



    # Define the loss function and its optimizer

    logits = tf.layers.dense(nn, 1, activation=None)

    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,

                                                            logits=logits)

    cost = tf.reduce_mean(cross_entropy)



    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):

        optimizer = tf.train.AdamOptimizer(

            learning_rate=learning_rate).minimize(cost)



    # Define the accuracy

    predicted = tf.nn.sigmoid(logits)

    correct_pred = tf.equal(tf.round(predicted), labels)

    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



    # Export the nodes

    export_nodes = ['inputs', 'labels', 'learning_rate', 'is_training',

                    'logits', 'cost', 'optimizer', 'predicted', 'accuracy']

    Graph = namedtuple('Graph', export_nodes)

    local_dict = locals()

    graph = Graph(*[local_dict[each] for each in export_nodes])



    return graph
def perform_epoch(model,

                  sess,

                  is_train=True,

                  lr_value=0.01,

                  batch_shape=(20, 16000),

                  sample_augmentation=0):

    '''Perform one epoch, either train or valid'''

    epoch_loss = []

    epoch_acc = []



    # Loop through epoch samples (batchs)

    for batch_x, batch_y in utils.dataset_gen(

            is_train=is_train,

            sample_augmentation=sample_augmentation):



        # Do the training

        batch_loss, _, batch_acc = sess.run(

            [model.cost, model.optimizer, model.accuracy],

            feed_dict={

                model.inputs: batch_x,

                model.labels: batch_y,

                model.learning_rate: lr_value,

                model.is_training: is_train

            })



        # Accumulate the resulting values

        epoch_loss.append(batch_loss)

        epoch_acc.append(batch_acc)



    return np.array(epoch_loss).mean(), np.array(epoch_acc).mean()
epochs = 100

lr_value = 0.005

collector = {k: [] for k in ['X',

                             'train_loss',

                             'train_acc',

                             'valid_loss',

                             'valid_acc']}



model = build_neural_network()

saver = tf.train.Saver()

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    iteration = 0

    for e in range(epochs):



        # TRAIN on one epoch and collect epoch losses and accuracies

        epoch_loss, epoch_acc = perform_epoch(model, sess, True, lr_value)

        lr_value = lr_value * 0.98

        collector['X'].append(e)

        collector['train_loss'].append(epoch_loss)

        collector['train_acc'].append(epoch_acc)



        # VALIDATION and collect results

        epoch_loss, epoch_acc = perform_epoch(model,

                                              sess,

                                              False,

                                              0,

                                              (20,16000 * 1))

        collector['valid_loss'].append(epoch_loss)

        collector['valid_acc'].append(epoch_acc)



        # Do some printing

        print(("At Epoch {}/{} Train is {:.3f} & {:.2f}% and Valid is " +

              "{:.3f} & {:.2f}%").format(e + 1,

                                       epochs,

                                       collector['train_loss'][-1],

                                       collector['train_acc'][-1] * 100,

                                       collector['valid_loss'][-1],

                                       collector['valid_acc'][-1] * 100))



    saver.save(sess, "./cats_dogs.ckpt")

fig, axs = plt.subplots(1, 2, figsize=(15, 7))

axs[0].plot(collector['X'], collector['train_loss'], "r--")

axs[0].plot(collector['X'], collector['valid_loss'], "g--")

axs[0].set_title('Loss')

axs[1].plot(collector['X'], collector['train_acc'], "r--")

axs[1].plot(collector['X'], collector['valid_acc'], "g--")

axs[1].set_title('Accuracy')



plt.show()