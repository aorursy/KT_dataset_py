from __future__ import print_function

import numpy as np

import tensorflow as tf

from six.moves import cPickle as pickle

from six.moves import range

import matplotlib.pyplot as plt

import pandas as pd

from keras.utils.np_utils import to_categorical
train_df = pd.read_csv("../input/train.csv")

test_dataset = pd.read_csv("../input/test.csv")

train_dataset = train_df.drop(['label'], axis=1).values.astype('float32')

train_labels = train_df['label'].values

test_dataset = test_dataset.values.astype('float32')

print(train_dataset.shape)

print(train_labels.shape)

#print(test_dataset.shape)
image_size = 28

num_labels = 10

num_channels = 1 # grayscale



import numpy as np



def reformat(dataset):

  dataset = dataset.reshape(

    (-1, image_size, image_size, num_channels)).astype(np.float32)

  return dataset

train_dataset = reformat(train_dataset)

#valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)

test_dataset = reformat(test_dataset)

train_labels = to_categorical(train_labels)

print('Training set', train_dataset.shape)

print (train_labels.shape)

#print('Validation set', valid_dataset.shape, valid_labels.shape)

print('Test set', test_dataset.shape)
def accuracy(predictions, labels):

  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))

          / predictions.shape[0])
batch_size = 16

patch_size = 3

depth = 16

num_hidden = 64

graph = tf.Graph()

with graph.as_default():

    model_train_data = tf.placeholder(tf.float32,shape=[batch_size,image_size,image_size,num_channels])

    model_train_labels = tf.placeholder(tf.float32,shape=[batch_size,num_labels])

    #model_validation_data = tf.constant(valid_dataset)

    model_test_data = tf.constant(test_dataset)

    test_image = tf.placeholder(tf.float32,shape=[1,image_size,image_size,num_channels])

    #layer matrices

    #conv layer1

    layer1_weights = tf.Variable(tf.truncated_normal([patch_size,patch_size,num_channels,depth],stddev=0.1),name='layer1_w')

    layer1_bias = tf.Variable(tf.zeros([depth]),name='layer1_b')

    #conv layer2

    layer2_weights = tf.Variable(tf.truncated_normal([patch_size,patch_size,depth,depth],stddev=0.1),name='layer2_w')

    layer2_bias = tf.Variable(tf.constant(1.0,shape=[depth]),name='layer2_b')

    

    #fully hidden1

    shape = ((image_size-patch_size+1)//2 - patch_size +1)//2

    layer3_weights = tf.Variable(tf.truncated_normal([shape*shape*depth,num_hidden],stddev=0.1),name='layer3_w')

    layer3_bias = tf.Variable(tf.constant(1.0,shape=[num_hidden]),name='layer3_b')

    #output

    layer4_weights = tf.Variable(tf.truncated_normal([num_hidden,num_labels],stddev=0.1),name='layer4_w')

    layer4_bias = tf.Variable(tf.constant(1.0,shape=[num_labels]),name='layer4_b')

    

    def model(data):

        with tf.name_scope('layer1'):

            with tf.name_scope('weights'):

                conv1 = tf.nn.conv2d(data,layer1_weights,[1, 1, 1, 1],padding='VALID')

            with tf.name_scope('biases'):

                bias1 = tf.nn.relu(conv1+layer1_bias)



        #pooling 1

        with tf.name_scope('pool1'):

            pool1 = tf.nn.avg_pool(bias1,[1, 2, 2, 1],[1, 2, 2, 1],padding='VALID')

        with tf.name_scope('layer2'):

            with tf.name_scope('weights'):

                conv2 = tf.nn.conv2d(pool1,layer2_weights,[1, 1, 1, 1],padding='VALID')

            with tf.name_scope('biases'):

                bias2 = tf.nn.relu(conv2 + layer2_bias)

        #pooing 2

        

        with tf.name_scope('pool2'):

            pool2 = tf.nn.avg_pool(bias2,[1, 2, 2, 1],[1, 2, 2, 1],padding='VALID')

        

        shape = pool2.get_shape().as_list()

        reshape = tf.reshape(pool2, [shape[0], shape[1] * shape[2] * shape[3]])

        with tf.name_scope('layer3'):

            hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_bias)

        return tf.matmul(hidden, layer4_weights) + layer4_bias

    with tf.name_scope('logits'):

        logit = model(model_train_data)

    with tf.name_scope('loss'):

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit,labels=model_train_labels))

    #optmizer

    with tf.name_scope('train'):

        optimizer = tf.train.AdamOptimizer(0.03).minimize(loss)

    #tf.histogram_summary("weights1", layer1_weights)

    #tf.histogram_summary("biases1", layer1_bias)

    #tf.histogram_summary("weights2", layer2_weights)

    #tf.histogram_summary("biases2", layer2_bias)

    #tf.histogram_summary("weights3", layer3_weights)

    #tf.histogram_summary("biases3", layer3_bias)

    #tf.histogram_summary("weights4", layer4_weights)

    #tf.histogram_summary("biases4", layer4_bias)

    #tf.scalar_summary('loss',loss)

    #summarizer = tf.merge_all_summaries()

    train_prediction = tf.nn.softmax(logit)

    #valid_prediction = tf.nn.softmax(model(model_validation_data))

    test_prediction = tf.nn.softmax(model(model_test_data))

    image_prediction = tf.nn.softmax(model(test_image))
num_steps = 1201



with tf.Session(graph=graph) as session:

    tf.initialize_all_variables().run()

    print('Initialized')

    saver = tf.train.Saver()

    for step in range(num_steps):

        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)

        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]

        batch_labels = train_labels[offset:(offset + batch_size), :]

        feed_dict = {model_train_data : batch_data, model_train_labels : batch_labels}

        _, l, predictions= session.run(

          [optimizer, loss, train_prediction], feed_dict=feed_dict)

        if (step % 200 == 0):

            print('Minibatch loss at step %d: %f' % (step, l))

            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))

            #train_writer.add_summary(summary, step)

#     saver.save(session, "./model.ckpt")

    np.savetxt('mnist_output.csv', np.c_[range(1,len(test_dataset)+1),np.argmax(test_prediction.eval(), 1)], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')