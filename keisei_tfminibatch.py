# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from __future__ import absolute_import

from __future__ import division

from __future__ import print_function



import argparse

import sys



from tensorflow.examples.tutorials.mnist import input_data



import tensorflow as tf

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

from pylab import rcParams



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

filepath = '../input/train.csv'



FLAGS = None

rcParams['figure.figsize'] = 5, 10



def main(_):

    # Import data

    # Any results you write to the current directory are saved as output.

    filename_queue = tf.train.string_input_producer(["../input/train.csv"])



    reader = tf.TextLineReader(skip_header_lines=1)

    key, value = reader.read(filename_queue)



    # Default values, in case of empty columns. Also specifies the type of the

    # decoded result.

    record_defaults = [[0] for _ in range(28**2+1)]

    cols = tf.decode_csv(value, record_defaults=record_defaults)

    target = tf.one_hot(cols[0], 10)

    features = tf.stack(cols[1:])

    

    batch_size = 100

    min_after_dequeue = 10000

    capacity = min_after_dequeue + 3 * batch_size

    

    example_batch, label_batch = tf.train.shuffle_batch(

        [features, target], batch_size=batch_size, capacity=capacity,

        min_after_dequeue=min_after_dequeue)

    

    test_example_batch, test_label_batch = tf.train.shuffle_batch(

        [features, target], batch_size=batch_size*10, capacity=min_after_dequeue + 30 * batch_size,

        min_after_dequeue=min_after_dequeue)



    # Create the model

    x = tf.placeholder(tf.float32, [None, 784])

    W = tf.Variable(tf.zeros([784, 10]))

    b = tf.Variable(tf.zeros([10]))

    y = tf.matmul(x, W) + b



    y_ = tf.placeholder(tf.float32, [None, 10])



    cross_entropy = tf.reduce_mean(

      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

    

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        

    with tf.Session() as sess:

        tf.global_variables_initializer().run()

        

        # Start populating the filename queue.

        coord = tf.train.Coordinator()

        threads = tf.train.start_queue_runners(coord=coord)



        #test_batch_xs, test_batch_ys = sess.run([test_example_batch, test_label_batch])

        test_batch_xs, test_batch_ys = sess.run([example_batch, label_batch])

        

        # Train

        for i in range(501):

            batch_xs, batch_ys = sess.run([example_batch, label_batch])

            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

            if i%50==0:

                fig = plt.figure(figsize=(5,5))

                gs1 = gridspec.GridSpec(10, 10)

                gs1.update(wspace=0, hspace=0)

                for j in range(100):

                    ax = fig.add_subplot(gs1[j])

                    ax.imshow(batch_xs[j].reshape(28,28), cmap='gray')

                    ax.axis('off')

                    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)

                plt.show()

                print("batch ",i," : ",sess.run(accuracy, feed_dict={x: test_batch_xs,

                                          y_: test_batch_ys}))

            

            

        coord.request_stop()

        coord.join(threads)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',

                      help='Directory for storing input data')

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)()