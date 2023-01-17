import tensorflow as tf

from vgg16 import vgg16

import numpy as np

import os

from datalab import DataLabTrain
def train(n_iters):

    model, params = vgg16(fine_tune_last=True, n_classes=2)

    X = model['input']

    Z = model['out']

    Y = tf.placeholder(dtype=tf.float32, shape=[None, 2])

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z[:, 0, 0, :], labels=Y))

    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    saver = tf.train.Saver()



    with tf.Session() as sess:

        try:

            sess.run(tf.global_variables_initializer())

            for i in range(n_iters):

                dl = DataLabTrain('./datasets/train_set/')

                train_gen = dl.generator()

                dev_gen = DataLabTrain('./datasets/dev_set/').generator()

                for X_train, Y_train in train_gen:

                    print('Samples seen: '.format(dl.cur_index), end='\r')

                    sess.run(train_step, feed_dict={X: X_train, Y: Y_train})

                print()

                l = 0

                count = 0

                for X_test, Y_test in dev_gen:

                    count += 1

                    l += sess.run(loss, feed_dict={X: X_test, Y: Y_test})



                print('Epoch: {}\tLoss: {}'.format(i, l/count))

                saver.save(sess, './model/vgg16-dog-vs-cat.ckpt')

                print("Model Saved")



        finally:

            sess.close()
train(n_iters=1)
from make_file import make_sub





def predict(model_path, batch_size):

    model, params = vgg16(fine_tune_last=True, n_classes=2)

    X = model['input']

    Y_hat = tf.nn.softmax(model['out'])



    saver = tf.train.Saver()



    dl_test = DataLabTest('./datasets/test_set/')

    test_gen = dl_test.generator()



    Y = []

    with tf.Session() as sess:

        saver.restore(sess, model_path)

        for i in range(12500//batch_size+1):

            y = sess.run(Y_hat, feed_dict={X: next(test_gen)})

            #print(y.shape, end='   ')

            Y.append(y[:,0, 0, 1])

            print('Complete: {}%'.format(round(len(Y) / dl_test.max_len * 100, 2)), end='\r')

    Y = np.concatenate(Y)



    print()

    print('Total Predictions: '.format(Y.shape))

    return Y



Y = predict('./model/vgg16-dog-vs-cat.ckpt', 16)

np.save('out.npy', Y)

make_sub('sub_1.csv')