import tensorflow as tf

import pandas as pd

import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt

from tqdm import tqdm

%matplotlib inline
data_path = './data'

train = pd.read_json(data_path + '/' + 'train.json')

test = pd.read_json(data_path + '/' + 'test.json')

submission = pd.read_csv(data_path + '/' + 'sample_submission.csv').set_index('id')
train_band_1 = np.array([np.array(band).astype(np.float32).reshape((75,75)) for band in train.band_1])

train_band_2 = np.array([np.array(band).astype(np.float32).reshape((75,75)) for band in train.band_2])

X_train = np.concatenate([train_band_1[:, :, :, np.newaxis], 

                          train_band_2[:, :, :, np.newaxis],

                          ((train_band_1+train_band_2)/2)[:, :, :, np.newaxis]], axis=-1)



test_band_1 = np.array([np.array(band).astype(np.float32).reshape((75,75)) for band in test.band_1])

test_band_2 = np.array([np.array(band).astype(np.float32).reshape((75,75)) for band in test.band_2])

X_test = np.concatenate([test_band_1[:, :, :, np.newaxis], 

                         test_band_2[:, :, :, np.newaxis],

                         ((test_band_1+test_band_2)/2)[:, :, :, np.newaxis]], axis=-1)



y = np.array([target for target in train.is_iceberg]).reshape((-1,1))



print ('train_band_1 shape: {}'.format(train_band_1.shape))

print ('train_band_2 shape: {}'.format(train_band_2.shape))

print ('train_train shape:  {}'.format(X_train.shape))

print ('train label shape:  {}'.format(y.shape))

print ('test_band_1 shape:  {}'.format(test_band_1.shape))

print ('test_band_2 shape:  {}'.format(test_band_2.shape))

print ('test_train shape:   {}'.format(X_test.shape))
lbl = OneHotEncoder()

lbl.fit([[0],[1]])

y = lbl.transform(y).toarray()
X_min = np.min(X_train)

X_max = np.max(X_train)

X_train = (X_train - X_min)/(X_max - X_min)

X_test = (X_test - X_min)/(X_max - X_min)
fig, ax = plt.subplots(2,4, figsize=[12,8])



ax[0,0].imshow(X_train[0,:,:,0])

ax[0,1].imshow(X_train[0,:,:,2])



ax[0,2].imshow(X_train[2,:,:,0])

ax[0,3].imshow(X_train[2,:,:,2])



ax[1,0].imshow(X_train[1,:,:,0])

ax[1,1].imshow(X_train[1,:,:,2])



ax[1,2].imshow(X_train[6,:,:,0])

ax[1,3].imshow(X_train[6,:,:,2])
def get_batches(x, y, batch_size=10):

    n_batches = len(x)//batch_size

    for ii in range(0, n_batches*batch_size, batch_size):

        if ii != (n_batches-1)*batch_size:

            X, Y = x[ii: ii+batch_size], y[ii: ii+batch_size] 

        else:

            X, Y = x[ii:], y[ii:]

        yield X, Y
train_data, train_label = X_train[:1400,:,:,:], y[:1400,:]

val_data, val_label = X_train[1400:,:,:,:], y[1400:,:]
inputs = tf.placeholder(tf.float32, [None, 75, 75, 3])

labels = tf.placeholder(tf.int32)



conv1 = tf.layers.conv2d(inputs=inputs, filters=8, kernel_size=(7,7), strides=(1,1), 

                         padding='SAME', activation=tf.nn.relu, use_bias=True)

pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=(2,2), strides=(2,2), padding='SAME')



conv2 = tf.layers.conv2d(inputs=pool1, filters=16, kernel_size=(5,5), strides=(1,1), 

                         padding='SAME', activation=tf.nn.relu, use_bias=True)

pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=(2,2), strides=(2,2), padding='SAME')



conv3 = tf.layers.conv2d(inputs=pool2, filters=16, kernel_size=(3,3), strides=(1,1), 

                         padding='SAME', activation=tf.nn.relu, use_bias=True)

pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=(2,2), strides=(2,2), padding='SAME')



flat = tf.reshape(pool3, [-1,1600])



fc1 = tf.layers.dense(flat, units=256, use_bias=True, activation=tf.nn.relu)

dp1 = tf.layers.dropout(fc1, rate=0.25)



fc2 = tf.layers.dense(dp1, units=64, use_bias=True, activation=tf.nn.relu)

dp2 = tf.layers.dropout(fc2, rate=0.25)



logits = tf.layers.dense(dp2, units=2, use_bias=True)



loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.cast(labels, tf.float32))

cost = tf.reduce_mean(loss)



predicted = tf.nn.softmax(logits)

correct_pred = tf.equal(tf.argmax(predicted, 1), tf.argmax(labels, 1))

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)



n_epoches = 100

batch_size = 32



saver = tf.train.Saver()



with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for epoch in range(n_epoches):

        for X_batch, y_batch in get_batches(train_data, train_label, batch_size):

            feed_dict = {inputs:X_batch, labels:y_batch}

            train_cost,_ = sess.run([cost, optimizer], feed_dict=feed_dict)

            

        feed_dict = {inputs:X_train, labels:y}

        train_accuracy = sess.run(accuracy, feed_dict=feed_dict)

        feed_dict = {inputs:val_data, labels:val_label}

        val_accuracy = sess.run(accuracy, feed_dict=feed_dict)

        print('epoch {}, train accuracy: {:5f}, validation accuracy: {:.5f}'.format(epoch+1, train_accuracy, val_accuracy))

    saver.save(sess, "checkpoints/cnn_100.ckpt")
test_batch_size = 128

test_pred_res = []

with tf.Session() as sess:

    saver.restore(sess, "checkpoints/cnn_100.ckpt")

    for i in tqdm(range(0, X_test.shape[0], test_batch_size)):

        test_batch = X_test[i:i+test_batch_size,:,:,:]

        feed_dict = {inputs: test_batch}

        test_pred = sess.run(predicted, feed_dict=feed_dict)

        test_pred_res.append(test_pred.tolist())

    test_pred_res = np.concatenate(test_pred_res)
cnn_submit = submission.copy()

cnn_submit.is_iceberg = test_pred_res[:,1]

cnn_submit.to_csv('./cnn_100_submit.csv')