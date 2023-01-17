import tensorflow as tf

import pandas as pd

import numpy as np

import math
# load data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
# shape train data

train_labels = train.as_matrix(columns=['label'])

train_images = train.drop('label', axis = 1)

train_images = train_images.as_matrix()



# shape test data

test_images = test.as_matrix()
# converts a digit into one hot encoding

def one_hot(value):

    arr = np.zeros(10)

    arr[value] = 1.0

    return arr



train_labels = np.apply_along_axis(one_hot, axis = 1, arr = train_labels)
# convert int value to float value [0~1]

train_images = train_images / 255.0

test_images = test_images / 255.0
print(train_images.shape, test_images.shape)
# print sample

train_labels[0]
# print sample

train_images[0]
# parameters for training

training_epochs = 2500

batch_size = 50

learning_rate = 0.001
# create LeNet 5 model

# We use keep_prob and dropout to reduce overfitting.



tf.reset_default_graph()



X = tf.placeholder('float32', shape = [None, 784])

y = tf.placeholder('float32', shape = [None, 10])



X_image = tf.reshape(X, [-1, 28, 28, 1])

keep_prob = tf.placeholder('float32', name = 'keep_prob')



with tf.variable_scope('conv_layer1') as scope:

    W1 = tf.get_variable('W_conv1', shape = [5, 5, 1, 32], initializer=tf.contrib.layers.xavier_initializer_conv2d())

    b1 = tf.Variable(tf.random_normal([32]), name = 'b_conv1')

    conv1 = tf.nn.relu(tf.nn.conv2d(X_image, W1, strides=[1,1,1,1], padding='SAME'))

    pool1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    

with tf.variable_scope('conv_layer2') as scope:

    W2 = tf.get_variable('W_conv2', shape = [5, 5, 32, 64], initializer=tf.contrib.layers.xavier_initializer_conv2d())

    b2 = tf.Variable(tf.random_normal([64]), name = 'b_conv2')

    conv2 = tf.nn.relu(tf.nn.conv2d(pool1, W2, strides=[1,1,1,1], padding='SAME'))

    pool2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')



with tf.variable_scope('fc_layer1') as scope:

    W3 = tf.get_variable('W_fc1', shape = [7 * 7 * 64, 1024], initializer=tf.contrib.layers.xavier_initializer())

    b3 = tf.Variable(tf.random_normal([1024]), name = 'b_fc1')

    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    fc1 = tf.nn.relu(tf.matmul(pool2_flat, W3) + b3)

    fc1_drop = tf.nn.dropout(fc1, keep_prob)

    

with tf.variable_scope('fc_layer2') as scope:

    W4 = tf.get_variable('W_fc2', shape = [1024, 10], initializer=tf.contrib.layers.xavier_initializer())

    b4 = tf.Variable(tf.random_normal([10]), name = 'b_fc2')

    y_ = tf.add(tf.matmul(fc1_drop, W4), b4)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_, labels = y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
# add saver op to save model

saver = tf.train.Saver()
# batch function for SGD

epochs_completed = 0

index_in_epoch = 0

num_examples = train_images.shape[0]



def next_batch(batch_size):

    

    global train_images

    global train_labels

    global index_in_epoch

    global epochs_completed

    

    start = index_in_epoch

    index_in_epoch += batch_size

    

    # if epoch finished, shuffle data

    if index_in_epoch > num_examples:

        epochs_completed += 1

        # shuffle the data

        perm = np.arange(num_examples)

        np.random.shuffle(perm)

        train_images = train_images[perm]

        train_labels = train_labels[perm]

        

        # start next epoch

        start = 0

        index_in_epoch = batch_size

        assert batch_size <= num_examples # check

        

    end = index_in_epoch    

    return train_images[start:end], train_labels[start:end]
# train model

init = tf.global_variables_initializer()

sess = tf.InteractiveSession()

sess.run(init)



for epoch in range(training_epochs):

    #get new batch

    batch_xs, batch_ys = next_batch(batch_size=batch_size)

    feed_dict = {X: batch_xs, y: batch_ys, keep_prob: 0.5}

    sess.run(optimizer, feed_dict=feed_dict)



    # for every 100 epoch, print accuracy for small sample

    if(epoch % 100 == 99):

        print('Epoch:', '%02d' % (epoch + 1), 'Done')

        print(accuracy.eval(session=sess, feed_dict = {X: train_images[0:128],

                                                       y: train_labels[0:128],

                                                       keep_prob: 1.0}))

        # save model for every 500 epoch

        if(epoch % 500 == 499):

            save_path = saver.save(sess, "./tmp/model_" + str(int((epoch + 1) / 500)) + ".ckpt")

            print('Model saved at:', save_path)



# close session

sess.close()
# initialize array to store predicted labels

labels = np.array([]).astype(int)



with tf.Session() as sess:

    # restore saved model

    saver.restore(sess, './tmp/model_5.ckpt')

    

    # predict labels of 1000 images per iteration due to memory issue

    for i in range(0, len(test_images), 1000):

        images = test_images[i:(i+1000)]

        print('Predicting Label For Image', i+1, 'to', i+1000)

        prediction = y_.eval(session=sess, feed_dict = {X: images, keep_prob: 1.0})

        labels = np.append(labels, prediction.argmax(axis=1).astype(int))
# save result as csv

pd.DataFrame({'ImageId': np.arange(1, len(labels)+1), 'Label': labels}).to_csv('result.csv', index=False)