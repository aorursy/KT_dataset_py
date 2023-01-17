import tensorflow as tf

import numpy as np

import pandas as pd

import time

from sklearn.utils import shuffle



# Define paramaters for the model

learning_rate = 0.01

batch_size = 33

n_epochs = 100



# Step 1.1: Read in data

train_df = pd.read_csv('../input/train.csv') 

test_df  = pd.read_csv('../input/test.csv') 



# Step 1.2: create train and test data array

train_data  = train_df.loc[:,'margin1':'texture64']

test_data   = test_df.loc[:,'margin1':'texture64']

target_data = pd.get_dummies(train_df.species)



X_train = train_data.as_matrix()

X_test  = test_data.as_matrix()

y_train = target_data.as_matrix()



num_train = X_train.shape[0]

num_test  = X_test.shape[0]
X = tf.placeholder(tf.float32, [None, 192], name='X_placeholder') 

Y = tf.placeholder(tf.float32, [None, 99], name='Y_placeholder')
w = tf.Variable(tf.random_normal(shape=[192, 99], stddev=0.01), name='weights')

b = tf.Variable(tf.zeros([1, 99]), name="bias")
logits = tf.matmul(X, w) + b 
y = tf.nn.softmax(logits)

loss = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y), reduction_indices=[1]))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)



with tf.Session() as sess:

    # to visualize using TensorBoard

    #writer = tf.summary.FileWriter('./graph/leaf_lr', sess.graph)



    start_time = time.time()

    sess.run(tf.global_variables_initializer())	

    n_batches = int(num_train/batch_size)



    for i in range(n_epochs): # train the model n_epochs times



        # shuffle X, y

        X_train, y_train = shuffle(X_train, y_train)

        total_loss   = 0



        for j in range(n_batches):

            X_batch, Y_batch = X_train[j*batch_size:(j+1)*batch_size], y_train[j*batch_size:(j+1)*batch_size]

            _, loss_batch = sess.run([optimizer, loss], feed_dict={X: X_batch, Y:Y_batch}) 

            total_loss += loss_batch



        print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))



    print('Total time: {0} seconds'.format(time.time() - start_time))



    print('Optimization Finished!') # should be around 0.35 after 25 epochs



    # test the model

    logits_test = sess.run(logits, feed_dict={X: X_test}) 

    Y_pred = sess.run(tf.nn.softmax(logits_test))



    #writer.close()
sample_submission  = pd.read_csv('../input/sample_submission.csv')

col_idx = list(sample_submission)[1:]

row_idx = sample_submission.id.values



submission = pd.DataFrame(data=Y_pred, index=row_idx, columns=col_idx)

print(submission.head())