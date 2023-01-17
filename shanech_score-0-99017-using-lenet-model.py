# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf

import time

%matplotlib inline
train = pd.read_csv("../input/train.csv")

test = pd.read_csv('../input/test.csv')

train.head(1)
#show digits and labels

plt.title("number of %d" %train['label'][0])

plt.imshow(train.drop(['label'], axis=1).iloc[0,:].reshape(28,28))
#samples quantity

train.shape
#split mnist data to train and validation

np.random.seed(1)

idx = np.random.permutation(train.shape[0])

mnist_train = train.iloc[idx[:40000],:]

mnist_validation = train.iloc[idx[40000:],:]



mnist_validation_images = mnist_validation.drop(['label'], axis=1).values

mnist_validation_labels = pd.get_dummies(mnist_validation['label']).values

mnist_train_images = mnist_train.drop(['label'], axis=1).values

mnist_train_labels = pd.get_dummies(mnist_train['label']).values
#define mini batch function

start = 0

def next_batch(batch_size):

    global start

    global mnist_train_images

    global mnist_train_labels

    

    start = start+ batch_size

    if start >= mnist_train_labels.shape[0]:

        start = 0

        permutation = np.random.permutation(mnist_train_labels.shape[0])

        mnist_train_images = mnist_train_images[permutation, :]

        mnist_train_labels = mnist_train_labels[permutation, :]

    return mnist_train_images[start:start+batch_size,:], mnist_train_labels[start:start+batch_size,:]
#model: from LeNet-5

"""

parameters:

        data, labels, test: training images, labels, and test images

"""

#

def model(data, labels, test, learning_rate_base=0.001, num_epoch=30000, batch_size=128, L2_lambd=0.0003 ,keep_prob=1.0):

    tf.reset_default_graph()

    #placeholder

    X = tf.placeholder(tf.float32, shape=(None,28,28,1),name="X")

    Y = tf.placeholder(tf.float32, shape=(None,10),name="Y")

    dropout_op = tf.placeholder(tf.float32)

    #froward conv and pool

    W1 = tf.get_variable("W1", shape=(5,5,1,32), initializer=tf.truncated_normal_initializer(stddev=0.1))

    b1 = tf.get_variable("b1", shape=(32), initializer=tf.constant_initializer(0.))

    W2 = tf.get_variable("W2",shape=(5,5,32,64), initializer=tf.truncated_normal_initializer(stddev=0.1))

    b2 = tf.get_variable("b2", shape=(64), initializer=tf.constant_initializer(0.))

    

    conv_1 = tf.nn.conv2d(X, W1, strides=(1,1,1,1), padding="SAME")

    conv_1 = tf.nn.relu(conv_1 + b1)

    pool_1 = tf.nn.max_pool(conv_1, ksize=(1,2,2,1), strides=(1,2,2,1), padding="SAME")

    conv_2 = tf.nn.conv2d(pool_1, W2, strides=(1,1,1,1), padding="SAME")

    conv_2 = tf.nn.relu(conv_2+b2)

    pool_2 = tf.nn.max_pool(conv_2, ksize=(1,2,2,1), strides=(1,2,2,1), padding="SAME")

    #forward fullconnection

    pool_2 = tf.contrib.layers.flatten(pool_2)



    W3 = tf.get_variable("W3", shape=(3136, 512), initializer=tf.truncated_normal_initializer(stddev=0.1))

    b3 = tf.get_variable("b3", shape=(1, 512), initializer=tf.constant_initializer(0.1))

    fc_1 = tf.nn.relu(tf.matmul(pool_2, W3) + b3)

    #L2 regularization for full connection layes

    if L2_lambd != 0 :

        loss = tf.contrib.layers.l2_regularizer(L2_lambd)(W3)

        tf.add_to_collection("loss", loss)

    #also using dropout

    if keep_prob != 0:

        fc_1 = tf.nn.dropout(fc_1, dropout_op)

    

    W4 = tf.get_variable("W4", shape=(512, 10), initializer=tf.truncated_normal_initializer(stddev=0.1))

    b4 = tf.get_variable("b4", shape=(1, 10), initializer=tf.constant_initializer(0.1))

    Z4 = tf.matmul(fc_1, W4) + b4

    if L2_lambd != 0 :

        loss = tf.contrib.layers.l2_regularizer(L2_lambd)(W4)

        tf.add_to_collection("loss", loss)

    #compute cost

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z4, labels=Y)) 

    if L2_lambd != 0:

        cost = cost + tf.add_n(tf.get_collection("loss"))

    #train

    global_step = tf.Variable(0, trainable=False)

    learning_rate = tf.train.exponential_decay(learning_rate_base,

                                              global_step,

                                              data.shape[0]/batch_size,

                                              0.99)

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step = global_step)

    #prediction

    prediction = tf.cast(tf.argmax(Z4, axis=1), tf.float32)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y, 1), tf.argmax(Z4, 1)), tf.float32))

    

    train_cost = []

    with tf.Session() as sess:

        tf.global_variables_initializer().run()

        coord=tf.train.Coordinator()

        threads= tf.train.start_queue_runners(sess = sess,coord=coord)

        seed = 0

        for epoch in range(num_epoch):

            #mini batch

            mini_batch_X, mini_batch_Y = next_batch(batch_size)#sess.run([image_batch, label_batch])

            c, _ = sess.run([cost, optimizer], feed_dict={X: mini_batch_X.reshape(-1,28,28,1), Y: mini_batch_Y, dropout_op:keep_prob})

            

            if epoch % 10 ==0:

                train_cost.append(c/batch_size)

            if epoch%1000 == 0:

                valid_acc = accuracy.eval({X:mnist_validation_images.reshape(-1,28,28,1), Y:mnist_validation_labels, dropout_op:1.0})

                print("%d epoch, validation accuracy: %g" %(epoch, valid_acc))

                print("%d epoch, train cost: %g" %(epoch, c/batch_size))

                print("--------------------------------------------------------")

        #Because my computer doesnt have enough memory to load all testing data

        for i in range(10):

            y_pred = prediction.eval({X:test[i*2800:(i+1)*2800,:].reshape(-1,28,28,1), Y:np.zeros((2800,10)), dropout_op:1.0})

            df = pd.DataFrame(np.arange(2800), columns=['ImageId'])

            df['Label'] = y_pred

            df.to_csv("y_pred"+str(i)+".csv", index=False)

#         print("%d epoch, test accuracy: %g"%(epoch, test_acc))

        plt.plot(np.squeeze(train_cost))
model(mnist_train_images, mnist_train_labels, test, batch_size=128, keep_prob=0.5, L2_lambd=0.003)
df = pd.DataFrame()

for i in range(10):

    tmp = pd.read_csv('y_pred'+str(i)+'.csv')

    tmp['ImageId'] = 2800*i + tmp['ImageId']

    df = pd.concat([df, tmp])



df['ImageId'] += 1

df['ImageId'] = df['ImageId'].values.astype('int')

df['Label'] = df['Label'].values.astype('int')

df.to_csv('y_pred.csv', index=False)