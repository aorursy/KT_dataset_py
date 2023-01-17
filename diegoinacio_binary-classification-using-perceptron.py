# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt



import tensorflow as tf



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
plt.rcParams['figure.figsize'] = (16, 8)
df_train = pd.read_csv('../input/train.csv')

df_train.head()
df_train.info()
X_train = df_train.filter(regex='pixel*').values

Y_train = df_train['label'].values



# data cleaning

# using only zeros and ones

X_train = X_train[Y_train <= 1]

Y_train = Y_train[Y_train <= 1]



print('X_train:', X_train.shape)

print('Y_train:', Y_train.shape)
fig, AX = plt.subplots(3, 6, figsize=(2048//72, 1024//72))

AX = [b for a in AX for b in a]



np.random.seed(1)

for ax in AX:

    index = np.random.randint(10)

    ax.imshow(X_train[index].reshape(28, 28))

    ax.set_title('y = {}'.format(Y_train[index]), fontsize=20)
# data preparation

# scales, dimensions and dtypes

x_train, y_train = X_train/255, Y_train[np.newaxis].T



x_train = x_train.astype(np.float32).reshape(-1, 28*28)

y_train = y_train.astype(np.float32)



print('x_train:', x_train.shape)

print('y_train:', y_train.shape)
x_test, y_test = x_train[:816], y_train[:816]

x_train, y_train = x_train[816:], y_train[816:]



print('x_train:', x_train.shape)

print('y_train:', y_train.shape)

print('x_test:', x_test.shape)

print('y_test:', y_test.shape)
EPOCHS = 500 # epochs

ALPHA = 0.001 # learning rate

BATCH = 100   # batch size



# m is the number of examples

# n_x is the input size 28x28=784

m, n_x = x_train.shape



X = tf.placeholder(tf.float32, shape=[None, n_x], name='X')

Y = tf.placeholder(tf.float32, shape=[None, 1], name='Y')



# variables initialization

W = tf.Variable(tf.zeros([n_x, 1]), tf.float32, name='W')

B = tf.Variable(tf.zeros([1, 1]), tf.float32, name='B')



init_variables = tf.global_variables_initializer()



# model

Z = tf.add(tf.matmul(X, W), B)

A = tf.nn.sigmoid(Z)



# training graph and optimization

loss = tf.reduce_mean(tf.losses.mean_squared_error(predictions=A, labels=Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=ALPHA).minimize(loss)



# prediction graph

prediction = tf.round(A) 

compare = tf.equal(prediction, Y)

cast = tf.cast(compare, tf.float32)

accuracy = tf.reduce_mean(cast)*100



# loss and accuracy storage

loss_plot = []; accA_plot = []



with tf.Session() as sess:

    sess.run(init_variables)

    for epoch in range(EPOCHS + 1):

        # randomic batch definition

        rbatch = np.random.choice(y_train.size, size=BATCH)

        # training, metrics and storage

        sess.run(optimizer, feed_dict={X: x_train[rbatch], Y: y_train[rbatch]})

        L = sess.run(loss, feed_dict={X: x_train[rbatch], Y: y_train[rbatch]})

        acc = sess.run(accuracy, feed_dict={X: x_train, Y: y_train})

        loss_plot += [L]; accA_plot += [acc]

        if (not epoch % 100) and (epoch != 0):

            print('epoch: {0:04d} | loss: {1:.3f} | accuracy: {2:06.2f} %'.format(epoch, L, acc))

    w_ = sess.run(W) # store W and B for visualization and test

    b_ = sess.run(B)
axA = plt.subplot(121)

axA.imshow(w_.T.reshape(28, 28))

cb = axA.set_title('w')



axB = plt.subplot(222)

axB.plot(loss_plot)

axB.set_ylabel('loss')



axC = plt.subplot(224)

axC.plot(accA_plot)

axC.set_ylabel('accuracy')



plt.xlabel('epochs')



plt.show()
fig, AX = plt.subplots(3, 6, figsize=(2048//72, 1024//72))

AX = [b for a in AX for b in a]



np.random.seed(1)

for ax in AX:

    index = np.random.randint(y_test.size)

    z_ = np.dot(w_.T, x_test[index]) + b_

    a_ = 1/(1 + np.exp(-z_))

    y_ = 1 if a_ > 0.5 else 0

    ax.imshow(x_test[index].reshape(28, 28))

    ax.set_title(r'$y$ = ' + str(int(y_test[index])) + r' ; $\hat{y}$ = ' + str(y_), fontsize=20)