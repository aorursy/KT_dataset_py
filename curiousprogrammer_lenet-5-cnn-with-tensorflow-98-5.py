# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow as tf

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv')
X_train = df_train.iloc[:, 1:]

Y_train = df_train.iloc[:, 0]
X_train.head()
Y_train.head()
X_train = np.array(X_train)

Y_train = np.array(Y_train)
X_train = X_train/255.0
# dev-val split

X_dev, X_val, Y_dev, Y_val = train_test_split(X_train, Y_train, test_size=0.03, shuffle=True, random_state=2019)



#Reshape the arrays to match the input in tensorflow graph

X_dev = X_dev.reshape((X_dev.shape[0], 28, 28, 1))

X_val = X_val.reshape((X_val.shape[0], 28, 28, 1))
def plot_digits(X, Y):

    for i in range(20):

        plt.subplot(4, 5, i+1)

        plt.tight_layout()

        plt.imshow(X[i].reshape((28, 28)), cmap='gray')

        plt.title('Digit:{}'.format(Y[i]))

        plt.xticks([])

        plt.yticks([])

    plt.show()
plot_digits(X_train[-20:], Y_train[-20:])
x = tf.placeholder(dtype=tf.float32, shape=(None, 28, 28, 1), name='X')

y = tf.placeholder(dtype=tf.float32, shape=(None, 10), name='Y')
conv1 = tf.layers.conv2d(x, filters=6, kernel_size=5, padding='same', strides=1, activation='relu', name='CONV1')

pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2, name='POOL1')

conv2 = tf.layers.conv2d(pool1, filters=16, kernel_size=5, strides=1, activation='relu', name='CONV2')

pool2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2, name='POOL2')

flatten1 = tf.layers.Flatten()(pool2)

fc1 = tf.layers.Dense(120, activation='relu')(flatten1)

fc2 = tf.layers.Dense(84, activation='relu')(fc1)

out = tf.layers.Dense(10, activation='softmax')(fc2)
batch_size = 100

learning_rate = 5e-4

epochs = 20
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=out, name='cost'))

opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)

equal_pred = tf.equal(tf.argmax(y, 1), tf.argmax(out, 1))

acc = tf.reduce_mean(tf.cast(equal_pred, tf.float32))
init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)
T_dev = pd.get_dummies(Y_dev).values

T_val = pd.get_dummies(Y_val).values

for epoch in range(epochs):

    start_index = 0

    s = np.arange(X_dev.shape[0])

    np.random.shuffle(s)

    X_dev = X_dev[s, :]

    T_dev = T_dev[s]

    while start_index < X_dev.shape[0]:

        end_index = start_index + batch_size

        if end_index > X_dev.shape[0]:

            end_index = X_dev.shape[0]

        x_dev = X_dev[start_index:end_index, :]

        t_dev = T_dev[start_index:end_index]

        dev_cost, dev_acc, _ = sess.run([cost, acc, opt], feed_dict={x:x_dev, y:t_dev})

        start_index = end_index

    dev_cost, dev_acc = sess.run([cost, acc], feed_dict={x:X_dev, y:T_dev})

    val_cost, val_acc = sess.run([cost, acc], feed_dict={x:X_val, y:T_val})

    print('Epoch:{0} Cost:{1:5f} Acc:{2:.5f} Val_Cost:{3:5f} Val_Accuracy:{4:.5f}'.

          format(epoch+1, dev_cost, dev_acc, val_cost, val_acc))
X_test = pd.read_csv('../input/test.csv')

X_test = np.array(X_test)

X_test = X_test / 255.0

X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

T_test = sess.run([out], feed_dict={x:X_test})
Y_test = np.argmax(T_test[0], axis=1)

Y_test[:5]
df_out = pd.read_csv('../input/sample_submission.csv')

df_out['Label'] = Y_test

df_out.to_csv('out.csv', index=False)