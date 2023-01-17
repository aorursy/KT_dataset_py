# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

import tensorflow as tf

from keras.utils import to_categorical

import numpy as np

import pandas as pd
tf.set_random_seed(42)

np.random.seed(42)
train = pd.read_csv('../input/fashion-mnist_train.csv', skiprows=1)

images = np.array(train.iloc[:, 1:])

labels = to_categorical(np.array(train.iloc[:, 0]))
def dense(x, W, b):

    z = tf.add(tf.matmul(x, W), b)

    return tf.nn.relu(z)
def conv2D(x, W, b, stride_length=1):

    xW = tf.nn.conv2d(x, W, strides=[1, stride_length, stride_length, 1], padding='SAME')

    z = tf.nn.bias_add(xW, b)

    return tf.nn.relu(z)
def maxpooling2D(x, p_size):

    z = tf.nn.max_pool(x, ksize=[1, p_size, p_size, 1], strides=[1, p_size, p_size, 1], padding='SAME')

    return z
def network(x, weights, biases, n_in, mp_size, mp_dropout, dense_dropout):

    sq_dim = int(np.sqrt(n_in))

    sq_x = tf.reshape(x, shape=[-1, sq_dim, sq_dim, 1])

    

    conv1 = conv2D(sq_x, weights['w_c1'], biases['b_c1'])

    conv2 = conv2D(conv1, weights['w_c2'], biases['b_c2'])

    

    mp1 = maxpooling2D(conv2, mp_size)

    mp1 = tf.nn.dropout(mp1, 1- mp_dropout)

    

    flat = tf.reshape(mp1, [-1, weights['w_d1'].get_shape().as_list()[0]])

    dense1 = dense(flat, weights['w_d1'], biases['b_d1'])

    dense1 = tf.nn.dropout(dense1, 1 - dense_dropout)

    

    out = tf.add(tf.matmul(dense1, weights['w_out']), biases['b_out'])

    return out
n_conv1 = 32

k_conv1 = 3

n_conv2 = 64

k_conv2 = 3

n_pool_size = 2

mp_dropout = 0.25

dense_dropout = 0.5

n_classes = 10

n_inputs = 784

n_dense = 128
weight_initializer = tf.contrib.layers.xavier_initializer()



biases = {

    'b_c1': tf.Variable(tf.zeros([n_conv1])),

    'b_c2': tf.Variable(tf.zeros([n_conv2])),

    'b_d1': tf.Variable(tf.zeros([n_dense])),

    'b_out': tf.Variable(tf.zeros([n_classes])),

}



full_sq_len = np.sqrt(n_inputs)

pooled_sq_len = int(full_sq_len / n_pool_size)

dense_inputs = pooled_sq_len ** 2  * n_conv2



weights = {

    'w_c1': tf.get_variable('w_c1', [k_conv1, k_conv1, 1, n_conv1], initializer=weight_initializer),

    'w_c2': tf.get_variable('w_c2', [k_conv2, k_conv2, n_conv1, n_conv2], initializer=weight_initializer),

    'w_d1': tf.get_variable('w_d1', [dense_inputs, n_dense], initializer=weight_initializer),

    'w_out': tf.get_variable('w_out', [n_dense, n_classes], initializer=weight_initializer)

}
x = tf.placeholder(tf.float32, [None, n_inputs])

y = tf.placeholder(tf.float32, [None, n_classes])
predictions = network(x, weights, biases, n_inputs, n_pool_size, mp_dropout, dense_dropout)



cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y))

optimizer = tf.train.AdamOptimizer().minimize(cost)
correct_predictions = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))

accuracy_pct = tf.reduce_mean(tf.cast(correct_predictions, tf.float32)) * 100
initializer_op = tf.global_variables_initializer()

def next_batch(batch_size):

    idx = np.arange(0 , 59999)

    np.random.shuffle(idx)

    idx = idx[:batch_size]

    data_shuffle = [images[i] for i in idx]

    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)
with tf.Session() as session:

    session.run(initializer_op)

    

    n_epocs = 100

    batch_size = 128

    

    for epoch in range(n_epocs):

        avg_cost = 0.0

        avg_accuracy_pct = 0.0

        n_batches = int(60000/batch_size)

        

        for i in range(n_batches):

            batch_x, batch_y = next_batch(batch_size)

            

            _, batch_cost, batch_acc = session.run([optimizer, cost, accuracy_pct], feed_dict = {x:batch_x, y:batch_y})

            avg_cost += batch_cost / n_batches

            avg_accuracy_pct += batch_acc / n_batches



        

        print("Epoch ", '%03d' % (epoch+1), 

                      ": cost = ", '{:.3f}'.format(avg_cost), 

                      ", accuracy = ", '{:.2f}'.format(avg_accuracy_pct), "%", 

                      sep='')