import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

import numpy as np

import cv2

import os

from tqdm import tqdm_notebook as tqdm # Jupyter notebook should use this



os.environ['CUDA_VISIBLE_DEVICES'] = '0'

DATASET_PATH = '../input/15-scene/15-Scene/'



one_hot_lookup = np.eye(15) # 15 classes



dataset_x = []

dataset_y = []



for category in sorted(os.listdir(DATASET_PATH)):

    print('loading category: '+str(int(category)))

    for fname in os.listdir(DATASET_PATH+category):

        img = cv2.imread(DATASET_PATH+category+'/'+fname, 2)

        img = cv2.resize(img, (224,224))

        dataset_x.append(np.reshape(img, [224,224,1]))

        dataset_y.append(np.reshape(one_hot_lookup[int(category)], [15]))



dataset_x = np.array(dataset_x)

dataset_y = np.array(dataset_y)



"""shuffle dataset"""

p = np.random.permutation(len(dataset_x))

dataset_x = dataset_x[p]

dataset_y = dataset_y[p]

        

test_x = dataset_x[:int(len(dataset_x)/10)]

test_y = dataset_y[:int(len(dataset_x)/10)]

train_x = dataset_x[int(len(dataset_x)/10):]

train_y = dataset_y[int(len(dataset_x)/10):]
print(train_x.shape)

print(train_y.shape)

print(test_x.shape)

print(test_y.shape)
"""fully connected layer"""

def dense(input_tensor, input_units, units, name_scope):

    with tf.name_scope(name_scope):

        W = tf.Variable(tf.random_normal((input_units, units)))

        b = tf.Variable(tf.random_normal((units,)))

        return tf.nn.relu(tf.matmul(input_tensor, W) + b)

        

"""convolution layer in 2D"""

def conv2d(input_tensor, filter_size, in_channels, out_channels, name_scope, activation=True):

    with tf.name_scope(name_scope):

        kernel = tf.Variable(tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 

                                                 dtype=tf.float32, stddev=1e-1), name='weights', trainable=True)

        conv = tf.nn.conv2d(input_tensor, kernel, [1, 1, 1, 1], padding='SAME')

        biases = tf.Variable(tf.constant(0.0, shape=[out_channels], dtype=tf.float32), trainable=True, name='biases')

        conv_biased = tf.nn.bias_add(conv, biases)

        if activation == False:

            return conv_biased # no ReLU activation

        relu_conv = tf.nn.relu(conv_biased)

        return relu_conv

    



"""build the CNN"""

X = tf.placeholder(tf.float32, [None, 224, 224, 1])

Y = tf.placeholder(tf.float32, [None, 15])



conv1 = conv2d(input_tensor=X, filter_size=3, in_channels=1, out_channels=16, name_scope='conv1') # output [224,224,16]

pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # output [112,112,16]

conv2 = conv2d(input_tensor=pool1, filter_size=3, in_channels=16, out_channels=32, name_scope='conv2')

pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # output [56,56,32]

conv3 = conv2d(input_tensor=pool2, filter_size=3, in_channels=32, out_channels=32, name_scope='conv3')

pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #

conv4 = conv2d(input_tensor=pool3, filter_size=3, in_channels=32, out_channels=32, name_scope='conv4') 

pool4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # 





flatten = tf.reshape(pool4, [-1, 14*14*32])



fc1 = tf.layers.dense(flatten, 256, activation=tf.nn.relu)

fc2 = tf.layers.dense(fc1, 128, activation=tf.nn.relu)



output = tf.layers.dense(fc2, 15, activation=None)



loss = tf.losses.softmax_cross_entropy(onehot_labels=Y, logits=output)

loss = tf.reduce_mean(loss)

    

optimizer = tf.train.AdamOptimizer(0.0004).minimize(loss)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y, 1), tf.argmax(output,1)), dtype=tf.float32))
sess = tf.Session()

init = tf.global_variables_initializer()

sess.run(init)
BATCH_SIZE = 64

epoch = 0

while epoch < 20:

    total_steps = int(len(train_x)/BATCH_SIZE)

    loss_value = 0

    for step in tqdm(range(total_steps), desc=('Epoch '+str(epoch))):

        """get training batch"""

        if step*BATCH_SIZE + BATCH_SIZE < len(train_x):

            BATCH_X = train_x[step*BATCH_SIZE: step*BATCH_SIZE+BATCH_SIZE]

            BATCH_Y = train_y[step*BATCH_SIZE: step*BATCH_SIZE+BATCH_SIZE]

        else:

            BATCH_X = train_x[step*BATCH_SIZE:]

            BATCH_Y = train_y[step*BATCH_SIZE:]

            

        """train"""

        [loss_value, accuracy_value, _] = sess.run([loss, accuracy, optimizer], feed_dict={X: BATCH_X, Y: BATCH_Y})



    """end of epoch"""

    print('Epoch loss = ',loss_value, ' training accuracy = ',accuracy_value)

    test_acc = sess.run(accuracy, feed_dict={X: test_x[:100], Y: test_y[:100]})

    print('testing accuracy = ', test_acc)



    epoch+=1