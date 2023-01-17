import numpy as np

import pandas as pd

import os

import glob

import cv2

from sklearn.cross_validation import KFold

from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Flatten

from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D

from keras.optimizers import SGD, Adagrad

from keras.callbacks import EarlyStopping

from keras.utils import np_utils

from keras.constraints import maxnorm

from sklearn.metrics import log_loss

from keras import __version__ as keras_version

import tensorflow as tf

print('完成读取相应包')

#img = cv2.imread('../input/train/ALB/img_00967.jpg')

#resized = cv2.resize(img, (128, 128), cv2.INTER_LINEAR)

#Gray = resized[:,:,1]*0.299 + resized[:,:,1]*0.587 + resized[:,:,1]*0.114

#x = np.reshape(Gray, (1,128*128))
def get_im_cv2(path):

    img = cv2.imread(path)

    resized = cv2.resize(img, (32, 32), cv2.INTER_LINEAR)

    resized = resized[:,:,0]*0.299 + resized[:,:,1]*0.587 + resized[:,:,2]*0.114

    resized = np.reshape(resized,(1,32*32))

    return resized



def load_train():

    X_train = []

    X_train_id = []

    y_train = []



    print('Read train images')

    folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

    for fld in folders:

        index = folders.index(fld)

        print('Load folder {} (Index: {})'.format(fld, index))

        path = os.path.join('..', 'input', 'train', fld, '*.jpg')

        files = glob.glob(path)

        for fl in files:

            flbase = os.path.basename(fl)

            img = get_im_cv2(fl)

            X_train.append(img)

            X_train_id.append(flbase)

            y_train.append(index)



    return X_train, y_train, X_train_id
def load_test():

    path = os.path.join('..', 'input', 'test_stg1', '*.jpg')

    files = sorted(glob.glob(path))



    X_test = []

    X_test_id = []

    for fl in files:

        flbase = os.path.basename(fl)

        img = get_im_cv2(fl)

        X_test.append(img)

        X_test_id.append(flbase)



    return X_test, X_test_id



def create_submission(predictions, test_id, info):

    result1 = pd.DataFrame(predictions, columns=['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])

    result1.loc[:, 'image'] = pd.Series(test_id, index=result1.index)

    sub_file = 'submission1.csv'

    result1.to_csv(sub_file, index=False)
#读取数据

train_data, train_target, train_id = load_train()

#转换为array

train_data = np.array(train_data, dtype=np.uint8)

train_target = np.array(train_target, dtype=np.uint8)

#转换矩阵顺序

#train_data = train_data.transpose((0, 3, 1, 2))

#数据归一化到0~1之间

train_data = train_data.astype('float32')

train_data = train_data / 255

train_data = train_data[:,0,:]

#label转换为哑变量

train_target = np_utils.to_categorical(train_target, 8)
#读取测试数据

test_data, test_id = load_test()

test_data = np.array(test_data, dtype=np.uint8)

#test_data = test_data.transpose((0, 3, 1, 2))

test_data = test_data.astype('float32')

test_data = test_data / 255

test_data = test_data[:,0,:]
learning_rate = 0.01

training_iters = 200

display_step = 10



# Network Parameters

n_input = 32*32 #data input (img shape: 128*128)

n_classes = 8 # total classes (0-7 digits)

dropout = 0.75 # Dropout, probability to keep units



x = tf.placeholder(tf.float32, [None, n_input])

y = tf.placeholder(tf.float32, [None, n_classes])

keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)



# Create some wrappers for simplicity

def conv2d(x, W, b, strides=1):

    # Conv2D wrapper, with bias and relu activation

    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')

    x = tf.nn.bias_add(x, b)

    return tf.nn.relu(x)





def maxpool2d(x, k=2):

    # MaxPool2D wrapper

    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],

                          padding='SAME')





# Create model

def conv_net(x, weights, biases, dropout):

    # Reshape input picture

    x = tf.reshape(x, shape=[-1, 32, 32, 1])



    # Convolution Layer

    conv1 = conv2d(x, weights['wc1'], biases['bc1'])

    # Max Pooling (down-sampling)

    conv1 = maxpool2d(conv1, k=2)



    # Convolution Layer

    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])

    # Max Pooling (down-sampling)

    conv2 = maxpool2d(conv2, k=2)



    # Fully connected layer

    # Reshape conv2 output to fit fully connected layer input

    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])

    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])

    fc1 = tf.nn.relu(fc1)

    # Apply Dropout

    fc1 = tf.nn.dropout(fc1, dropout)



    # Output, class prediction

    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    return out



# Store layers weight & bias

weights = {

    # 5x5 conv, 1 input, 32 outputs

    'wc1': tf.Variable(tf.random_normal([8, 8, 1, 32])),

    # 5x5 conv, 32 inputs, 64 outputs

    'wc2': tf.Variable(tf.random_normal([8, 8, 32, 64])),

    # fully connected, 7*7*64 inputs, 1024 outputs

    'wd1': tf.Variable(tf.random_normal([8*8*64, 1024])),

    # 1024 inputs, 10 outputs (class prediction)

    'out': tf.Variable(tf.random_normal([1024, n_classes]))

}



biases = {

    'bc1': tf.Variable(tf.random_normal([32])),

    'bc2': tf.Variable(tf.random_normal([64])),

    'bd1': tf.Variable(tf.random_normal([1024])),

    'out': tf.Variable(tf.random_normal([n_classes]))

}
2
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables

init = tf.initialize_all_variables()
train_target.shape
sess = tf.Session()

sess.run(init)



for i in range(training_iters):

    feed={x:train_data, y:train_target,keep_prob: dropout}

    sess.run(optimizer, feed_dict=feed)

    if i % 1000 == 0 or i == ITERATIONS-1:

        loss, acc = sess.run([cost, accuracy], feed_dict={x: train_target, y: keep_prob,keep_prob: 1.})

        print("Iter " + str(training_iters) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))

print("Optimization Finished!")        
predicted = sess.run(y, feed_dict={x:test_features})





#sess.run(accuracy, feed_dict={x: mnist.test.images[:256],

#                                      y: mnist.test.labels[:256],

#                                     keep_prob: 1.}))