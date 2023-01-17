%matplotlib inline

import os

import random

import tensorflow as tf

import time

from datetime import datetime

import pandas as pd

import numpy as np

import cv2

import matplotlib.pyplot as plt

import matplotlib

matplotlib.style.use('fivethirtyeight')

matplotlib.rcParams['font.size'] = 12

matplotlib.rcParams['figure.figsize'] = (15,8)
categories = {

    0:'T-shirt/top',

    1:'Trouser',

    2: 'Pullover',

    3: 'Dress',

    4: 'Coat',

    5: 'Sandal',

    6: 'Shirt',

    7: 'Sneaker',

    8: 'Bag',

    9: 'Ankle boot' ,

}



train_data = np.array(pd.read_csv('../input/fashion-mnist_train.csv', low_memory=False))

val_data = train_data[:10000]

train_data = train_data[10000:]

test_data = np.array(pd.read_csv('../input/fashion-mnist_test.csv', low_memory=False))

print('Train Data Shape:'+str(train_data.shape))

print('Val Data Shape:'+str(val_data.shape))

print('Test Data Shape:'+str(test_data.shape))



def data_generator(mode='train', batch_size=64):

    if mode == 'train':

        choice = np.random.choice(train_data.shape[0], size=batch_size, replace=False)

        batch = train_data[choice]

        return batch[:, 1:].reshape((-1,28,28,1)), np.eye(10)[batch[:, 0]]

    if mode == 'val':

        choice = np.random.choice(val_data.shape[0], size=batch_size, replace=False)

        batch = val_data[choice]

        return batch[:, 1:].reshape((-1,28,28,1)), np.eye(10)[batch[:, 0]]

    if mode == 'test':

        choice = np.random.choice(test_data.shape[0], size=batch_size, replace=False)

        batch = test_data[choice]

        return batch[:, 1:].reshape((-1,28,28,1)), np.eye(10)[batch[:, 0]]



print('Train batch shape' + str(data_generator('train')[0].shape))

print('Val batch shape' + str(data_generator('val')[0].shape))

print('Test batch shape' + str(data_generator('test')[0].shape))
x_rand, y_rand = data_generator('train', 16)



for i, img in enumerate(x_rand):

    plt.subplot(4, 4, i+1)

    plt.imshow(img.reshape(28,28), cmap='gray')

    plt.title(categories[np.argmax(y_rand[i], axis=-1)])

    plt.tight_layout()
def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='VALID'):

    """Create a convolution layer"""

    # Get number of input channels

    input_channels = int(x.get_shape()[-1])



    with tf.variable_scope(name) as scope:

        # Create tf variables for the weights and biases of the conv layer

        weights = tf.get_variable('weights', shape=[filter_height,

                                                    filter_width,

                                                    input_channels,

                                                    num_filters])

        biases = tf.get_variable('biases', shape=[num_filters])



    return tf.nn.relu(tf.nn.conv2d(input=x, filter=weights, strides=[1, stride_y, stride_x, 1],padding=padding)+biases)



def fc(x, num_in, num_out, name, relu=True):

    """Create a fully connected layer."""

    with tf.variable_scope(name) as scope:



        # Create tf variables for the weights and biases

        weights = tf.get_variable('weights', shape=[num_in, num_out],

                                  trainable=True)

        biases = tf.get_variable('biases', [num_out], trainable=True)



        # Matrix multiply weights and inputs and add bias

        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)



    if relu:

        # Apply ReLu non linearity

        relu = tf.nn.relu(act)

        return relu

    else:

        return act



def max_pool(x, filter_height, filter_width, stride_y, stride_x, name,

             padding='SAME'):

    """Create a max pooling layer."""

    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],

                          strides=[1, stride_y, stride_x, 1],

                          padding=padding, name=name)



def lrn(x, radius, alpha, beta, name, bias=1.0):

    """Create a local response normalization layer."""

    return tf.nn.local_response_normalization(x, depth_radius=radius,

                                              alpha=alpha, beta=beta,

                                              bias=bias, name=name)



def dropout(x, keep_prob):

    """Create a dropout layer."""

    return tf.nn.dropout(x, keep_prob)



global_step = tf.Variable(1, trainable=False, name='global_step')





print('Dimension of each layer : ')

# Change /gpu:0 to /cpu:0 if gpu is not set up

with tf.device('/cpu:0'):

    """Create the network graph."""

    # Input tensors

    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))

    y = tf.placeholder(tf.int64, shape=(None, 10))

    rate = tf.placeholder(tf.float32, shape=[])

    keep_prob = tf.placeholder(tf.float32)



    # 1st Layer: Conv (w ReLu) -> Pool -> Lrn

    conv1 = conv(x, 5, 5, 96, 1, 1, padding='VALID', name='conv1')

    print('conv1'+str(conv1.get_shape()))



    # 2nd Layer: Conv (w ReLu) -> Pool -> Lrn with 2 groups

    conv2 = conv(conv1, 3, 3, 256, 1, 1, name='conv2')

    pool2 = max_pool(conv2, 2, 2, 2, 2, padding='VALID', name='pool2')

    norm2 = lrn(pool2, 2, 2e-05, 0.75, name='norm2')

    print('conv2'+str(conv2.get_shape()))

    print('pool2'+str(pool2.get_shape()))

    print('norm2'+str(norm2.get_shape()))



    # 3rd Layer: Conv (w ReLu) splitted into two groups

    conv3 = conv(norm2, 3, 3, 384, 1, 1, padding='VALID', name='conv3')

    print('conv3'+str(conv3.get_shape()))



    # 4th Layer: Conv (w ReLu) -> Pool splitted into two groups

    conv4 = conv(conv3, 3, 3, 384, 1, 1, name='conv4')

    pool4 = max_pool(conv4, 2, 2, 2, 2, padding='SAME', name='pool4')

    norm4 = lrn(pool4, 2, 2e-05, 0.75, name='norm4')

    print('conv4'+str(conv4.get_shape()))

    print('pool4'+str(pool4.get_shape()))



    # 5th Layer: Flatten -> FC (w ReLu) -> Dropout

    norm4_dims = int(norm4.get_shape()[1] * norm4.get_shape()[2] * norm4.get_shape()[3])

    flattened = tf.reshape(norm4, [-1, norm4_dims])

    fc5 = fc(flattened, norm4_dims, 1024, name='fc5')

    dropout5 = dropout(fc5, keep_prob)



    # 6th Layer: FC (w ReLu) -> Dropout

    fc6 = fc(dropout5, 1024, 512, name='fc6')

    dropout6 = dropout(fc6, keep_prob)



    # 7th Layer: C (w ReLu) -> Dropout

    fc7 = fc(dropout6, 512, 128, name='fc7')

    dropout7 = dropout(fc7, keep_prob)



    # 8th Layer: FC and return unscaled activations

    fc8 = fc(dropout7, 128, 10, relu=False, name='fc8')



    print('fc6'+str(fc6.get_shape()))

    print('fc7'+str(fc7.get_shape()))

    print('fc8'+str(fc8.get_shape()))



    # Define the loss and the accuracies

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc8, labels=y), name="cross_ent_loss")

    optimizer = tf.train.AdamOptimizer(rate)

    train_step = optimizer.minimize(loss=loss, global_step=global_step)



prediction = tf.argmax(tf.nn.softmax(fc8), axis=-1)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(fc8, 1), tf.argmax(y, 1)), tf.float32), name="accuracy")



sess = tf.Session()

sess.run(tf.global_variables_initializer())

def validate(g_step, samples=100):

    acc_val = 0

    acc_train = 0

    for iter in np.arange(samples):

        x_batch, y_batch = data_generator(mode='val')

        acc_val += sess.run(accuracy, feed_dict={ x: x_batch,

                                                    y: y_batch,

                                                    keep_prob: 1.0})

        x_batch, y_batch = data_generator(mode='train')

        acc_train += sess.run(accuracy, feed_dict={ x: x_batch,

                                                    y: y_batch,

                                                    keep_prob: 1.0})

    acc_val /= samples

    acc_train /=samples

    

    return [acc_train, acc_val]



def predict( image):

        return prediction.eval(session=sess, feed_dict={x: [image.reshape(28, 28, 1)], 

                                                                  keep_prob: 1.0})[0]
def train(iterations=1000, learning_rate=1e-4, display_step=50, validate_step=200):

    # Initialize the log vars

    loss_log = np.zeros(shape=(1,2))

    rate_log = np.zeros(shape=(1,2))

    val_acc_log = np.zeros(shape=(1,2))

    train_acc_log = np.zeros(shape=(1,2))

    

    start_time = time.time()



    for iter in np.arange(iterations):

        g_step = tf.train.global_step(sess, global_step)

        x_batch, y_batch = data_generator()

        _, loss_scalar, a0, bb1, bb2 = sess.run(fetches=[train_step, loss, optimizer._lr, optimizer._beta1_power, optimizer._beta2_power],

                                        feed_dict={ x: x_batch,

                                                    y: y_batch,

                                                    rate: learning_rate,

                                                    keep_prob: 0.5})

        rate_scalar = a0* (1-bb2)**0.5 /(1-bb1)

        # Log the loss and rates

        loss_log = np.append(loss_log, [[g_step, loss_scalar]], axis=0)

        rate_log = np.append(rate_log, [[g_step, rate_scalar]], axis=0)

        

        # Verbose

        if g_step % display_step == 0:

            print('global_step {} finished in {:.2f} s loss = {} '.format(g_step, time.time() - start_time, loss_scalar))

            start_time = time.time()

        if g_step % validate_step == 0:

            train_acc, val_acc = validate(g_step)

            print('\nglobal_step {} training acc = {}% validation acc {}%\n'.format(g_step, train_acc*100, val_acc*100))

            # Log the accuracies 

            train_acc_log = np.append(train_acc_log, [[g_step, train_acc]], axis=0)

            val_acc_log = np.append(val_acc_log, [[g_step, val_acc]], axis=0)

            start_time = time.time()

            

    # Final Validation        

    train_acc, val_acc = validate(g_step)

    print('Final Accuracies:\n')

    print('global_step {} training acc = {} validation acc {}\n'.format(g_step, train_acc*100, val_acc*100))

    train_acc_log = np.append(train_acc_log, [[g_step, train_acc]], axis=0)

    val_acc_log = np.append(val_acc_log, [[g_step, val_acc]], axis=0)

    return loss_log, rate_log, train_acc_log, val_acc_log
# Calling the train function



loss_log, rate_log, train_acc_log, val_acc_log = train(iterations=1000, learning_rate=1e-3, display_step=100, validate_step=250)







plt.xlabel('Global Step')

plt.ylabel('Loss')

plt.title('LOSS')

plt.plot(loss_log[1:,0], loss_log[1:,1], linewidth=1)

plt.ylim(loss_log[1:,1].min()*1.1, loss_log[1:,1].mean()*2.5)

plt.show()
plt.xlabel('Global Step')

plt.ylabel('Rate')

plt.title('Learning Rate')

plt.plot(rate_log[1:,0], rate_log[1:,1], linewidth=1)

plt.ylim(rate_log[1:,1].min()*1.1, rate_log[1:,1].mean()*2.5)

plt.show()
plt.xlabel('Global Step')

plt.ylabel('Accuracy')

plt.title('Train and Val Accuracy')

plt.plot(train_acc_log[1:,0], train_acc_log[1:,1], linewidth=2, color="cyan", label='train acc')

plt.plot(val_acc_log[1:,0], val_acc_log[1:,1], linewidth=2, color="red", label='val acc')

plt.legend(loc='upper left', frameon=False)

plt.ylim(train_acc_log[1:,1].mean()*0.9, train_acc_log[1:,1].mean()*1.1)

plt.show()
x_test, y_test = data_generator('test', 9)

for i, img in enumerate(x_test):

    plt.subplot(3,3,i+1)

    plt.imshow(img.reshape(28,28), cmap='gray', interpolation='none')

    plt.title("Predicted {}, Class {}".format(categories[predict(x_test[i])], categories[np.argmax(y_test[i])]))

    plt.tight_layout()
acc_test = 0

for iter in np.arange(200):

    x_batch, y_batch = data_generator(mode='test')

    acc_test += sess.run(accuracy, feed_dict={ x: x_batch,

                                                y: y_batch,

                                                keep_prob: 1.0})

acc_test /=100



print('Test accuracy {}%'.format(acc_test*100))