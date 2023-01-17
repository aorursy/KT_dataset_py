import numpy as np

import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt

from tqdm import tqdm, tqdm_notebook

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

print(train.shape)

print(test.shape)



X = train.drop('label', axis = 1).values

y = train['label'].values

x_test_to_submit = test.values

del train, test

print(x.shape, y.shape, x_test_to_submit.shape)
#function to One-Hot-Encoding target variable

#example: 3 ---> [0,0,0,1,0,0,0,0,0,0]

def encodeTarget(Y):

    temp = np.zeros((y.size, y.max()+1))

    temp[np.arange(y.size), y] = 1

    return temp



y = encodeTarget(y)



#function for digits visualization  

def printDigit(matrix, hot_label):

    matrix = matrix.reshape((28, 28))

    label = np.where(hot_label==1)[0][0]  #decode hot-encoded label

    plt.figure(figsize = (1.5,1.5))

    plt.title('Label is {label}'.format(label=label))

    plt.imshow(matrix, cmap='plasma')

    plt.show()

    

printDigit(X[7,:], y[7])

printDigit(X[777,:], y[777])
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])  #features

y_ = tf.placeholder(tf.float32, shape=[None, 10])  #true targets
# weight inotialization

def weight_variable(shape):

    initial = tf.truncated_normal(shape, mean=0.0, stddev=0.1)   #init by random values ~N(0.0,0.1)

    return tf.Variable(initial)



def bias_variable(shape):

    initial = tf.constant(0.1, shape=shape)   #init by constant

    return tf.Variable(initial)
# convolution and pooling



def conv2d(x, W):

    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')



def max_pool_2x2(x):

    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],

                        strides=[1, 2, 2, 1], padding='SAME')
# First Convolutional Layer

W_conv1 = weight_variable([5, 5, 1, 32])     #32 maps, tune

b_conv1 = bias_variable([32])



x_image = tf.reshape(x, [-1,28,28,1])

#-1 - is the special value, the size of that dimension is computed so that the total size remains constant

#28,28 - width and height

#1 - RGB numbers (we have black and white pics, therefore we have only 1 number)



h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

h_pool1 = max_pool_2x2(h_conv1)
# second convolutional layer



W_conv2 = weight_variable([5, 5, 32, 64])    #64 maps, tune

b_conv2 = bias_variable([64])



h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

h_pool2 = max_pool_2x2(h_conv2)  #28*28 --> 14*14
# densely connected layer

 

W_fc1 = weight_variable([7 * 7 * 64, 1024])

b_fc1 = bias_variable([1024])



h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# dropout

keep_prob = tf.placeholder(tf.float32)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# readout layer

W_fc2 = weight_variable([1024, 10])

b_fc2 = bias_variable([10])



y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
#get next batch

def next_batch(batch_size):

    global data_index

    batch = X[data_index:data_index+batch_size,:]

    labels = y[data_index:data_index+batch_size,:]

    return (batch, labels)
%%time

# Train the Model



data_index = 0

batch_size = 25



cross_entropy = tf.reduce_mean(

    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())

for i in range(1000):

    batch = next_batch(batch_size)

    if i%100 == 0:

        train_accuracy = accuracy.eval(feed_dict={

            x:batch[0], y_: batch[1], keep_prob: 1.0})

        print("step %d, training accuracy %g"%(i, train_accuracy))

    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})



#print("test accuracy %g"%accuracy.eval(feed_dict={

#    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
#predict by batch (without such tric Memory Error raises):

size = 100

batches_generator = (x_test_to_submit[i:i + size] for i in range(0, len(x_test_to_submit), size))

predictions = []

y_pred_digits = tf.argmax(y_conv,1)

for test_batch in batches_generator:

    predictions.extend(y_pred_digits.eval(feed_dict={x:test_batch, keep_prob: 1.0}))
#save results

header = "ImageId,Label\n"

with open("results.csv", 'w') as f:

    f.write(header)

    f.write("\n".join(("{},{}".format(ind+1, dig) for ind,dig in enumerate(predictions))))
ls