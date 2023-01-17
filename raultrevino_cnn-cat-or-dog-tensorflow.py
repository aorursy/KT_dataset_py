import os

import cv2 

import pandas as pd 

import matplotlib.pyplot as plt

import matplotlib.image as mpimg 

import numpy as np

import tensorflow as tf

from tqdm import tqdm  

from tensorflow.python.framework import ops

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split
TRAIN_DIR = 'train/'

IMG_SIZE = 32

CHANNEL_NUMBER = 1

LR = 1e-3
def label_img(img):

    word_label = img.split('.')[-3]            

    if word_label == 'cat': return [1,0] #[much cat, no dog]

    elif word_label == 'dog': return [0,1]  #[no cat, very doggo]
def create_train_data():

    training_data = []

    for img in tqdm(os.listdir(TRAIN_DIR)):

        label = label_img(img)

        path = os.path.join(TRAIN_DIR,img)

        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))

        training_data.append([np.array(img),np.array(label)])

    shuffle(training_data)

    return training_data
train_data = create_train_data()
from sklearn.model_selection import train_test_split

train,test  = train_test_split(train_data,test_size = 0.3)
# The -1 in the reshape() function means that it will infer the first dimension on its own 

#but the rest of the dimension are fixed, that is, IMG_SIZE x IMG_SIZE x CHANNEL_NUMBER.

train_X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,CHANNEL_NUMBER)

train_Y = [i[1] for i in train]



test_X = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,CHANNEL_NUMBER)

test_Y = [i[1] for i in test]
TRAINING_ITERS = 50

BATCH_SIZE = 128

N_CLASSES = 2
X = tf.placeholder("float", [None, IMG_SIZE,IMG_SIZE,CHANNEL_NUMBER])

Y = tf.placeholder("float", [None, N_CLASSES])
def conv2d(x, W, b, strides=1):

    # Conv2D wrapper, with bias and relu activation

    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')

    x = tf.nn.bias_add(x, b)

    return tf.nn.relu(x) 



def maxpool2d(x, k=2):

    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')
weights = {

    'wc1': tf.get_variable('W0', shape=(3,3,1,32), initializer=tf.contrib.layers.xavier_initializer()), 

    'wc2': tf.get_variable('W1', shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()), 

    'wc3': tf.get_variable('W2', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()), 

    'wd1': tf.get_variable('W3', shape=(4*4*128,128), initializer=tf.contrib.layers.xavier_initializer()), 

    'out': tf.get_variable('W6', shape=(128,N_CLASSES), initializer=tf.contrib.layers.xavier_initializer()), 

}

biases = {

    'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),

    'bc2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),

    'bc3': tf.get_variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),

    'bd1': tf.get_variable('B3', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),

    'out': tf.get_variable('B4', shape=(N_CLASSES), initializer=tf.contrib.layers.xavier_initializer()),

}
def conv_net(x, weights, biases):  



    # here we call the conv2d function we had defined above and pass the input image x, weights wc1 and bias bc1.

    conv1 = conv2d(x, weights['wc1'], biases['bc1'])

    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 14*14 matrix.

    conv1 = maxpool2d(conv1, k=2)



    # Convolution Layer

    # here we call the conv2d function we had defined above and pass the input image x, weights wc2 and bias bc2.

    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])

    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 7*7 matrix.

    conv2 = maxpool2d(conv2, k=2)



    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])

    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 4*4.

    conv3 = maxpool2d(conv3, k=2)



    # Fully connected layer

    # Reshape conv2 output to fit fully connected layer input

    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])

    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])

    fc1 = tf.nn.relu(fc1)

    # Output, class prediction

    # finally we multiply the fully connected layer with the weights and add a bias term. 

    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    return out
pred = conv_net(X, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(cost)

# Here you check whether the index of the maximum value of the predicted image is equal 

# to the actual labelled image. and both will be a column vector.

correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))

# Calculate accuracy across all the given images and average them out. 

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init = tf.global_variables_initializer()
with tf.Session() as sess:

    sess.run(init) 

    train_loss = []

    test_loss = []

    train_accuracy = []

    test_accuracy = []

   

    for i in range(TRAINING_ITERS):

        for batch in range(len(train_X)//BATCH_SIZE):

            batch_x = train_X[batch*BATCH_SIZE:min((batch+1)*BATCH_SIZE,len(train_X))]

            batch_y = train_Y[batch*BATCH_SIZE:min((batch+1)*BATCH_SIZE,len(train_Y))]    

            # Calculate batch loss and accuracy

            opt = sess.run(optimizer, feed_dict={X:batch_x,Y:batch_y})

            loss, acc = sess.run([cost, accuracy], feed_dict={X: batch_x, Y: batch_y})

            

        print("Iter " + str(i) + ", Loss= " + \

                      "{:.6f}".format(loss) + ", Training Accuracy= " + \

                      "{:.5f}".format(acc))

        

        print("Optimization Finished!")



        test_acc,valid_loss = sess.run([accuracy,cost], feed_dict={X:test_X,Y:test_Y})

        train_loss.append(loss)

        test_loss.append(valid_loss)

        train_accuracy.append(acc)

        test_accuracy.append(test_acc)

        

        print("Testing Accuracy:","{:.5f}".format(test_acc))