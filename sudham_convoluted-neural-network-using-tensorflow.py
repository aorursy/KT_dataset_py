import tensorflow as tf

import numpy as np

import math

import os

from six.moves import xrange

import matplotlib.pyplot as plt

%matplotlib inline

import pandas as pd

import sklearn.preprocessing as pre
#Input data

#read datasets

train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")



#Prepare the training images

images = train_data.drop('label',1)

observations, features = images.shape

pixel_width = int(np.sqrt(features))

X = images.as_matrix()

X_train = X.reshape(observations, pixel_width, pixel_width,1)

print("Image Array", X_train.shape)



#Prepare the labels

labels = train_data['label']

Y = labels.as_matrix()

labels = pre.LabelEncoder().fit_transform(labels)[:, None]

Y_train = pre.OneHotEncoder().fit_transform(labels).todense()

print("Label Array", Y_train.shape)



#Prepare the test images

t = test_data.as_matrix()

tr, tc = t.shape

test=t.reshape(tr, pixel_width, pixel_width,1)

print("Image Array", test.shape)
#Peek into the data (take a look at some random images)

def showImage(X, index):

    N, w, h, c = X.shape

    grid = np.zeros((w, h))

    for i in range(w):

        for j in range(h):

            grid[i,j] = X[index,i,j,0]

    plt.rcParams["figure.figsize"] = [1.5,1.5]

    plt.imshow(grid, cmap='gray')

    plt.ion()

    plt.show()

    

showImage(X_train,3)

showImage(X_train,875)

showImage(X_train,40000)

#Define the training model : 

#Image and Label

X = tf.placeholder("float",[None,28, 28, 1])   

Y = tf.placeholder("float",[None,10])   



# Learning rate with a decay

lr = tf.placeholder(tf.float32)



# Dropout: None for test data and 25% for training data 

pkeep = tf.placeholder(tf.float32)

pkeep_conv = tf.placeholder(tf.float32)



# three convolutional layers with their channel counts, and a

K = 24  # first convolutional layer output depth

L = 48  # second convolutional layer output depth

M = 64  # third convolutional layer

N = 200  # fully connected layer



W1 = tf.Variable(tf.truncated_normal([6, 6, 1, K], stddev=0.1))  # 5x5 patch, 1 input channel, K output channels

B1 = tf.Variable(tf.ones([K])/10)



W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))

B2 = tf.Variable(tf.ones([L])/10)



W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))

B3 = tf.Variable(tf.ones([M])/10)



W4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1))

B4 = tf.Variable(tf.ones([N])/10)



W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))

B5 = tf.Variable(tf.ones([10])/10)
def batchnorm(Ylogits,beta,convolutional=False):

    #exp_moving_avg = tf.train.ExponentialMovingAverage(0.998, iteration)

    bnepsilon = 1e-5

    if convolutional:

        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])

    else:

        mean, variance = tf.nn.moments(Ylogits, [0])

    BN = tf.nn.batch_normalization(Ylogits,mean,variance,beta,None,bnepsilon)

    return BN
stride = 1  # output is 28x28

Y1 = tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME')

BN1 = batchnorm(Y1, B1,convolutional=True)

Y1_BN = tf.nn.relu(BN1)



stride = 2  # output is 14x14

Y2 = tf.nn.conv2d(Y1_BN, W2, strides=[1, stride, stride, 1], padding='SAME')

BN2 = batchnorm(Y2, B2,convolutional=True)

Y2_BN = tf.nn.relu(BN2)



stride = 2  # output is 7x7

Y3 = tf.nn.conv2d(Y2_BN, W3, strides=[1, stride, stride, 1], padding='SAME')

BN3 = batchnorm(Y3, B3,convolutional=True)

Y3_BN = tf.nn.relu(BN3)
# reshape the output from the third convolution for the fully connected layer

YY = tf.reshape(Y3, shape=[-1, 7 * 7 * M])
Y4 = tf.matmul(YY, W4)

BN4 = batchnorm(Y4, B4)

Y4_BN = tf.nn.relu(BN4)

YY4 = tf.nn.dropout(Y4_BN, pkeep)



Ylogits = tf.matmul(YY4, W5) + B5

YHat = tf.nn.softmax(Ylogits)



cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y)

cross_entropy = tf.reduce_mean(cross_entropy)

correct_prediction = tf.equal(tf.argmax(YHat, 1), tf.argmax(Y_train, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

optimizer = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

predict = tf.argmax(YHat,1)
#Train the model 

training_epochs = 2

batch_size = 100

max_learning_rate = 0.02

min_learning_rate = 0.0001

decay_speed = 1600.0 #

init =tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)

    total_batch = int(observations/batch_size)

    batch_no = 1

    print ("Optimization In Progress")

    for epoch in range(training_epochs):

        c=0.0

        avg_cost=0.0

        # Loop over all batches

        for i in range(total_batch):

            start = i*100

            end = start +batch_size-1

            #print("Start image no: ", start ," to end image no: " ,end)

            learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)

            batch_X = X_train[start:end]

            batch_Y = Y_train[start:end]

            #print(batch_X.shape)

            _,c=sess.run([optimizer,cross_entropy], feed_dict={X:batch_X,Y:batch_Y,lr: learning_rate,pkeep: 0.75})

            avg_cost += c / total_batch



        print("epoch No {} cross entropy={}".format(epoch+1,avg_cost))

            

    print ("Optimization Completed")

    #print( "Accuracy:", accuracy.eval({X: X_train, Y: Y_train, pkeep: 1.0}))

    

    print("Predictions")

    

    test_batch = batch_size

    predictions = np.zeros(test.shape[0])

    for i in range(0,test.shape[0]//test_batch):

        predictions[i*test_batch : (i+1)*test_batch] = predict.eval(feed_dict={X: test[i*test_batch : (i+1)*test_batch], 

                                                                                pkeep: 1.0})
for i in range(0,test.shape[0],1000):

    print("Predicted Value", predictions[i])

    showImage(test, i)
#Prepare test submission

submission = pd.DataFrame(data={'ImageId':(np.arange(test.shape[0])+1), 'Label':predictions})

submission.to_csv('submission.csv', index=False)

submission.head()