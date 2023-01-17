# An initializer for the bias vector. If None, the default initializer will be used.# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import tensorflow as tf  
import matplotlib.pyplot as plot
trainDF = pd.read_csv('../input/digit-recognizer/train.csv')
testDF = pd.read_csv('../input/digit-recognizer/test.csv')
print(trainDF.shape, testDF.shape)
X_train = trainDF.iloc[:,1:]
for i in range(10) :
    plot.subplot(1,10,i+1)
    plot.imshow(X_train.values[i].reshape(28,28))
    
plot.show()

print((trainDF.iloc[0:10, 0]).tolist())
for i in range(10) :
    plot.subplot(1,10,i+1)
    plot.imshow(testDF.values[i].reshape(28,28))
    
plot.show()
trainLabels = trainDF['label'].tolist()
ohTrainLabelsTensor = tf.one_hot(trainLabels, depth=10)
lblTrainArr = tf.Session().run(tf.cast(ohTrainLabelsTensor, tf.float32))
#Variables
num_input = 28*28*1 #img shape (28*28)
num_classes = 10 ##total classes (0-9 digits)

#placeholders are used to send and get information from graph
x_ = tf.placeholder("float", shape=[None, num_input], name='X')
y_ = tf.placeholder("float", shape=[None, num_classes], name='Y')

is_training = tf.placeholder(tf.bool) #Add dropout to fully connected layer
#convert the feature vector to 28*28*1 image
x_image = tf.reshape(x_, [-1,28,28,1])  ## -1 represents here variable size of batches
#convolution layer 1
conv1 = tf.layers.conv2d(inputs=x_image, 
                         filters=32, 
                         kernel_size=[5, 5], 
                         padding="same", 
                         activation=tf.nn.relu)

print(conv1.shape)
#pooling layer 1
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)
print(pool1.shape)
#convolution layer 2
conv2 = tf.layers.conv2d(inputs=pool1, 
                         filters=64, 
                         kernel_size=[5, 5], 
                         padding="same", 
                         activation=tf.nn.relu)

print(conv2.shape)
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)
print(pool2.shape)
pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
print(dense.shape)
dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=is_training)
logits = tf.layers.dense(inputs=dropout, units=10)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
Y = tf.nn.softmax(logits)
predictions = tf.argmax(Y, 1)
for i in range (10000) :
    startIndx = (i * 50) %2000
    endIndx = startIndx + 50
    batch_train = trainDF[startIndx : endIndx] 
    batch_test = testDF[startIndx : endIndx]
    
    batch_label_train =  lblTrainArr[startIndx : endIndx]
    batch_image_train =  batch_train.loc[:, batch_train.columns != 'label']
        
    
    if i % 50 == 0:
        train_accuracy = accuracy.eval(session = sess, feed_dict={x_ : batch_image_train, y_ : batch_label_train, is_training:True})
        print("step %d, training accuracy %g" %(i, train_accuracy))
        train_step.run(session=sess, feed_dict={x_ : batch_image_train, y_ : batch_label_train, is_training:True})
        

    if i % 100 == 0:
        test_accuracy = accuracy.eval(session = sess, feed_dict={x_ : batch_test, y_ : batch_label_train, is_training:False})
        print("step %d, test_accuracy %g" %(i, train_accuracy))
prediction = sess.run(tf.argmax(logits,1), feed_dict={x_: testDF, is_training:False})
print ("Prediction for test image:", np.squeeze(prediction))
ids = list(range(prediction.shape[0]))

sub = pd.DataFrame({
    "ImageId": ids,
    "Label": np.squeeze(prediction)
})

sub.to_csv("./submission.csv", index=False)