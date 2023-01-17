



import numpy as np 

import pandas as pd 

import tensorflow as tf

import sklearn

import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv("../input/train.csv")

test = data = pd.read_csv("../input/test.csv")

print (train.shape)

print (test.shape)

train.head(5)
target = train['label']

features = train.drop('label',axis=1)
import matplotlib.pyplot as plt

%matplotlib inline



def showImage(index):

    label = target[index].argmax(axis=0)

    img = features.iloc[index].reshape([28,28])

    plt.title("Index: {} , Label: {}".format(index, label))

    plt.imshow(img, cmap='gray')

    plt.show()

    

showImage(50)    
#Reshape the image to [28,28]

#features = np.array(features)

#features = np.reshape(features, [-1,28,28,1])

for i in range(len(features)):

    features.iloc[i].reshape([28,28])
def conv_layer(x, height, width, input_channels, output_channels):

    weights = tf.Variable(tf.truncated_normal([height, width, input_channels, output_channels], stddev=0.1))

    biases  = tf.Variable(tf.zeros(output_channels))

    layer = (tf.nn.conv2d(x, weights, strides = [1,2,2,1], padding = 'SAME') + biases)

    layer = tf.nn.relu(layer)

    layer = tf.nn.max_pool(layer, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

    return layer



def fully_connected(x, outputs):

    weights = tf.Variable(tf.truncated_normal([x.get_shape().as_list()[1], outputs], stddev=0.1))

    biases = tf.Variable(tf.zeros(outputs))

    layer = tf.matmul(x, weights) + biases

    layer = tf.nn.relu(layer)

    return layer



final_weights = tf.Variable(tf.truncated_normal([56 ,10], stddev=0.1))

final_biases = tf.Variable(tf.ones(10))

X = tf.placeholder(tf.float32, [None,28,28,1])

y = tf.placeholder(tf.float32, [None,10])

keep_prob = tf.placeholder(tf.float32)



layer1 = conv_layer(X, 1, 1, 1, 4)

layer2 = conv_layer(layer1, 2, 2, 4, 8)

final = tf.contrib.layers.flatten(layer1)



final1 = fully_connected(final, 28)

final1 = tf.nn.dropout(final1, keep_prob)



final2 = fully_connected(final1, 56)

final2 = tf.nn.dropout(final2, keep_prob)



logits = tf.matmul(final2, final_weights) + final_biases



epochs = 5

keep_prob = 0.5



# Loss and Optimizer

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

optimizer = tf.train.AdamOptimizer(0.05).minimize(cost)



# Accuracy

correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for i in range(epochs):

        sess.run(optimizer, feed_dict= {X: features, y: target, keep_prob: keep_prob})

        print ("Loss: {}".format(session.run(cost, feed_dict = {X: features, y: target})))

            

        