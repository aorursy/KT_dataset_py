import numpy as np

import pandas as pd 

import tensorflow as tf

import matplotlib.pyplot as plt
alphabet = pd.read_csv('../input/emnist-letters-train.csv')

#shuffle the data set

alphabet=alphabet.sample(frac=1)

#split features and labels

images=alphabet.iloc[:,1:].values

raw_labels=alphabet.iloc[:,0].values.ravel()

print(images.shape,raw_labels.shape)
def one_hot(labels):

    num_labels=labels.shape[0]

    result=np.zeros((num_labels,26))

    offset=np.arange(num_labels)*26

    result.flat[offset+labels.ravel()]=1

    return result  

labels=one_hot(raw_labels)

print(labels[51211],raw_labels[51211])
validation=20000



train_images = images[validation:]

train_labels = labels[validation:]



validation_images = images[:validation]

validation_labels = labels[:validation]



n_batch = train_images.shape[0] / 100
x=tf.placeholder(tf.float32,[None,784])

y=tf.placeholder(tf.float32,[None,26])

keep_prob = tf.placeholder(tf.float32)
def weight_variable(shape):

    return tf.Variable(tf.random_normal(shape))





def bias_variable(shape):

    return tf.Variable(tf.random_normal(shape))





def conv2d(x, W):

    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')





def max_pool_2x2(x):

    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
x_image = tf.reshape(x, [-1, 28, 28, 1])
#How we get the parameter

#https://github.com/wwzzyyzzrr/DaChuang/blob/97e07590453a9d6fb3a644ce6abc4a99e2c2d015/Recognition/prediction.py

W_conv1 = weight_variable([11, 11, 1, 64])

b_conv1 = bias_variable([64])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

h_pool1 = max_pool_2x2(h_conv1)

#h_pool1 = tf.nn.dropout(h_pool1, 0.8)
W_conv2 = weight_variable([5, 5, 64,192])

b_conv2 = bias_variable([192])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

h_pool2 = max_pool_2x2(h_conv2)
W_conv3 = weight_variable([3, 3, 192,384])

b_conv3 = bias_variable([384])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
W_conv4 = weight_variable([3, 3, 384,256])

b_conv4 = bias_variable([256])

h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)
W_conv5 = weight_variable([3, 3, 256,256])

b_conv5 = bias_variable([256])

h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5) + b_conv5)

h_pool5 = max_pool_2x2(h_conv5)
dense1 = tf.reshape(h_pool5, [-1, weight_variable([4*4*256, 1024]).get_shape().as_list()[0]])

dense1 = tf.nn.relu(tf.matmul(dense1, weight_variable([4*4*256, 1024])) + bias_variable([1024]),name='fc1')

dense2 = tf.nn.relu(tf.matmul(dense1, weight_variable([1024, 1024])) + bias_variable([1024]),name='fc2')



y_conv = tf.matmul(dense2,  weight_variable([1024, 26])) + bias_variable([26])
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_conv))

train_step = tf.train.AdadeltaOptimizer(learning_rate=0.1).minimize(loss)





correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_conv, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



global_step = tf.Variable(0, name='global_step', trainable=False)

saver = tf.train.Saver()



init = tf.global_variables_initializer()
with tf.Session() as sess:

     sess.run(init)



     for epoch in range(1,20):

         for batch in range(int(n_batch)):

             batch_x = train_images[batch*100:(batch+1)*100]

             batch_y = train_labels[batch*100:(batch+1)*100]



             sess.run(train_step,feed_dict = {x:batch_x,y:batch_y,keep_prob:0.5})



         accuracy_n = sess.run(accuracy,feed_dict={x:validation_images, y:validation_labels,keep_prob:1.0})

         print("Round:" + str(epoch) +",accuracy:"+str(accuracy_n))



         #30

         global_step.assign(epoch).eval()

         saver.save(sess,"../model.ckpt",global_step = global_step)