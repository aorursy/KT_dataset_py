import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

test = pd.read_csv('../input/test.csv')
test = StandardScaler().fit_transform(np.float32(test.values))
test = test.reshape(-1, 28, 28, 1)
k = 7 #choose which image to take from the test set
kernel = 4 #Kernel size
strides = 1 #Stride size
kernel_mp = 2 #Kernel size of the max pooling operation
strides_mp = 1 #Stride size in max pooling operation
plt.imshow(test[k,:,:,0])
plt.show()
graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    w = tf.Variable(tf.truncated_normal([kernel, kernel, 1, 16], stddev=0.1))
    cl = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')
    cl_mp = tf.nn.max_pool(cl, ksize=[1, kernel_mp, kernel_mp, 1], strides=[1, strides_mp, strides_mp, 1], padding='SAME')
with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    CL = sess.run(cl, feed_dict={x:[test[k]]})
    print("Image Shape =", CL.shape[1:3])
    f, ax = plt.subplots(4,4, figsize=(20,20))
    for i in range(16):
        ax[i//4,i%4].imshow(CL[0,:,:,i])
        ax[i//4,i%4].axis('off')
    plt.show()
with tf.Session(graph=graph) as sess: #start the session
    tf.global_variables_initializer().run() #initialize variables
    CL = sess.run(cl_mp, feed_dict={x:[test[k]]})
    print("Image Shape =", CL.shape[1:3])
    f, ax = plt.subplots(4,4, figsize=(20,20))
    for i in range(16):
        ax[i//4,i%4].imshow(CL[0,:,:,i])
        ax[i//4,i%4].axis('off')
    plt.show()
graph = tf.Graph()
with graph.as_default():

    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    w1 = tf.Variable(tf.truncated_normal([8, 8, 1, 16], stddev=0.1))
    w2 = tf.Variable(tf.truncated_normal([5, 5, 16, 32], stddev=0.1))
    w3 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.1))
    
    # Convolutional layer 1
    cl1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME')
    cl1_mp = tf.nn.max_pool(cl1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')        
    # Convolutional layer 2
    cl2 = tf.nn.conv2d(cl1_mp, w2, strides=[1, 1, 1, 1], padding='SAME')
    cl2_mp = tf.nn.max_pool(cl2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # Convolutional layer 3
    cl3 = tf.nn.conv2d(cl2_mp, w3, strides=[1, 1, 1, 1], padding='SAME')
    cl3_mp = tf.nn.max_pool(cl3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#After Layer 1
with tf.Session(graph=graph) as sess: #start the session
    tf.global_variables_initializer().run() #initialize variables
    CL = sess.run(cl1_mp, feed_dict={x:[test[k]]})
    print("Image Shape =", CL.shape[1:3])
    print("Number of Images =", CL.shape[-1])
    f, ax = plt.subplots(4,4, figsize=(20,20))
    for i in range(16):
        ax[i//4,i%4].imshow(CL[0,:,:,i])
        ax[i//4,i%4].axis('off')
    plt.show()
#After Layer 2
with tf.Session(graph=graph) as sess: #start the session
    tf.global_variables_initializer().run() #initialize variables
    CL = sess.run(cl2_mp, feed_dict={x:[test[k]]})
    print("Image Shape =", CL.shape[1:3])
    print("Number of Images =", CL.shape[-1])
    f, ax = plt.subplots(4,8, figsize=(20,10))
    for i in range(32):
        ax[i//8,i%8].imshow(CL[0,:,:,i])
        ax[i//8,i%8].axis('off')
    plt.show()
#After Layer 3
with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    CL = sess.run(cl3_mp, feed_dict={x:[test[k]]})
    print("Image Shape =", CL.shape[1:3])
    print("Number of Images =", CL.shape[-1])
    f, ax = plt.subplots(8,8, figsize=(20,20))
    for i in range(64):
        ax[i//8,i%8].imshow(CL[0,:,:,i])
        ax[i//8,i%8].axis('off')
    plt.show()
