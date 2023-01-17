# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)
import tensorflow as tf

#cleaning of the graph before using that

tf.reset_default_graph()



N_class = 10

X=tf.placeholder(tf.float32,[None,32,32,3])

Y=tf.placeholder(tf.int64,[None,1])

X_extend = tf.reshape(X,[-1,32,32,3])

Y_onehot = tf.one_hot(indices=Y,depth=N_class)

conv1_w = tf.get_variable("conv1_w", [3,3,3,64], initializer=tf.random_normal_initializer(stddev=1e-2))

conv1_b = tf.get_variable("conv1_b",[64], initializer=tf.random_normal_initializer(stddev=1e-2))

conv1 = tf.nn.conv2d(X_extend, conv1_w, strides= [1,1,1,1], padding='SAME') + conv1_b

relu1 = tf.nn.relu(conv1)

pool1 = tf.nn.max_pool(value=relu1,ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME') 
conv2_w = tf.get_variable("conv2_w", [3,3,64,64], initializer=tf.random_normal_initializer(stddev=1e-2))

conv2_b = tf.get_variable("conv2_b",[64], initializer=tf.random_normal_initializer(stddev=1e-2))

conv2 = tf.nn.conv2d(pool1, conv2_w, strides= [1,1,1,1], padding='SAME') + conv2_b

relu2 = tf.nn.relu(conv2)

pool2 = tf.nn.max_pool(value=relu2,ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
conv3_w = tf.get_variable("conv3_w",[3,3,64,64],initializer=tf.random_normal_initializer(stddev=1e-2))

conv3_b = tf.get_variable("conv3_b",[64],initializer=tf.random_normal_initializer(stddev=1e-2))



conv3 = tf.nn.conv2d(pool2,conv3_w,strides=[1,1,1,1],padding='SAME') + conv3_b

relu3 = tf.nn.relu(conv3)

pool3= tf.nn.max_pool(value=relu3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')



print(relu3)
flattern = tf.reshape(relu3,[-1,8*8*64])
fc1 =tf.layers.dense(inputs=flattern,units=512,activation=tf.nn.relu,use_bias=True)
fc2 = tf.layers.dense(inputs=fc1,units=512,activation=tf.nn.relu,use_bias=True)
output =tf.layers.dense(inputs=fc2,units=N_class,activation=None,use_bias=True)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_onehot,logits=output))
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output,axis=1),Y[:,0]),dtype=tf.float32))
opt = tf.train.AdamOptimizer(0.001).minimize(loss)
sess = tf.Session()

init = tf.global_variables_initializer()

sess.run(init)
import tensorflow as tf

from tqdm import tqdm_notebook as tqdm



"""Training loop"""

EPOCHS=30

BATCH_SIZE=64



for epoch in range(0,EPOCHS):

    for step in tqdm(range(int(len(x_train)/BATCH_SIZE)), desc=('Epoch '+str(epoch))):

        x_batch=x_train[step*BATCH_SIZE:step*BATCH_SIZE+BATCH_SIZE]

        y_batch=y_train[step*BATCH_SIZE:step*BATCH_SIZE+BATCH_SIZE]

        loss_value, _ =sess.run([loss,opt], feed_dict={X:x_batch, Y:y_batch})

    loss_value,accuracy_value = sess.run([loss,accuracy],feed_dict={X: x_test[:1000], Y:y_test[:1000]})

    print('Epoch loss: ', loss_value)

    print('accuracy: ', accuracy_value)
import matplotlib.pyplot as plt

plt.imshow(x_train[9])

import numpy as np

conv1_w_extract= sess.run(conv1_w)

print(conv1_w_extract.shape)

plt.figure(figsize=(20,20))

for i in range(10):

	plt.subplot(1,10,i+1)

	plt.imshow(np.reshape(conv1_w_extract[:,:,:,i]*100, [3,3,3]))
conv1_fmaps = sess.run(relu1,feed_dict= {X:[x_train[0]]})

print(conv1_fmaps.shape)

plt.figure(figsize=(34,34))





for i in range(10):

    plt.subplot(1,10,i+1)

    plt.imshow(np.reshape(conv1_fmaps[0,:,:,i],[32,32]))