import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))
# IMPORTING DATA 

train=pd.read_csv('../input/fashion-mnist_train.csv')

test=pd.read_csv('../input/fashion-mnist_test.csv')
# For Plotting

import matplotlib.pyplot as plt
train.shape , test.shape
train_label=train.iloc[:,0]

del train['label']
plt.imshow(train.values[250].reshape(28,28))

print('This is product number',train_label[250])
plt.imshow(train.values[20].reshape(28,28))

print('This is product number',train_label[20])
train_label=train_label.astype('category')
train_label=pd.get_dummies(train_label)

train_label.head()
import tensorflow as tf
# Input 

x=tf.placeholder(tf.float32,name='x',shape=[None,784])

y=tf.placeholder(tf.float32,name='y',shape=[None,10])

keep=tf.placeholder(tf.float32)
def function(x,w_shape,b_shape):

    w_init=tf.random_normal_initializer(stddev=0.2)

    b_init=tf.constant_initializer(0.2)

    w=tf.get_variable(name='w',shape=w_shape,initializer=w_init)

    b=tf.get_variable(name='b',shape=b_shape,initializer=b_init)

    return (tf.add(tf.matmul(x,w),b))
x=tf.reshape(x,shape=[-1,28,28,1])
def conv2d(x,weight_shape,bias_shape):

    w_init=tf.truncated_normal_initializer(stddev=0.3)

    b_init=tf.constant_initializer(0.1)

    w=tf.get_variable(name='w',shape=weight_shape,initializer=w_init)

    b=tf.get_variable(name='b',shape=bias_shape,initializer=b_init)

    out=tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

    return tf.add(out,b)
def max_pool(x,k=2):

    return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME')
with tf.variable_scope('layer_1'):

    hidden1=conv2d(x,[3,3,1,32],[32])

    out1=max_pool(hidden1)

with tf.variable_scope('layer_2'):

    hidden2=conv2d(out1,[3,3,32,64],[64])

    out2=max_pool(hidden2)

with tf.variable_scope('layer_3'):

    hidden3=conv2d(out2,[3,3,64,128],[128])

    out3=max_pool(hidden3)

with tf.variable_scope('layer_4'):

    new=tf.reshape(out3,shape=[-1,4*4*128])

    out=function(new,[4*4*128,1024],[1024])

    out2=tf.nn.dropout(out,keep)

with tf.variable_scope('layer_5'):

    out1=function(out2,[1024,10],[10])
#LOSS FUNCTION

cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=out1))
# Optimizer

opti=tf.train.AdamOptimizer(learning_rate=0.001)

step=opti.minimize(cross_entropy)
# Accuracy checker

correct=tf.equal(tf.argmax(out1,1),tf.argmax(y,1))

accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))
# For mini batch

iteration=3000

batch_size=128
df=train.values
df_label=train_label.values
sess=tf.Session()

var_init=tf.initialize_all_variables()

sess.run(var_init)
for i in range(iteration):

    choice=np.random.choice(60000,size=batch_size)

    df=df.reshape([-1,28,28,1])

    sess.run(step,feed_dict={x:df[choice],y:df_label[choice],keep:0.4})

    if i%200 == 0:

        loss,accu =sess.run([cross_entropy,accuracy],feed_dict={x:df[choice],y:df_label[choice],keep:1})

        print ('loss is',loss,'|   Accuracy is',accu)
test_label=test.iloc[:,0]
del test['label']
test=test.values
test=test.reshape([-1,28,28,1])
test_label=test_label.astype('category')
test_label=pd.get_dummies(test_label)
test_label=test_label.values
loss,accu=sess.run([cross_entropy,accuracy],feed_dict={x:test,y:test_label,keep:1})

print ('accuracy is',accu,'|   Loss is',loss)