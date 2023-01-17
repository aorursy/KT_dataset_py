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
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#載入數據集
#mnist=input_data.read_data_sets("MNIST_data",one_hot=True)
mnist = input_data.read_data_sets('../input', one_hot=True)

#每個批次的大小
batch_size=100

#計算共有多少個批次
n_batch=mnist.train.num_examples // batch_size

#定義兩個placeholder
x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])

#創建一個簡單的神經網路
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
prediction=tf.nn.softmax(tf.matmul(x,W)+b)

#二次代價函數
loss=tf.reduce_mean(tf.square(y-prediction))
#使用梯度下降法
train_step=tf.train.GradientDescentOptimizer(0.2).minimize(loss)

#初始化變量
init=tf.global_variables_initializer()

#結果存放在一個布爾型陣列表中
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1)) #argmax返回一違張量中最大值的所在位置

#求準確率
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
            
        acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Iter "+str(epoch)+",Test  Accuracy "+str(acc))

