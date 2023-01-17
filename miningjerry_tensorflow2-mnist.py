# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import tensorflow as tf



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



print(train.shape)

data = train.iloc[:,1:]

label = train.iloc[:,0]

print(data.shape, label.shape)

print(test.shape)
# 分割测试样本

from sklearn.model_selection import train_test_split

X_train, X_validation, y_train, y_validation = train_test_split(

    data, label, test_size=0.2, random_state=0)

print(X_train.shape, X_validation.shape)
import random 

x = tf.placeholder('float', [None, 784])

W = tf.Variable(tf.zeros([784, 10]))

b = tf.Variable(tf.zeros([10]))



y = tf.nn.softmax(tf.matmul(x, W) + b)



#计算交叉熵

y_ = tf.placeholder('float', [None, 10])

cross_entropy = - tf.reduce_sum(y_ * tf.log(y))



# 梯度下降，以0.01的学习率最小化交叉熵

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)



init = tf.initialize_all_variables()





correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
batch_xs,  x_other, batch_ys, y_other = train_test_split(

    data, label, test_size=0.99, random_state=0)



print(batch_xs.shape, batch_ys.shape)

print(data.shape, label.shape)
# 将label转换为one_hot

label_matrix = [0]*10*label.shape[0]

print(len(label_matrix))

label_matrix = np.array(label_matrix)

label_matrix = label_matrix.reshape(label.shape[0],10,)

print(label_matrix.shape)

for i, l in enumerate(label[:10]):

    label_matrix.flat[i*10 + l] = 1

    

print(label_matrix)
# 分割训练集和测试集

X_train, y_train = data[:30000], label_matrix[:30000]

X_test, y_test = data[30000:], label_matrix[30000:]
with tf.Session() as sess:

    sess.run(init)

    

    #训练模型，循环1000次

    for i in range(1000):

        batch_xs, batch_ys, = X_train[i:i+100], y_train[i:i+100]

            

        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        if (i+1) %100 ==0:

            print("train {0} samples".format(i+1))

    print ("test accuracy %g"%accuracy.eval(feed_dict={x:  X_test, y_: y_test}))
cross_entropy = - tf.reduce_sum(y_ * tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))



init = tf.initialize_all_variables()

with tf.Session() as sess:

    sess.run(init)

    for i in range(1000):

        

        if i%100 == 0:

            train_accuracy = accuracy.eval(feed_dict={x: X_train[i:i+100], y_: y_train[i:i+100]})

            print ("step %d, training accuracy %g"%(i, train_accuracy))

            

        train_step.run(feed_dict={x: X_train[i:i+100], y_: y_train[i:i+100]})



        

    print ("test accuracy %g"%accuracy.eval(feed_dict={

            x:  X_test, y_: y_test}))