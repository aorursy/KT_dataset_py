import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))
import tensorflow as tf

print(tf.__version__)

#使用 placeholder 占位

x = tf.placeholder(tf.float32, [None, 784])

y = tf.placeholder(tf.float32, [None, 10])

#将图片转换成4维

x_image = tf.reshape(x, [-1, 28, 28, 1])
def weight_variable(shape):

    initial = tf.truncated_normal(shape, stddev=0.1)

    return tf.Variable(initial)



def bias_variable(shape):

    initial = tf.constant(0.1, shape=shape)

    return tf.Variable(initial)
#定义第一个卷积层

conv1_kernel = weight_variable([3, 3, 1, 32])

conv1_bias = bias_variable([32])

conv1_function = tf.nn.relu(tf.nn.conv2d(x_image, conv1_kernel, strides=[1, 1, 1, 1], padding='SAME') + conv1_bias)

#池化

conv1_pool = tf.nn.max_pool(conv1_function, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#定义第二个卷积层

conv2_kernel = weight_variable([3, 3, 32, 64])

conv2_bias = bias_variable([64])

conv2_function = tf.nn.relu(tf.nn.conv2d(conv1_pool, conv2_kernel, strides=[1, 1, 1, 1], padding='SAME') + conv2_bias)

#池化

conv2_pool = tf.nn.max_pool(conv2_function, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# 然后连接一个1024全连接层

fc1_w = weight_variable([7 * 7 * 64, 1024])

fc1_b = bias_variable([1024])

conv2_pool_flat = tf.reshape(conv2_pool, [-1, 7*7*64])

fc1_f = tf.nn.relu(tf.matmul(conv2_pool_flat, fc1_w) + fc1_b)
# 添加 dropout 层，抑制过拟合

keep_prob = tf.placeholder(tf.float32)

fc1_drop = tf.nn.dropout(fc1_f, keep_prob)
#再连接一个256全连接层

fc2_w = weight_variable([1024, 256])

fc2_b = bias_variable([256])

fc2_f = tf.nn.relu(tf.matmul(fc1_drop, fc2_w) + fc2_b)
# 添加 dropout 层，抑制过拟合

fc2_drop = tf.nn.dropout(fc2_f, keep_prob)
#接 SOFTMAX 分类

fc_s_w = weight_variable([256, 10])

fc_s_b = bias_variable([10])

y_hat = tf.nn.softmax(tf.matmul(fc2_drop, fc_s_w) + fc_s_b)
# get loss

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_hat), reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# get acc

correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))

#强制转换

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#创建上下文

sess = tf.InteractiveSession()

#初始化 变量

tf.global_variables_initializer().run()



def getRoundTestData(r):

    return train_X[ r*4200 : r*4200 + 4200 ]/255, train_Y[ r*4200 : r*4200 + 4200 ]



def getRoundTrainData(r):

    if r>0 :

        arr1 = np.arange( 0 , r*4200)

        arr2 = np.arange( (r+1)*4200 , 42000 )

        arr = np.append(arr1, arr2)

        return arr

    elif r==0 :

        arr = np.arange( (r+1)*4200 , 42000 )

        return arr
def getshuffletrain(n, arr, batch):

    need_ = arr[ n*batch : n*batch+batch]

    x1 = train_X[need_[0]:need_[0]+1]

    y1 = train_Y[need_[0]]

    

    f=0

    for i in need_:

        if f>0:

            x1 = x1.append(train_X[i:i+1])

            y2 = train_Y[i]

            y1 = np.vstack((y1, y2))

        f+=1

    return x1/255, y1
#读取数据

# read train data

train = pd.read_csv('../input/train.csv')

# read test data

test_X = pd.read_csv('../input/test.csv')

test_X.shape 
train_Y1 = train['label']

train_X = train.drop(labels=['label'], axis=1)

print("train_Y{}  train_X{}".format(train_Y1.shape, train_X.shape))
def one_hot(y):

    y_ohe = np.zeros(10)

    y_ohe[y] = 1

    return y_ohe

train_Y = np.array([ one_hot(train_Y1[i]) for i in range(len(train_Y1)) ])
#将训练数据分成十份，每次九份训练，一份测试，训练十回合

batch = 50

for r in range(10):

    print('第 %d 回合' % (r+1))

    x_test, y_test = getRoundTestData(r)

    arr = getRoundTrainData(r)

    

    for i in range(2):

        #打乱顺序

        np.random.shuffle(arr)

        #计算循环次数

        n_batch = int( len(arr)/batch )

        

        for n in range(n_batch):

            x_train, y_train = getshuffletrain(n, arr, batch)

            if n % 100==0:

                accuracy_rate = sess.run(accuracy, feed_dict={x:x_train, y:y_train, keep_prob:1.0})

                print("step %d, training accuracy %g" % (n, accuracy_rate))

            train_step.run(feed_dict={x:x_train, y:y_train, keep_prob:0.7})

        print("test accuracy %g" % sess.run(accuracy, feed_dict={x: x_test, y: y_test, keep_prob: 1.0}))
submission = pd.read_csv('../input/sample_submission.csv', index_col='ImageId')

l = sess.run(y_hat, feed_dict={x: test_X, keep_prob: 1.0})

submission.Label = sess.run(tf.argmax(l, 1))

len(submission)
submission.to_csv('submission.csv',index=True)