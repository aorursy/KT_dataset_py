import numpy as np

import pandas as pd

import tensorflow as tf

import os



#读取数据，提取训练集和测试集

train  = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')

x_train = (train.iloc[:,1:].values).astype(np.float)

x_test = (test.values).astype(np.float)

#线性归一化，加快收敛速度，利于梯度下降

x_train = np.multiply(x_train, 1.0/255)

x_test = np.multiply(x_test, 1.0/255)

#相关变量设置

image_size = x_train.shape[1]

image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)

label_train = train.iloc[:,0].values

label_count = np.unique(label_train).shape[0]



#独热编码，用于神经网络FC层后输出分类，例如数字4=[0,0,0,1,0,0,0,0,0,0]

def one_hot(label_dense, count):

    num_label = label_dense.shape[0]

    index_offset = np.arange(num_label) * count

    label_one_hot = np.zeros((num_label, count))

    label_one_hot.flat[index_offset + label_dense.ravel()] = 1

    return label_one_hot



labels = one_hot(label_train, label_count)

labels = labels.astype(np.uint8)



#设定batchsize

batch_size = 100

n_batch = int(x_train.shape[0] / batch_size)

x = tf.placeholder(tf.float32, [None, 784])

y = tf.placeholder(tf.float32, [None, 10])



#权重矩阵初始化，正态分布，mean=0， stddev=0.1

def weight_variable(shape):

    initial = tf.truncated_normal(shape, mean = 0, stddev = 0.1)

    return tf.Variable(initial)



#偏置矩阵初始化，常值0.05

def bias_variable(shape):

    initial = tf.constant(0.05, shape = shape)

    return tf.Variable(initial)



#卷积，滑动步长在两个方向均为strides[1]=strides[2]=1

def conv2d(x, w):

    return tf.nn.conv2d(x, w, strides = [1, 1, 1, 1], padding = 'SAME')



#卷积，滑动步长在两个方向均为strides[1]=strides[2]=2

def conv2ds2(x, w):

    return tf.nn.conv2d(x, w, strides = [1, 2, 2, 1], padding = 'SAME')



#池化，滑动步长在两个方向均为strides[1]=strides[2]=2，做2x2中的maxpool

def max_pool_2x2(x):

    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')



#x原本为长度784的行向量，reshape为具有宽高的矩阵

x_image = tf.reshape(x, [-1, 28, 28, 1])



#第1层conv->relu

#28x28 -> 28x28 每个输入产生32个特征图

W_conv1 = weight_variable([3, 3, 1, 32])

b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)



#第2层conv->relu

#28x28 -> 28x28 每个输入产生32个特征图

W_conv2 = weight_variable([3, 3, 32, 32])

b_conv2 = bias_variable([32])

h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)



#第3层conv->relu

#28x28->  14x14 ，每个输入产生64个特征图

W_conv3 = weight_variable([5, 5, 32, 32])

b_conv3 = bias_variable([32])

h_conv3 = tf.nn.relu(conv2ds2(h_conv2, W_conv3) + b_conv3)



#第4层conv->relu

#14x14 -> 14x14 , 每个输入产生64个特征图

W_conv4 = weight_variable([3, 3, 32, 64])

b_conv4 = bias_variable([64])

h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)



#第5层conv->relu

#14x14 -> 14x14 每个输入产生64个特征图

W_conv5 = weight_variable([3, 3, 64, 64])

b_conv5 = bias_variable([64])

h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5) + b_conv5)



#第6层conv->relu

#14x14 ->  7x7 ，每个输入产生64个特征图

W_conv6 = weight_variable([5, 5, 64, 64])

b_conv6 = bias_variable([64])

h_conv6 = tf.nn.relu(conv2ds2(h_conv5, W_conv6) + b_conv6)



#第5层fc->relu+dropout，防止过拟合

#所有的特征reshape成一列，7x7x64 -> 128

h_conv6_flat = tf.reshape(h_conv6, [-1, 7*7*64])

w_fc1 = weight_variable([7*7*64, 1024])

b_fc1 = bias_variable([1024])

h_fc1 = tf.nn.relu(tf.matmul(h_conv6_flat, w_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)



#第6层fc->softmax

#128 -> 10，生成prediction

w_fc2 = weight_variable([1024, 10])

b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2

prediction = tf.nn.softmax(y_conv)



#softmax->求交叉熵->平均求loss

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = y_conv))

#优化器优化loss，learning_rate = 0.1 

train_step_1 = tf.train.AdadeltaOptimizer(0.1).minimize(loss)

#argmax返回每行最大值索引，即每个图片的label，equal逐个元素判断是否相同

correction_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_conv, 1))

#boolean转化为float，求准确率

accuracy = tf.reduce_mean(tf.cast(correction_prediction, tf.float32))



global_step = tf.Variable(0, name = 'global_step', trainable = False)



#初始化变量

init = tf.global_variables_initializer()

#训练结果保存在saver

saver = tf.train.Saver()



with tf.Session() as sess:

    #初始化

    sess.run(init)

    #迭代20次

    for epoch in range(30):

        print('epoch', epoch+1)

        #每批100个样本

        for batch in range(n_batch):

            batch_x = x_train[(batch) * batch_size : (batch + 1) * batch_size]

            batch_y = labels[(batch) * batch_size : (batch + 1) * batch_size]

            sess.run(train_step_1, feed_dict = {x : batch_x, y : batch_y, keep_prob : 0.6})

        batch_x = x_train[n_batch * batch_size:]

        batch_y = labels[n_batch * batch_size:]

        sess.run(train_step_1, feed_dict = {x : batch_x, y : batch_y, keep_prob : 0.6})

    #保存模型参数

    saver.save(sess, 'model.ckpt')

    saver.restore(sess, 'model.ckpt')

    filename = 'cnn_result.csv'

    #用测试集开始预测

    test_batch = x_test[:]

    myPrediction = sess.run(prediction, feed_dict = {x : test_batch, keep_prob : 1.0})

    label_test = np.argmax(myPrediction, axis = 1)

    result = pd.Series(label_test, name='Label')

    submission = pd.concat([pd.Series(range(1, 28001), name = 'ImageId'), result], axis = 1)

    submission.to_csv(filename, index = False)