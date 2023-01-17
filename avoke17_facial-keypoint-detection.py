import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf
train_data = pd.read_csv('../input/facial-keypoint/training.csv')

test_data = pd.read_csv('../input/facial-keypoint/test.csv')

train_data.dropna(axis=0, how='any', inplace=True)

train_image = np.zeros([train_data.shape[0], 96*96])

test_image = np.zeros([test_data.shape[0], 96*96])

i = 0

for index, row in train_data.iterrows():

    train_image[i] = list(map(eval, row['Image'].split()))

    i = i + 1

    

i =0

for index, row in test_data.iterrows():

    test_image[i] = list(map(eval, row['Image'].split()))

    i = i + 1

    

train_image = train_image / 255

test_image = test_image / 255



y_train = np.zeros([train_image.shape[0], 30])



train_data = train_data.drop('Image', axis = 1)

FEATURES = list(train_data.columns)



i = 0

for index, row in train_data.iterrows():

    y_train[i] = np.array(row, dtype = 'float32')

    i = i + 1



#设定batchsize

batch_size = 100

n_batch = int(int(train_image.shape[0]) / batch_size)



x = tf.placeholder(tf.float32, [None, 96*96])

y = tf.placeholder(tf.float32, [None, 30])

is_train = tf.placeholder(tf.bool)

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



#池化，滑动步长在两个方向均为strides[1]=strides[2]=2，做2x2中的avgpool

def avg_pool_2x2(x):

    return tf.nn.avg_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')



X_train = tf.reshape(x, [-1, 96, 96, 1])

#第1层conv->relu

#96x96 -> 96x96 ，每个输入产生16个特征图

W_conv1 = weight_variable([3, 3, 1, 16])

b_conv1 = bias_variable([16])

res1 = conv2d(X_train, W_conv1) + b_conv1

bn1 = tf.layers.batch_normalization(res1, training=is_train)

h_conv1 = tf.nn.relu(res1)



#第3层conv->relu

#96x96 -> 96x96 ，每个输入产生32个特征图

W_conv3 = weight_variable([3, 3, 16, 32])

b_conv3 = bias_variable([32])

res3 = conv2d(h_conv1, W_conv3) + b_conv3

bn3 = tf.layers.batch_normalization(res3, training=is_train)

h_conv3 = tf.nn.relu(res3)





#第5层conv->relu

#96x96 -> 96x96 ，每个输入产生64个特征图

W_conv5 = weight_variable([3, 3, 32, 64])

b_conv5 = bias_variable([64])

res5 = conv2d(h_conv3, W_conv5) + b_conv5

bn5 = tf.layers.batch_normalization(res5, training=is_train)

h_conv5 = tf.nn.relu(res5)



#第6层conv->relu

#96x96 -> 48x48 ，每个输入产生64个特征图

W_conv6 = weight_variable([5, 5, 64, 64])

b_conv6 = bias_variable([64])

res6 = conv2ds2(h_conv5, W_conv6) + b_conv6

bn6 = tf.layers.batch_normalization(res6, training=is_train)

h_conv6 = tf.nn.relu(res6)



#第7-8层conv->relu->pool

#48x48 -> 48x48 ->24x24，每个输入产生64个特征图

W_conv7 = weight_variable([3, 3, 64, 64])

b_conv7 = bias_variable([64])

res7 = conv2d(h_conv6, W_conv7) + b_conv7

bn7 = tf.layers.batch_normalization(res7, training=is_train)

h_conv7 = tf.nn.relu(res7)

h_pool7 = avg_pool_2x2(h_conv6)





#第9层fc->relu+dropout，防止过拟合

#所有的特征reshape成一列，24x24x64 ->4096

h_pool7_flat = tf.reshape(h_pool7, [-1, 24*24*64])

w_fc1 = weight_variable([24*24*64, 4096])

b_fc1 = bias_variable([4096])

h_fc1 = tf.nn.relu(tf.matmul(h_pool7_flat, w_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)



#10层fc->mseloss

#4096 -> 30，生成prediction

w_fc2 = weight_variable([4096, 30])

b_fc2 = bias_variable([30])

y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2



#softmax->求交叉熵->平均求loss

mse = tf.reduce_sum(tf.square(y - y_conv))



#优化器优化loss，learning_rate = 0.1 

train_step_1 = tf.train.AdadeltaOptimizer(0.1).minimize(mse)



global_step = tf.Variable(0, name = 'global_step', trainable = False)



#初始化变量

init = tf.global_variables_initializer()

#训练结果保存在saver

saver = tf.train.Saver(var_list=tf.global_variables())



with tf.Session() as sess:

    #初始化

    sess.run(init)

    #迭代40次

    for epoch in range(40):

        print('epoch', epoch+1)

        #每批100个样本

        for batch in range(n_batch):

            batch_x = train_image[(batch) * batch_size : (batch + 1) * batch_size]

            batch_y = y_train[(batch) * batch_size : (batch + 1) * batch_size]

            sess.run(train_step_1, feed_dict = {x : batch_x, y : batch_y, keep_prob : 0.8, is_train : True})

            

        batch_x = train_image[n_batch * batch_size:]

        batch_y = y_train[n_batch * batch_size:]

        sess.run(train_step_1, feed_dict = {x : batch_x, y : batch_y, keep_prob : 0.8, is_train : True})

    #保存模型参数

    saver.save(sess, 'model.ckpt')

    saver.restore(sess, 'model.ckpt')

    filename = 'cnn_result.csv'

    #用测试集开始预测

    test_batch = test_image[:]

    myPrediction = sess.run(y_conv, feed_dict = {x : test_batch, keep_prob : 1.0, is_train : False})

    print(myPrediction)

    print(myPrediction.shape)

    print(np.max(myPrediction))

    plt.imshow(test_image[1500].reshape(96, 96), cmap='gray')

    plt.scatter(myPrediction[1500][0::2], myPrediction[1500][1::2], c='red', marker='x')

    locations = []

    rows = []

    lookup_df = pd.read_csv('../input/facial-keypoint/IdLookupTable.csv')

    for row_id, img_id, feature_name, loc in lookup_df.values:

        fi = FEATURES.index(feature_name)

        loc = myPrediction[img_id - 1][fi]

        locations.append(loc)

        rows.append(row_id)

    row_id_series = pd.Series(rows, name='RowId')

    loc_series = pd.Series(locations, name='Location')

    sub_csv = pd.concat([row_id_series, loc_series], axis=1)

    sub_csv.to_csv('face_key_detection_submission.csv',index = False)