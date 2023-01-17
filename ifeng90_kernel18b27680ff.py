# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

# print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import tensorflow as tf
# 1 加载数据集，把对输入和结果进行分开

train = pd.read_csv("../input/train.csv")

images = train.iloc[:,1:].values

labels_flat = train.iloc[:,0].values.ravel()
# 2 对输入进行处理

images = images.astype(np.float)



images = np.multiply(images, 1.0 / 255.0)

print('输入数据的数量: (%g, %g)' % images.shape)



image_size = images.shape[1]

print ('输入数据的维度=> {0}'.format(image_size))



image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)

print ('图片的长 => {0}\n图片的高 => {1}'.format(image_width,image_height))



x = tf.placeholder('float', shape=[None, image_size])
# 3 对结果进行处理



labels_count = np.unique(labels_flat).shape[0]

print('结果的种类 => {0}'.format(labels_count))

# 进行One-hot编码

def dense_to_one_hot(labels_dense, num_classes):

    num_labels = labels_dense.shape[0]

    index_offset = np.arange(num_labels) * num_classes

    labels_one_hot = np.zeros((num_labels, num_classes))

    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot



labels = dense_to_one_hot(labels_flat, labels_count)

labels = labels.astype(np.uint8)



print('结果的数量：({0[0]},{0[1]})'.format(labels.shape))



y = tf.placeholder('float', shape=[None, labels_count])
# 4 把输入数据划分训练集和验证集

# 把40000个数据作为训练集，2000个数据作为验证集

VALIDATION_SIZE = 2000



validation_images = images[:VALIDATION_SIZE]

validation_labels = labels[:VALIDATION_SIZE]



train_images = images[VALIDATION_SIZE:]

train_labels = labels[VALIDATION_SIZE:]

# 5 对训练集进行分批

batch_size = 100

n_batch = int(len(train_images)/batch_size)
# 6 创建一个简单的神经网络用来对图片进行识别

weights = tf.Variable(tf.zeros([784,10]))

biases = tf.Variable(tf.zeros([10]))

result = tf.matmul(x,weights)+biases

prediction = tf.nn.softmax(result)
# 7 创建损失函数,以交叉熵的平均值为衡量

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits =prediction ))
# 8 用梯度下降法优化参数

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# 9 初始化变量

init = tf.global_variables_initializer()
# 10 计算准确度

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))





with tf.Session() as sess:

    #初始化

    sess.run(init)

    #循环50轮

    for epoch in range(50):

        for batch in range(n_batch):

            #按照分片取出数据

            batch_x = train_images[batch*batch_size:(batch+1)*batch_size]

            batch_y = train_labels[batch*batch_size:(batch+1)*batch_size]

            #进行训练

            sess.run(train_step,feed_dict = {x:batch_x,y:batch_y})

        #每一轮计算一次准确度

        accuracy_n = sess.run(accuracy,feed_dict={ x: validation_images, y: validation_labels})        

        print ("第" + str(epoch+1)+"轮，准确度为："+str(accuracy_n))