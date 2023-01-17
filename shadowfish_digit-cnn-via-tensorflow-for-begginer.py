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
# Model is not tuned , only used to learn how to build a CNN model

# two conv-layers and three fc-layers



import tensorflow as tf

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical



BATCH_SIZE = 200

LEARNINT_RATE = 1e-4



# prepare data for training

data = pd.read_csv("../input/train.csv")

x = data.drop("label",axis=1)

y = data["label"]

X = np.reshape(np.asarray(x), newshape=[-1,28,28,1])

Y = to_categorical(y,num_classes=10)

del data,x,y
# input

x = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32,name="x")

y = tf.placeholder(shape=[None,10], dtype=tf.float32,name="y")

keep_prob = tf.placeholder(dtype=tf.float32,name="keep_prob")

# some useful func



def split_data(ratio, x=None, y=None):

    # return train_test_split(X,Y,test_size=ratio,random_state=40)

    if (x is not None) and (y is not None):

        return train_test_split(x, y, test_size=ratio)

    else:

        return train_test_split(X, Y, test_size=ratio)



def conv_filter(shape):

    return tf.Variable(tf.truncated_normal(shape=shape,stddev=0.1),dtype=tf.float32)

def bias(shape):

    init = tf.constant(value=0.1, shape=shape)

    return tf.Variable(init, dtype=tf.float32)
# model defined

w_conv1_1 = conv_filter([3,3,1,64])

conv1_1 = tf.nn.relu(tf.nn.conv2d(x,w_conv1_1,strides=[1,1,1,1],padding="SAME"))

w_conv1_2 = conv_filter([3,3,64,64])

conv1_2 = tf.nn.relu(tf.nn.conv2d(conv1_1,w_conv1_2,strides=[1,1,1,1],padding="SAME"))

w_conv1_3 = conv_filter([3,3,64,64])

conv1_3 = tf.nn.relu(tf.nn.conv2d(conv1_2,w_conv1_3,strides=[1,1,1,1],padding="SAME"))

pool1 = tf.nn.max_pool(conv1_3,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")



w_conv2_1 = conv_filter([3,3,64,128])

conv2_1 = tf.nn.relu(tf.nn.conv2d(pool1,w_conv2_1,strides=[1,1,1,1],padding="SAME"))

w_conv2_2 = conv_filter([3,3,128,128])

conv2_2 = tf.nn.relu(tf.nn.conv2d(conv2_1,w_conv2_2,strides=[1,1,1,1],padding="SAME"))

w_conv2_3 = conv_filter([3,3,128,128])

conv2_3 = tf.nn.relu(tf.nn.conv2d(conv2_2,w_conv2_3,strides=[1,1,1,1],padding="SAME"))

pool2 = tf.nn.max_pool(conv2_3,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")



flatten = tf.reshape(pool2,shape=[-1,7*7*128])

fc_weight = tf.Variable(tf.truncated_normal(shape=[7*7*128,4096],stddev=0.1),dtype=tf.float32)

fc_bias = bias([4096])

fc_layer1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(flatten, fc_weight),fc_bias))

drop1 = tf.nn.dropout(fc_layer1,keep_prob=keep_prob)



fc_weight2 = tf.Variable(tf.truncated_normal(shape=[4096, 4096],stddev=0.1),dtype=tf.float32)

fc_bias2 = bias([4096])

fc_layer2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(drop1, fc_weight2),fc_bias2))

drop2 = tf.nn.dropout(fc_layer2,keep_prob=keep_prob)



fc_weight3 = tf.Variable(tf.truncated_normal(shape=[4096, 10],stddev=0.1),dtype=tf.float32)

fc_bias3 = bias([10])

fc_layer3 = tf.nn.bias_add(tf.matmul(drop2, fc_weight3),fc_bias3)



logits = tf.nn.softmax(fc_layer3)

result = tf.argmax(logits, axis=1, name="result")



loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc_layer3, labels=y))

precision = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits,axis=1), tf.argmax(y,axis=1)),dtype=tf.float32))

optimizer = tf.train.AdamOptimizer(learning_rate=LEARNINT_RATE).minimize(loss)

# training



with tf.Session() as sess:

    tf.global_variables_initializer().run()

    saver = tf.train.Saver()

    

    # split data into training & testing

    train_x, test_x, train_y, test_y = split_data(ratio=0.2)



    for epoch in range(51):

        

        input = {

            x:test_x,

            y:test_y,

            keep_prob:1.0

        }

        prec, los = sess.run([precision,loss], feed_dict=input)

        print("epoch:", epoch, " test_precision:", prec," test_loss:", los)



        for batch in range(50):

            _, xx, _, yy = split_data(0.02,train_x,train_y)

            input = {

                x: xx,

                y: yy,

                keep_prob: 0.5

            }

            sess.run(optimizer, feed_dict=input)



#         if epoch % 50 == 0 and epoch!=0:

#             save_file = os.path.join("./","model")

#             if epoch==50:

#                 saver.save(sess,save_file,global_step=epoch, write_meta_graph=True)

#             else:

#                 saver.save(sess, save_file, global_step=epoch, write_meta_graph=False)

    

    # predict & save

    

    def get_data(n):

        data = pd.read_csv("../input/test.csv")

        data = np.reshape(np.asarray(data),newshape=[-1,28,28,1])

        length = data.shape[0]

        for i in range(0, length, n):

            yield data[i:min(i + n,length)]



    

    results = []

    datas = get_data(1000)

    for data in datas:

        input = {x:data,

                 keep_prob:1.0}

        tmp = sess.run(result, feed_dict=input)

        results.extend(tmp)

    results = pd.Series(results,name="Label").astype(int)

    results = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

    results.to_csv("results.csv",index=False)