# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline

import tensorflow as tf
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
train_data.shape , test_data.shape
train_data.head(2)
test_data.head(2)
def on_hot_encode(vec, vals=10):
    n = len(vec)
    out = np.zeros((n,vals))
    out[range(n),vec] = 1
    return out
train_labels = on_hot_encode(np.hstack([lb for lb in train_data['label']]))
col_name =  list(test_data.columns)
train_x = train_data[col_name][0:30000]
train_y = train_labels[0:30000]
test_x = train_data[col_name][30000:42000]
test_y = train_labels[30000:42000]
train_x.shape, test_x.shape, train_y.shape, test_y.shape
def init_weights(shape):
    w_value = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(w_value)    
def init_bias(shape):
    b_value = tf.constant(0.1,shape=shape)
    return tf.Variable(b_value)
# X is [ batchs, H, W, channels]
# W should be [filter H, filter W, channels in, channels out]
def conv2d(x,w):
    return tf.nn.conv2d(input=x,filter=w, strides=[1,1,1,1],padding='SAME')
# x is [ batchs, H, W, channels]
def max_pool_2by2(x):
    return tf.nn.max_pool(value=x,ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
# Convolutional layer
def convolutional_layer(input_x,shape):
    w = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x,w)+b)
# Create Fully Connected layer
def fully_connected_layer(input_layer,size):
    input_size = int(input_layer.get_shape()[1])
    w = init_weights([input_size,size])
    b = init_bias([size])
    return tf.matmul(input_layer,w)+b
# create placeholder for train_x
# And also here have to mention the columns no.
# None for length of batch
x = tf.placeholder(tf.float32,shape=[None,784])
y_true = tf.placeholder(tf.float32,shape=[None,10])
x_image = tf.reshape(x,[-1,28,28,1])
# 5x5 convolutional layer and 
# 1 means grayscale channels (which is channels_in)
# Extracted 32 features which is channels output
con1 = convolutional_layer(x_image,shape=[5,5,1,32])
con_pool1 = max_pool_2by2(con1)
# here also take same 5x5 convolutional layer 
# 32 channels input
# Extracted 64 features which is channels output
con2 = convolutional_layer(con_pool1,shape=[5,5,32,64])
con_pool2 = max_pool_2by2(con2)
# take output of con_pool2 and then make it sigle dimention
# mention 1024 Neuron in fully connected layer
con2_flat = tf.reshape(con_pool2,[-1,7*7*64])
full_layer_one = tf.nn.relu(fully_connected_layer(con2_flat,1024))
# Dropout 
hold_prob = tf.placeholder(tf.float32)
full_one_dropout = tf.nn.dropout(full_layer_one,keep_prob=hold_prob)
y_pred = fully_connected_layer(full_one_dropout,10)
predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=y_pred, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(y_pred, name="softmax_tensor")
  }
# Loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
train = optimizer.minimize(cross_entropy)
init = tf.global_variables_initializer()
batchsize = 150
batchlist = list(np.arange(0,30000,batchsize))
nepoch = np.arange(1,201)
len(nepoch)
# return list of boolean and getting value frm this 
# make this float
with tf.Session() as sess:
    sess.run(init)
    print('Starting......')
    for n in nepoch:   
        for pos,data in enumerate(batchlist):
            if pos+1 < len(batchlist):
                batch_x = train_x[data:batchlist[pos+1]]
                batch_y = train_y[data:batchlist[pos+1]]
                sess.run(train,feed_dict={x:batch_x,y_true:batch_y,hold_prob:0.5})
        if n != 0:
            matches = tf.equal(tf.argmax(y_pred,1), tf.argmax(y_true,1))
            accuracy = tf.reduce_mean(tf.cast(matches,tf.float32))
            accu = sess.run(accuracy,feed_dict={x:test_x,y_true:test_y,hold_prob:1.0})
            print('NO OF EPOCH :: {} and ACCURACY :: {}'.format(n,accu))
            if n==len(nepoch):
                print('\n')
                print('Starting Prediction for Test data....')
                predictedvalue = sess.run(predictions,feed_dict={x:test_data,hold_prob:1.0})
                print('done')
#list(predictedvalue.get('classes'))
#list(np.arange(1,28001,1))
mycnnsubmission = pd.DataFrame({'ImageId':list(np.arange(1,28001,1)),
                                'Label':list(predictedvalue.get('classes'))})
pd.crosstab(index=predictedvalue.get('classes'), columns='counts')
mycnnsubmission.to_csv('mycnnsubmission.csv', index=False)
