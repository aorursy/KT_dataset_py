# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
# data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
print(train.shape)
test = pd.read_csv("../input/test.csv")
print(test.shape)
batch_size = 128
test_size = 256
img_size = 28
X = train.iloc[:,1:].values.astype('float32') 
Y = train.iloc[:,0].values.astype('int32')
from keras.utils.np_utils import to_categorical
Y= to_categorical(Y)
num_classes = Y.shape[1]
print(num_classes)

X_train,X_val,Y_train,Y_val = train_test_split(X,Y,test_size=0.1,random_state=42)
X_test = test.values.astype('float32')

X_train = X_train.reshape(-1,img_size,img_size,1)
X_val = X_val.reshape(-1,img_size,img_size,1)
X_test = X_test.reshape(-1,img_size,img_size,1)
print(X_test.shape)

img = X_test[5000]
plt.imshow(img.reshape(28,28))

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))
def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    conv1 = tf.nn.conv2d(X, w,strides=[1, 1, 1, 1],padding='SAME')
    conv1_a = tf.nn.relu(conv1)
    conv1 = tf.nn.max_pool(conv1_a, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')
    conv1 = tf.nn.dropout(conv1, p_keep_conv)
    conv2 = tf.nn.conv2d(conv1, w2,strides=[1, 1, 1, 1],padding='SAME')
    conv2_a = tf.nn.relu(conv2)
    conv2 = tf.nn.max_pool(conv2_a, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')
    conv2 = tf.nn.dropout(conv2, p_keep_conv)
    conv3=tf.nn.conv2d(conv2, w3,strides=[1, 1, 1, 1],padding='SAME')
    conv3 = tf.nn.relu(conv3)
    FC_layer = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')
    FC_layer = tf.reshape(FC_layer, [-1, w4.get_shape().as_list()[0]])
    FC_layer = tf.nn.dropout(FC_layer, p_keep_conv)
    output_layer = tf.nn.relu(tf.matmul(FC_layer, w4))
    output_layer = tf.nn.dropout(output_layer, p_keep_hidden)
    result = tf.matmul(output_layer, w_o)
    return result
x = tf.placeholder("float",[None,img_size,img_size,1])
y = tf.placeholder("float",[None,num_classes])
w = init_weights([3,3,1,32])
w2= init_weights([3,3,32,64])
w3 = init_weights([3,3,64,128])
w4 = init_weights([128*4*4,625])
w_o = init_weights([625,num_classes])
p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

py_x = model(x,w,w2,w3,w4,w_o,p_keep_conv,p_keep_hidden)
Y_ = tf.nn.softmax_cross_entropy_with_logits(logits=py_x,labels=y)
cost=tf.reduce_mean(Y_)
optimizer = tf.train.RMSPropOptimizer(0.001,0.9).minimize(cost)
predict_op = tf.argmax(py_x,1)
prediction = []
    
    
    

def train():
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for i in range(100):
            training_batch = zip(range(0,len(X_train),batch_size),range(batch_size,len(X_train)+1,batch_size))
            for start, end in training_batch:
                sess.run(optimizer, feed_dict={x: X_train[start:end],y: Y_train[start:end],p_keep_conv: 0.8,p_keep_hidden: 0.5})
            #test_in = np.arange(len(X_test))
            #test_in=test_in[0:test_size]
        prediction.append(sess.run(predict_op,feed_dict={x:X_test,p_keep_conv: 1.0,p_keep_hidden: 1.0}))
train() 

pred = []
for item in prediction:
    for i in item:
        pred.append(i)
print(len(pred))
df_predictions = pd.DataFrame({'ImageId' : np.arange(len(pred))+1,
                               'Label' : pred})
df_predictions.head(10)
df_predictions.to_csv('digit_recog_model.csv',index=False)