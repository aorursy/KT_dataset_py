# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("./"))

# Any results you write to the current directory are saved as output.
import os
import matplotlib.image as mpimg
from tensorflow.layers import conv2d,batch_normalization, dense, flatten, max_pooling2d
from tensorflow.nn import softmax_cross_entropy_with_logits
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
from glob import glob 
import pandas as pd
import matplotlib.pyplot as plt
path = "../input/chess_board/Chess_Board/"

def readCord():
    coordinates = pd.read_csv("../input/chess_board/Chess_Board/Text.csv")
    dataY = coordinates[['wk_x', 'wk_z', 'bk_x', 'bk_z', 'wq_x', 'wq_z', 'bq_x', 'bq_z']]
    imgName = coordinates.iloc[:,0]
    return imgName, dataY.values
imgName, dataY = readCord()
imgName.shape, dataY.shape
# imgName = imgName[:50]
# dataY = dataY[:50]
def setBatch(dataY, imgName, start, batchSize):
    batchImg = []
    batchCoordinates = []
    for img in imgName[start : start+batchSize]:
        im = cv2.imread(path+"/"+img)
        im = cv2.resize(im,(224,224))
        batchImg.append(im)
    
    value = dataY[start:start+batchSize]

    batchCoordinates.append(value)
    batchCoordinates = np.array(batchCoordinates).reshape(batchSize,8)
    
    return batchImg, batchCoordinates
tf.reset_default_graph()
x = tf.placeholder('float32', [None,224,224,3])
y = tf.placeholder('float32', [None,8])
LRate = 0.001
batchSize = 20
epoch = 100
def model(x):
    layer = conv2d(x, filters=32, kernel_size=[3,3], strides=1, padding="SAME", kernel_initializer=tf.contrib.layers.xavier_initializer(),activation=tf.nn.relu)
    layer = conv2d(layer, filters=32, kernel_size=[3,3], strides=1, padding="SAME", kernel_initializer=tf.contrib.layers.xavier_initializer(),activation=tf.nn.relu)    
    layer = max_pooling2d(layer, pool_size=[2,2], strides=2, padding="SAME")
    
    layer = conv2d(layer, filters=64, kernel_size=[3,3], strides=1, padding="SAME", kernel_initializer=tf.contrib.layers.xavier_initializer(),activation=tf.nn.relu)
    layer = conv2d(layer, filters=64, kernel_size=[3,3], strides=2, padding="SAME", kernel_initializer=tf.contrib.layers.xavier_initializer(),activation=tf.nn.relu)
    layer = max_pooling2d(layer, pool_size=[2,2], strides=2, padding="SAME")
    
    layer = conv2d(layer, filters=128, kernel_size=[3,3], strides=2, padding="SAME", kernel_initializer=tf.contrib.layers.xavier_initializer(),activation=tf.nn.relu)
    layer = conv2d(layer, filters=128, kernel_size=[3,3], strides=1, padding="SAME", kernel_initializer=tf.contrib.layers.xavier_initializer(),activation=tf.nn.relu)
    layer = max_pooling2d(layer, pool_size=[2,2], strides=2, padding="SAME")
    
    layer = conv2d(layer, filters=256, kernel_size=[3,3], strides=1, padding="SAME", kernel_initializer=tf.contrib.layers.xavier_initializer(),activation=tf.nn.relu)
    layer = conv2d(layer, filters=256, kernel_size=[3,3], strides=1, padding="SAME", kernel_initializer=tf.contrib.layers.xavier_initializer(),activation=tf.nn.relu)
    layer = max_pooling2d(layer, pool_size=[2,2], strides=2, padding="SAME")
    
    layer = conv2d(layer, filters=512, kernel_size=[3,3], strides=1, padding="SAME", kernel_initializer=tf.contrib.layers.xavier_initializer(),activation=tf.nn.relu)
    layer = conv2d(layer, filters=512, kernel_size=[3,3], strides=1, padding="SAME", kernel_initializer=tf.contrib.layers.xavier_initializer(),activation=tf.nn.relu)
    layer = max_pooling2d(layer, pool_size=[2,2], strides=2, padding="SAME")
    print(layer)
    layer = flatten(layer)
    print(layer)
    layer = dense(layer, 1000, activation=tf.nn.relu)
    layer = dense(layer, 100, activation=tf.nn.relu)
    layer = dense(layer, 8)
    return layer
pred = model(x)
cost = tf.sqrt(tf.reduce_mean((pred - y)**2))
optimizer = tf.train.AdamOptimizer(LRate).minimize(cost)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver()
batchSize = 200
epoch = 51
sub = len(dataY)//batchSize
for e in range(epoch):
    avgBatchLoss=0
    start=0
    for i in range(sub):
        batchImg, batchCoordinates= setBatch(dataY, imgName, start, batchSize)  
        start += batchSize
        batchLoss = sess.run(cost, feed_dict={x : batchImg, y : batchCoordinates})    
        sess.run(optimizer, feed_dict={x : batchImg, y : batchCoordinates})  
        if i % 10:
            print("Epoch",e,"/",epoch,"    Sub Epoch",i+1,"/",sub,"   Batch Loss", batchLoss)
        avgBatchLoss+=batchLoss
    print("Avg Loss ",avgBatchLoss/sub)
    if e%25 == 0:
        save_path = saver.save(sess, "./model"+str(e)+".ckpt")
        print("Model saved in path: %s" % save_path)