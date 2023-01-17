# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input/segementedimages"))

# Any results you write to the current directory are saved as output.
import random
import numpy as np
import tensorflow as tf 
import cv2 
import os
import math as m
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd

def Convolution_NN_Model():
    training_iters = 100
    learning_rate = 0.00005
    n_input = 3
    n_classes = 12
    batch_size= 64
#     model_path = "D:\\BARQ\\PlantSeedling\\CNN Model"
#     print (model_path)
    
    
    collection="../input/segementedimages/segementedImages"
    
    ALL_Data_Array=[]
    for j,imagename in  enumerate(os.listdir(collection)):
        ALL_Data_Array = np.append(ALL_Data_Array,imagename)
    random.shuffle(ALL_Data_Array)
    
    Label_Array = []
    for i in range(len(ALL_Data_Array)):
        string = ALL_Data_Array[i]
        string = string.split('_')
        classs = (string[1].split('.'))[0]
        classs = int(classs)
        Label_Array = np.append (Label_Array , classs)
        
#     print("dataset value counts:")
#     print(pd.DataFrame(Label_Array)[0].value_counts())
    
    Hot_Encoded_Labels = np.zeros ([len(Label_Array),12])
    
    for i in range (len(Label_Array)):
        Hot_Encoded_Labels[i,int(Label_Array[i]-1)] = 1
        
    Images = []
    for image_path in glob(os.path.join(collection ,"*.png")):
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        Images.append(image_bgr)
    print (len(Images))
    
    Images = np.asarray(Images)
    print(Images.shape)
        
    def weight_variable(shape):
        initial = tf.random_normal(shape, stddev=0.001)
        return tf.Variable(initial)
 
    def bias_variable(shape):
        initial = tf.constant(0.001, shape=shape)
        return tf.Variable(initial)
 
    W_conv1 = weight_variable([3, 3, 3, 16])
    W_conv1 = tf.where(tf.is_nan(W_conv1), tf.ones_like(W_conv1), W_conv1);
    b_conv1 = bias_variable([16])
 
    W_conv2 = weight_variable([3, 3, 16, 32])
    W_conv2 = tf.where(tf.is_nan(W_conv2), tf.ones_like(W_conv2), W_conv2);
    b_conv2 = bias_variable([32])
    
    W_conv3 = weight_variable([3, 3, 32, 64])
    W_conv3 = tf.where(tf.is_nan(W_conv3), tf.ones_like(W_conv3), W_conv3);
    b_conv3 = bias_variable([64])
 
    W_convd1 = weight_variable([8 * 8 * 64, 128])
    W_convd1 = tf.where(tf.is_nan(W_convd1), tf.ones_like(W_convd1), W_convd1);
    b_convd1 = bias_variable([128])
 
    W_convout = weight_variable([128, 12])
    W_convout = tf.where(tf.is_nan(W_convout), tf.ones_like(W_convout), W_convout);
    b_convout = bias_variable([12])
 
    x = tf.placeholder("float", [None, 64, 64, 3])
    y = tf.placeholder("float", [None, 12])
 
    def conv2d(x, W, b, strides=1):
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)
 
    def maxpool2d(x, k=2):
        return tf.nn.max_pool(x, ksize=[1, k, 2, 1], strides=[1, k, 2, 1], padding='SAME')
 
    def conv_net(x):
        conv1 = conv2d(x, W_conv1, b_conv1)
        conv1 = maxpool2d(conv1, k=2)
 
        conv2 = conv2d(conv1, W_conv2, b_conv2)
        conv2 = maxpool2d(conv2, k=2)
        
        conv3 = conv2d(conv2, W_conv3, b_conv3)
        conv3 = maxpool2d(conv3, k=2)
 
        fc1 = tf.reshape(conv3, [-1, W_convd1.get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, W_convd1), b_convd1)
        fc1 = tf.nn.relu(fc1)
 
        out = tf.add(tf.matmul(fc1, W_convout), b_convout)
        return out
 
    pred = conv_net(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
 
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cost)
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
 
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
 
 
    No_Of_Training_Data = int(m.floor((len(Images)) * 0.9))
    No_Of_Dev_Data = m.ceil((len(Images)) * 0.1)
    NoOfTrials = m.ceil((len(Images)) / No_Of_Dev_Data)
 
    test_acc1 = 0
 
    with tf.Session() as sess:
        sess.run(init)
        train_loss = []
        test_loss = []
        train_accuracy = []
        test_accuracy = []
        summary_writer = sess.graph
#         summary_writer = tf.summary.FileWriter('./Output', sess.graph)
 
        trial=0
        for i in range(training_iters):
            start2 = No_Of_Dev_Data
            if trial == NoOfTrials:
                trial =0
            if trial == 0:
                start = 0
                end = No_Of_Training_Data
                train_X = Images[start:end]
                train_y = Hot_Encoded_Labels[start:end]
                test_X = Images[end: len(Images)]
                test_y = Hot_Encoded_Labels[end:int(len(Images))]
 
            elif trial == 1:
                start = No_Of_Dev_Data
                end = int(len(Images))
                train_X = Images[start:end]
                train_y = Hot_Encoded_Labels[start:end]
                test_X = Images[0: start]
                test_y = Hot_Encoded_Labels[0: start]
 
            else:
                start1 = 0
                end1 = start2
                start2 = end1 + No_Of_Dev_Data
                end2 = int(len(Images))
                train_X = np.concatenate((Images[start1:end1], Images[start2:end2]), axis=0)
                train_y = np.concatenate((Hot_Encoded_Labels[start1:end1], Hot_Encoded_Labels[start2:end2]), axis=0)
                test_X = Images[(start2 - end1): start2]
                test_y = Hot_Encoded_Labels[(start2 - end1): start2]
 
            train_X = train_X.reshape(-1, 64, 64, 3)
            test_X = test_X.reshape(-1, 64, 64, 3)
            
            print(pd.DataFrame(train_y))
            print("------------------------------------------------")
#             counter = 0
#             for batch in range(len(train_X) // batch_size):
#                 counter = counter + 1
#                 trial = trial+1
#                 batch_x = train_X[batch * batch_size:min((batch + 1) * batch_size, len(train_X))]
#                 batch_y = train_y[batch * batch_size:min((batch + 1) * batch_size, len(train_y))]
#                 _ = sess.run(train_op, feed_dict={x: batch_x,
#                                                   y: batch_y})
 
#                 loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
#                                                                   y: batch_y})
 
#                 test_acc, valid_loss = sess.run([accuracy, cost], feed_dict={x: test_X,
#                                                                              y: test_y})
    
#                 print("epoch", counter, 'completed out of', (len(train_X) // batch_size))
#                 print ('Training loss: ', loss , "Cross validation loss: " , valid_loss)
#                 print('Training accuracy: ', acc , "Cross validation accuray :  " ,test_acc)
#                 print()
                
#             if test_acc > test_acc1:
# #                 save_path = saver.save(sess, model_path + "subject" + str(f))
# #                 print("Model saved in file: %s" % save_path)
#                 test_acc1=test_acc
 
#     print("final test accuracy: ", test_acc1)
#         summary_writer.close()
    return
Convolution_NN_Model()
