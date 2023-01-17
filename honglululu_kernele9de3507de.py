import tensorflow as tf

import keras



print("tensorflow version ",tf.__version__)

print("keras version ",keras.__version__)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# import tensorflow as tf

import cv2

from PIL import Image

import os



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory





for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

print("done")
from pathlib import Path

x_train = []

y_train = []



height = 30

width = 30

channels = 3

classes = 43

n_inputs = height * width * channels



cwd = os.getcwd()

print("currect directory ",os.getcwd())

# for x in os.listdir('../'):

#     print(x)



# "/input/gtsrb-german-traffic-sign/train"

# for x in os.listdir("../input/gtsrb-german-traffic-sign/train/0"):

#     print(x)



print("read all the training data to x_train and y_train")

for i in range(classes) :

    path = "../input/gtsrb-german-traffic-sign/train/{0}/".format(i)

    print(path)

    Class=os.listdir(path)

    for a in Class:

        try:

            image=cv2.imread(path+a)

            image_from_array = Image.fromarray(image, 'RGB')

            size_image = image_from_array.resize((height, width))

            x_train.append(np.array(size_image))

            y_train.append(i)

        except AttributeError:

            print(" ")

            

x_train = np.array(x_train)

y_train = np.array(y_train)

x_train = x_train.astype('float32')/255 
# convert the image to grayscale and apply histogram equalization

print(x_train[0].shape)

plt.figure(figsize=(2,2))

plt.imshow(x_train[0])

# plt.imshow(cv2.cvtColor(x_train[0], cv2.COLOR_GRAY2RGB))



# def gray(img):

#     gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY )

#     equ  = cv2.equalizeHist(gray_img)

#     return equ



# x_train = np.array([gray(img) for img in x_train])

# x_train = x_train[...,np.newaxis]
# shuffle and split the data to training set and validation set

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

import sys

print(sys.version)

print("import lib successfully")



# x_train, y_train = shuffle(x_train, y_train)

print("total number of data",len(x_train))

x_train,x_validation,y_train,y_validation = train_test_split(x_train,y_train,test_size=0.2,random_state=42)

print("number of train data",len(x_train))

print("number of validation data",len(x_validation))



# print(x_train[10].shape)

# plt.figure(figsize=(2,2))

# plt.imshow(x_train[10])



# print(x_validation[10].shape)

# plt.figure(figsize=(2,2))

# plt.imshow(x_validation[10])
# build the conv net

from keras.layers import Dense, Input

from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten

def convNet(img):

    net = Conv2D(filters=6, kernel_size=(5,5), activation='relu')(img)

    net = Conv2D(filters=16, kernel_size=(5,5), activation='relu')(net)

    net = MaxPool2D(pool_size=(2, 2))(net)

#     net  = Dropout(rate=0.25)(net)

    net = Conv2D(filters=400, kernel_size=(3, 3), activation='relu')(net)

#     net = Dropout(rate=0.25)(net)

    net = Flatten()(net)

    net = Dropout(rate=0.5)(net)

    logits = Dense(43)(net)

    return logits

print("model defined!!!")
import tensorflow.compat.v1 as tf

from tensorflow.compat.v1.math import subtract

from keras import backend as K

tf.disable_v2_behavior()



print("0 is testing mode and 1 is the training mode")

print(K.learning_phase())



rate = 0.001

EPOCHS = 20

BATCH_SIZE = 128



x = tf.placeholder(tf.float32, (None, 30, 30, 3))

y = tf.placeholder(tf.int32, (None))

one_hot_y = tf.one_hot(y, classes)



# one = tf.constant(value = 1.0, dtype = tf.float32)

# drop_out_prob = subtract(one,keep_prob)



logits = convNet(x)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = one_hot_y)

total_loss = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate = rate)

training_operation = optimizer.minimize(total_loss)





correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))

accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()



def evaluate(X_data,y_data):

    num_examples = len(X_data)

    total_accuracy = 0

    sess = tf.get_default_session()

    for offset in range(0, num_examples, BATCH_SIZE):

        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]

        accuracy = sess.run(accuracy_operation, 

                            feed_dict={x: batch_x, y: batch_y, K.learning_phase():0})

        total_accuracy += (accuracy * len(batch_x))

    return total_accuracy / num_examples









print("done!")
# training start

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    num_examples = len(x_train)

    print("training....")

    print()

    for i in range(EPOCHS):

        x_train,y_train = shuffle(x_train,y_train)

        for offset in range(0,num_examples,BATCH_SIZE):

            batch_x,batch_y = x_train[offset:offset+BATCH_SIZE],y_train[offset:offset+BATCH_SIZE]

            sess.run(training_operation,feed_dict={x: batch_x, y: batch_y,K.learning_phase(): 1})

            

        validation_accuracy = evaluate(x_validation, y_validation)

        print("EPOCH {} ...".format(i+1))

        print("Validation Accuracy = {:.3f}".format(validation_accuracy))

        print()

        

    saver.save(sess, 'convnet')

    print("Model saved")