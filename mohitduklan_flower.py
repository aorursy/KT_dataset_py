# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

'''

    

'''

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import glob

glob.glob('../input/flowers-recognition/flowers/flowers/*')

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import numpy as np

import cv2

import glob

import tensorflow as tf

from sklearn.utils import shuffle

import pandas as pd

from sklearn.metrics import accuracy_score
def readImages():

    images = []

    folders = glob.glob('../input/flowers-recognition/flowers/flowers/*')

    data = []

    labels = []

    for folder in folders:

        img_path = glob.glob(folder+"/*")

        for img in img_path:

            if img.split('.')[-1] != "jpg":

                continue

            im = cv2.imread(img)

            b,g,r = cv2.split(im)

            rgb_img = cv2.merge([r,g,b]) 

            data.append(cv2.resize(rgb_img, (100, 100)))

            labels.append(img.split("/")[-2])

    print("Normalizing Images and reshaping them into 100*100")

    return data, labels
data, orig_labels = readImages()

data, orig_labels = shuffle(data, orig_labels)

labels = pd.get_dummies(orig_labels).values

data = np.array(data)/255.0

total_labels = [ 'daisy', 'dandelion', 'rose', 'sunflower','tulip']

print('Labels are:',total_labels)
plt.figure(figsize=(20,10))

columns = 5

for i, image in enumerate(data[:10]):

    plt.subplot(5 / columns + 1, columns, i + 1)

    plt.imshow(image)

    
train_X, test_X, train_y, test_y = train_test_split(data,labels, test_size = .20)

print("Train Size : X:",train_X.shape," Y:",train_y.shape," \nTest Size : X:",test_X.shape," Y:",test_y.shape)
tf.reset_default_graph()

training_epoch = 10

learning_rate = 0.003

input_size = 100

n_class = labels.shape[1]

x = tf.placeholder('float32', [None, input_size, input_size, 3])

y = tf.placeholder('float32', [None, n_class])

training = tf.placeholder('bool')

print("Training Epoch : ", training_epoch, "\nLearning Rate : ", learning_rate,"\nInput Size : ",input_size, "\nClasses : ",n_class)
def model(x,training=False):



    conv1 = tf.layers.conv2d(x, strides=1, filters=16, kernel_size=[3,3], padding='SAME', activation = tf.nn.relu, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

    print(conv1)

    conv1 = tf.layers.max_pooling2d(conv1, pool_size=[2,2], strides=2, padding='SAME')

    print(conv1)

    

    conv1 = tf.layers.conv2d(conv1, strides=1, filters=32, kernel_size=[3,3], padding='SAME', activation = tf.nn.relu, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

    print(conv1)

    conv1 = tf.layers.max_pooling2d(conv1, pool_size=[2,2], strides=2, padding='SAME')

    print(conv1)

    

    conv1 = tf.layers.conv2d(conv1, strides=2, filters=64, kernel_size=[3,3], padding='SAME', activation = tf.nn.relu, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

    print(conv1)

    conv1 = tf.layers.max_pooling2d(conv1, pool_size=[2,2], strides=2, padding='SAME')

    print(conv1)

    

    fc1 = tf.contrib.layers.flatten(conv1)

    print(fc1)

    fc1 = tf.layers.dropout(fc1,rate=0.2,seed=24,training=training)

    print(fc1)

    fc1 = tf.layers.dense(fc1, 100, activation=tf.nn.relu)

    print(fc1)

    out = tf.layers.dense(fc1, n_class)

    print(out)

    return out

    
pred_y = model(x,training)

get_out = tf.nn.softmax(pred_y)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred_y, labels = y))

optimize = tf.train.AdamOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)

batch_size = 10

train_loss_mean = []

for itr in range(training_epoch):

    for batch in range(len(data)//batch_size):

        batch_x = data[batch*batch_size:min((batch+1)*batch_size,len(data))]

        batch_y = labels[batch*batch_size:min((batch+1)*batch_size,len(data))]    

        _, train_loss = sess.run([optimize, cost], feed_dict={x: batch_x, y: batch_y, training:True})

        train_loss_mean.append(train_loss)

        #print(loss)

    train_pred = sess.run(pred_y, feed_dict={x:train_X,training:False})

    train_acc = accuracy_score(np.argmax(train_pred,1), np.argmax(train_y,1))

    

    test_pred, test_loss = sess.run([pred_y, cost], feed_dict={x: test_X, y: test_y, training:False})

    test_acc = accuracy_score(np.argmax(test_pred,1), np.argmax(test_y,1))

    

    print("\n\nEpoch :",itr+1,"\tTraining Cost :\t",np.mean(train_loss_mean),"\tTrain Accuracy :\t",train_acc)

    print("Epoch :",itr+1,"\tTest Cost :\t",test_loss,"\tTest Accuracy :\t\t",test_acc)
rand = np.random.randint(len(test_X))

plt.imshow(test_X[rand])

predict = sess.run(get_out, feed_dict={x:[test_X[rand]], training:False})

predict = predict.reshape(-1)

print('True Value :',total_labels[np.argmax(test_y[rand])])

print('\n\nPredicted Values :\n')

sorted_predict, sorted_labels = zip(*sorted(zip(predict, total_labels),reverse=True))

for i in range(len(predict)):

    print(sorted_labels[i],'{0:15.2f}'.format(float(sorted_predict[i])*100)+'%')



index = np.argmax(predict)

