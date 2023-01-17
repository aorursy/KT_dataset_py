# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import tensorflow as tf

import math

import random

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory





# Any results you write to the current directory are saved as output.
#Load data set

train_data = pd.read_csv("../input/train.csv",index_col='PassengerId')

test_data = pd.read_csv("../input/test.csv")

train_data
train_data.info()
train_data.describe()
import matplotlib.pyplot as plt



fig = plt.figure()

fig.set(alpha=0.2)



survived_0 = train_data.Pclass[train_data.Survived == 0].value.counts()

survived_1 = train_data.Pclass[train_data.Survived == 1].value_counts()









train_data.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)

train_data.replace(["female","male"],[0,1],inplace=True)

train_data.replace(["C","S","Q"],[0,1,2],inplace=True)

train_data['Deceased'] = train_data['Survived'].apply(lambda s: 1 - s)

train_data
x_data = train_data[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]]

x_data
train_data['Deceased'] = train_data['Survived'].apply(lambda s: 1 - s)

train_y = train_data[['Deceased', 'Survived']].as_matrix()

train_y
#Processing age, if age is null,calcul medium

train_x = x_data.values

class1 = []

class2 = []

class3 = []

def calcul_medium(data):

    data = sorted(data)

    size = len(data)

    if (size % 2) == 0:

        return (data[size//2] + data[size//2+1]) // 2

    if (size % 2) == 1:

        return data[size//2]



for i in range(len(train_x)):

    if train_x[i][0] == 1 and not math.isnan(train_x[i][2]):

        class1.append(train_x[i][2])

    if train_x[i][0] == 2 and not math.isnan(train_x[i][2]):

        class2.append(train_x[i][2])

    if train_x[i][0] == 3 and not math.isnan(train_x[i][2]):

        class3.append(train_x[i][2])



for i in range(len(train_x)):

    if train_x[i][0] == 1 and math.isnan(train_x[i][2]):

        train_x[i][2] = calcul_medium(class1)

    if train_x[i][0] == 2 and math.isnan(train_x[i][2]):

        train_x[i][2] = calcul_medium(class2)

    if train_x[i][0] == 3 and math.isnan(train_x[i][2]):

        train_x[i][2] = calcul_medium(class3)

train_x
for i in range(len(train_x)):

    if train_x[i][2] <= 5:

        train_x[i][2] = 0

    elif train_x[i][2]>5 and train_x[i][2]<=18:

        train_x[i][2] = 1

    elif train_x[i][2]>18 and train_x[i][2]<=25:

        train_x[i][2] = 2

    elif train_x[i][2]>25 and train_x[i][2]<=60:

        train_x[i][2] = 3

    else:

        train_x[i][2] = 4

train_x[2]
#Def logistic function





def logistic(x,y):

    train_x_set = train_x[:700]

    train_y_set = train_y[:700]

    validation_x_dataset = train_x[700:]

    validation_y_dataset = train_y[700:]

    batch_size = 50

    epochs = 100

    out_put = 2

    x = tf.placeholder(tf.float32,[None,7],name='train_data_set')

    y = tf.placeholder(tf.float32,[None,out_put],name='label')

    Weight = tf.Variable(tf.random_normal([7,out_put]))

    biases = tf.Variable(tf.zeros([out_put])+0.1)

    Wx_plus_b = tf.matmul(x,Weight) + biases

    prediction = tf.nn.softmax(Wx_plus_b)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))

    predict_step = tf.argmax(prediction, 1)

    optimize = tf.train.AdamOptimizer(0.01).minimize(loss)

    init = tf.global_variables_initializer()

    

    with tf.Session() as sess:

        sess.run(init)

        epoch = 1

        best = 0

        while epoch <= epochs:

            print("epoch of logistic training: ", epoch)



            for i in range(0, 700, batch_size):

                sess.run(optimize, feed_dict={x: train_x_set[i:i + batch_size], y: train_y_set[i:i + batch_size]})

                correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

                print("mini_batch", i, "~", i + batch_size, "of", epoch, "epochs")

                print("accuracy on train set: {}".format(sess.run(accuracy, feed_dict={x: train_x_set, y: train_y_set})))

                print("accuracy on validation set: {}, the current best accuracy: {}".format(

                    sess.run(accuracy, feed_dict={x: validation_x_dataset, y: validation_y_dataset}), best))

                # best = max(best, sess.run(accuracy, feed_dict={x: X, y_: Y}))

                if best < sess.run(accuracy, feed_dict={x: validation_x_dataset, y: validation_y_dataset}):

                    best = sess.run(accuracy, feed_dict={x: validation_x_dataset, y: validation_y_dataset})

                    # saver.save(sess, "./save.ckpt")

                    #savePredict(sess.run(predict_step, feed_dict={x: testData}), savepath=savepath)

            epoch += 1

        print("The best accuracy: ", best)

logistic(train_x,train_y)