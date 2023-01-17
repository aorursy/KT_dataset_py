# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
dataframe = pd.read_csv('../input/train.csv')



# Drop useless values

dataframe.drop('PassengerId', axis=1, inplace=True)

dataframe.drop('Ticket', axis=1, inplace=True)

dataframe.drop('Name', axis=1, inplace=True)

dataframe.drop('Cabin', axis=1, inplace=True)



# Replace Sexes with integers (male -> 0, female -> 1)

dataframe['Sex'] = dataframe['Sex'].map({'female': 1, 'male': 0})

# Replace Embarked with integers

dataframe['Embarked'] = dataframe['Embarked'].map({'S': 0, 'C': 1, 'Q':2})



# Fill Age NaN values with age mean

dataframe['Age'] = dataframe['Age'].fillna(dataframe['Age'].mean())



# Split into train and test dataframes (80% train)

df_train = dataframe.ix[:int(dataframe.shape[0]*0.9)]

df_test = dataframe.ix[int(dataframe.shape[0]*0.9)+1:]



print(df_train.head(10))
# Create Vectors

X, Y = df_train[['Pclass','Sex','Age','SibSp','Parch','Fare', 'Embarked']], df_train['Survived']

# X, Y = X.as_matrix(), Y.as_matrix()

X, Y = X.values, Y.values

X_test, Y_test = df_test[['Pclass','Sex','Age','SibSp','Parch','Fare', 'Embarked']], df_test['Survived']

# X_test, Y_test = X_test.as_matrix(), Y_test.as_matrix()

X_test, Y_test = X_test.values, Y_test.values

print(X.shape, Y.shape, X_test.shape, Y_test.shape)
# Hyperparameters

learning_rate = 0.001

batch_size = 25

no_epoch = 1

keep_prob = 1
# Variables

x = tf.placeholder(tf.float32, [None, 7])

labels = tf.placeholder(tf.float32, [None])

labels = tf.to_int64(labels)

weights1 = tf.Variable(tf.random_normal([7, 10]))

biases1 = tf.Variable(tf.random_normal([10]))

weights2 = tf.Variable(tf.random_normal([10, 10]))

biases2 = tf.Variable(tf.random_normal([10]))

weights3 = tf.Variable(tf.random_normal([10, 2]))

biases3 = tf.Variable(tf.random_normal([2]))
# Graph

hidden1 = tf.nn.relu(tf.matmul(x, weights1) + biases1)

hidden2 = tf.nn.relu(tf.matmul(hidden1, weights2) + biases2)

logits = tf.matmul(hidden2, weights3) + biases3



# Calculate cross-entropy (tensorflow takes care of one hot encoding here)

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(

    logits, labels, name='xentropy') 



# Calculate loss

loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')



# Optimizer

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)



correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels,0))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Session

with tf.Session() as sess:

    init = tf.global_variables_initializer()

    sess.run(init)

    

    for epoch in range(no_epoch):

        print('--- Epoch: ' + str(epoch) + '---')

        for i in range(int(len(X)/batch_size)):

            batch = [X[i*batch_size:(i+1)*batch_size], Y[i*batch_size:(i+1)*batch_size]]

            optimizer.run(feed_dict={x: batch[0], labels: batch[1]})



        print("test accuracy %g"%accuracy.eval(feed_dict={x: X_test, labels: Y_test}))