import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

import seaborn as sns

import matplotlib.gridspec as gridspec

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
df = pd.read_csv('../input/creditcard.csv')
df.head()
df.isnull().sum()
#Select only the anonymized features.

v_features = df.ix[:,1:29].columns
plt.figure(figsize=(12,28*4))

gs = gridspec.GridSpec(28, 1)

for i, cn in enumerate(df[v_features]):

    ax = plt.subplot(gs[i])

    sns.distplot(df[cn][df.Class == 1], bins=50)

    sns.distplot(df[cn][df.Class == 0], bins=50)

    ax.set_xlabel('')

    ax.set_title('histogram of feature: ' + str(cn))

plt.show() 
#Drop all of the features that have very similar distributions between the two types of transactions.

df = df.drop(['V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8'], axis =1)
# create new fearures for distribution

df.loc[df.Class == 0, 'Normal'] = 1

df.loc[df.Class == 1, 'Normal'] = 0
#Rename 'Class' to 'Fraud'.

df = df.rename(columns={'Class': 'Fraud'})
#create Fraud and normal feature distribution

Fraud = df[df.Fraud == 1]

Normal = df[df.Normal == 1]
# create X_train by taking 80% of fraud transactions and 80% of normal transactions

X_train = Fraud.sample(frac=0.8)

count_Frauds = len(X_train)

X_train = pd.concat([X_train, Normal.sample(frac = 0.8)], axis = 0)

X_test = df.loc[~df.index.isin(X_train.index)]
# create Y_train by taking 80% of fraud transactions and 80% of normal transactions

y_train = X_train.Fraud

y_train = pd.concat([y_train, X_train.Normal], axis=1)

y_test = X_test.Fraud

y_test = pd.concat([y_test, X_test.Normal], axis=1)
# drop the guest features

X_train = X_train.drop(['Fraud','Normal'], axis = 1)

X_test = X_test.drop(['Fraud','Normal'], axis = 1)
# scale values of features

features = X_train.columns.values

for feature in features:

    mean, std = df[feature].mean(), df[feature].std()

    X_train.loc[:, feature] = (X_train[feature] - mean) / std

    X_test.loc[:, feature] = (X_test[feature] - mean) / std
#split the dataset for train,test & validation

split = int(len(y_test)/2)



inputX = X_train.as_matrix()

inputY = y_train.as_matrix()

inputX_valid = X_test.as_matrix()[:split]

inputY_valid = y_test.as_matrix()[:split]

inputX_test = X_test.as_matrix()[split:]

inputY_test = y_test.as_matrix()[split:]
#parameters

learning_rate = 0.005

training_epoch = 10

batch_size = 2048

display_step = 1
#tf graph input

x = tf.placeholder(tf.float32,[None,19])

y = tf.placeholder(tf.float32,[None,2])
#set model weights

w = tf.Variable(tf.zeros([19,2]))

b = tf.Variable(tf.zeros([2]))
#construct model using softmax activation

pred = tf.nn.softmax(tf.matmul(x,w) + b) 
#minimize error using cross entropy

cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred)))
#Gradient descent

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
#initializing variables

init = tf.global_variables_initializer()
#launch the graph

with tf.Session() as sess:

    sess.run(init)

    final_output_array = []

    #training cycle

    for epoch in range(training_epoch):

        total_batch = len(inputX)/batch_size

        avg_cost = 0

        #loop over all the batches

        for batch in range(int(total_batch)):

            batch_xs = inputX[(batch)*batch_size:(batch+1) *batch_size]

            batch_ys = inputY[(batch)*batch_size:(batch+1) *batch_size]



            # run optimizer and cost operation

            _,c= sess.run([optimizer,cost],feed_dict={x:batch_xs,y:batch_ys})

            avg_cost += c/total_batch



        correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))



        #disply log per epoch step

        if (epoch+1) % display_step == 0:

            train_accuracy, newCost = sess.run([accuracy, cost], feed_dict={x: inputX_test,y: inputY_test})

            print ("epoch:",epoch+1,"train_accuracy",train_accuracy,"cost",newCost,"valid_accuracy",sess.run([accuracy],feed_dict={x:inputX_valid,y:inputY_valid}))

            print ("")



    print ('optimization finished.')