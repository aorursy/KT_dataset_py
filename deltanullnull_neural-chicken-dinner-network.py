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
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_train.head()
#df_train['groupMembers'] = df_train.groupby('groupId')['groupId'].transform('count')
#df_test['groupMembers'] = df_train.groupby('groupId')['groupId'].transform('count')
#df_train['solo'] = df_train['numGroups'].apply(lambda x: 1 if x > 50 else 0)
#df_test['solo'] = df_train['numGroups'].apply(lambda x: 1 if x > 50 else 0)
df_train.head()
df_test.head()
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
df_train_sub = df_train.sample(10000)
sns.scatterplot('killPlace', 'winPlacePerc', data=df_train_sub)
f, ax = plt.subplots(figsize=(7,6))
sns.boxplot(x='killStreaks', y='winPlacePerc',data=df_train_sub, whis='range', palette='vlag')
sns.scatterplot('longestKill', 'winPlacePerc', data=df_train_sub)
#f, ax = plt.subplots(figsize=(7,6))
#sns.boxplot(x='solo', y='winPlacePerc',data=df_train_sub, whis='range', palette='vlag')
#f, ax = plt.subplots(figsize=(7,6))
#sns.boxplot(x='groupMembers', y='winPlacePerc',data=df_train_sub, whis='range', palette='vlag')
f, ax = plt.subplots(figsize=(7,6))
sns.boxplot(x='revives', y='winPlacePerc',data=df_train_sub, whis='range', palette='vlag')
f, ax = plt.subplots(figsize=(7,6))
sns.boxplot(x='boosts', y='winPlacePerc',data=df_train_sub, whis='range', palette='vlag')
f, ax = plt.subplots(figsize=(7,6))
sns.boxplot(x='weaponsAcquired', y='winPlacePerc',data=df_train_sub, whis='range', palette='vlag')
f, ax = plt.subplots(figsize=(7,6))
sns.boxplot(x='kills', y='winPlacePerc',data=df_train_sub, whis='range', palette='vlag')
f, ax = plt.subplots(figsize=(7,6))
sns.boxplot(x='DBNOs', y='winPlacePerc',data=df_train_sub, whis='range', palette='vlag')
import tensorflow as tf
from sklearn.model_selection import train_test_split
data_X = df_train.drop(['winPlacePerc', 'groupId', 'matchId', 'Id', 'teamKills'], axis=1)
data_y = df_train['winPlacePerc']
test_X = df_test.drop(['groupId', 'matchId', 'Id','teamKills'], axis=1)
x_train, x_eval, y_train, y_eval = train_test_split(data_X.values, data_y.values, test_size=0.3)
y_train = y_train.reshape(-1, 1)
y_eval = y_eval.reshape(-1, 1)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x_train)

x_train_scaled = scaler.transform(x_train)
x_eval_scaled = scaler.transform(x_eval)
x_test = test_X.values

x_test_scaled = scaler.transform(x_test)
print('Size of train set: %d' % x_train.shape[0])
print('Size of eval set: %d' % x_eval.shape[0])
n_features = x_train.shape[1]
n_data = x_train.shape[0]
n_data_eval = x_eval.shape[0]

print(n_features, n_data)
x = tf.placeholder(dtype=tf.float32, shape=[None, n_features])
y_true = tf.placeholder(dtype=tf.float32, shape=[None,1])
l0 = tf.layers.dense(inputs=x, units=32, activation=tf.nn.relu)
l1 = tf.layers.dense(inputs=l0, units=32, activation=tf.nn.relu)
#d0 = tf.layers.dropout(l1, rate=0.2)
l2 = tf.layers.dense(inputs=l1, units=32, activation=tf.nn.relu)
l3 = tf.layers.dense(inputs=l2, units=64, activation=tf.nn.relu)
#d1 = tf.layers.dropout(l3, rate=0.2)
l4 = tf.layers.dense(inputs=l3, units=32, activation=tf.nn.relu)
l5 = tf.layers.dense(inputs=l4, units=64, activation=tf.nn.relu)
#d2 = tf.layers.dropout(l5, rate=0.2)
l6 = tf.layers.dense(inputs=l5, units=64, activation=tf.nn.relu)
l7 = tf.layers.dense(inputs=l6, units=64, activation=tf.nn.relu)
l8 = tf.layers.dense(inputs=l7, units=32)
l9 = tf.layers.dense(inputs=l8, units=32, activation=tf.nn.relu)
#d2 = tf.layers.dropout(l9, rate=0.2)
l10 = tf.layers.dense(inputs=l9, units=32, activation=tf.nn.relu)
l11 = tf.layers.dense(inputs=l10, units=16, activation=tf.nn.relu)
y_pred = tf.layers.dense(inputs=l11, units=1, activation=tf.nn.sigmoid)

loss = tf.reduce_sum(tf.square(y_pred - y_true))
#loss = tf.nn.l2_loss(y_pred - y_true)

optimize = tf.train.AdamOptimizer().minimize(loss)
def build_graph():
    w_0 = tf.Variable(tf.random_normal(shape=[n_features, 20]))
    b_0 = tf.Variable(tf.constant(0.1, shape=[20]))

    a_0 = tf.nn.tanh(tf.matmul(x, w_0) + b_0)

    w_1 = tf.Variable(tf.random_normal(shape=[20, 20]))
    b_1 = tf.Variable(tf.constant(0.1, shape=[20]))

    a_1 = tf.nn.tanh(tf.matmul(a_0, w_1) + b_1)

    w_2 = tf.Variable(tf.random_normal(shape=[20, 20]))
    b_2 = tf.Variable(tf.constant(0.1, shape=[20]))

    a_2 = tf.nn.tanh(tf.matmul(a_1, w_2) + b_2)

    w_3 = tf.Variable(tf.random_normal(shape=[20, 1]))
    b_3 = tf.Variable(tf.constant(0.1, shape=[1]))

    y_pred = tf.sigmoid(tf.matmul(a_2, w_3) + b_3)

    loss = tf.reduce_sum((y_pred - y_true)**2)

    #correct = tf.cast(tf.equal(y_true, y_pred), dtype=tf.float32)
    #accuracy = tf.reduce_mean(correct)

    optimize = tf.train.AdagradOptimizer(learning_rate=0.001).minimize(loss)
batch_size = 40
n_epochs = 40000

batch_eval = 256

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    
    
    for e in range(n_epochs):
        
        # get batch
        batch_indices = np.random.randint(0, n_data, size=[batch_size])
    
        
        x_train_batch = x_train_scaled[batch_indices]
        y_train_batch = y_train[batch_indices]
        
        #print(y_train_batch)
        sess.run(optimize, feed_dict={x: x_train_batch, y_true: y_train_batch})
        
        if (e % 1000 == 0):
            batch_indices = np.random.randint(0, n_data, size=[batch_eval])
            x_train_batch = x_train_scaled[batch_indices]
            y_train_batch = y_train[batch_indices]
            loss_train = sess.run(loss, feed_dict={x: x_train_batch, y_true: y_train_batch})
            #print('Loss (Train): %.0f' % l)
            
            #y_p = sess.run(y_pred, feed_dict={x: x_train_batch})[0:5]
            #print('Pred',y_p)
            #print('True',y_train_batch[0:5])
            
            batch_indices_test = np.random.randint(0, n_data_eval, size=[batch_eval])
            x_eval_batch = x_eval_scaled[batch_indices_test]
            y_eval_batch = y_eval[batch_indices_test]
            loss_eval = sess.run(loss, feed_dict={x: x_eval_batch, y_true: y_eval_batch})
            print('Loss (Train): %.2f - (Eval): %.2f'% (loss_train, loss_eval))
            
    res = sess.run(y_pred, feed_dict={x: x_test_scaled})
df_result = pd.DataFrame(df_test['Id'].values, columns=['Id'])
df_result['winPlacePerc'] =  res
df_result
df_result.to_csv('result.csv',index=False)