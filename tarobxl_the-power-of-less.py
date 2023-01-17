# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import kagglegym

import numpy as np

import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt

from sklearn import preprocessing as pp



env = kagglegym.make()

o = env.reset()
col = ['technical_20']

train = o.train[col + ['id', 'timestamp', 'y']].copy(deep=True)



im = pp.Imputer(strategy='median')

train[col] = im.fit_transform(train[col])

sX = pp.StandardScaler()

train[col] = sX.fit_transform(train[col])

train['b'] = 1



y_min = train.y.min()

y_max = train.y.max()



df_id = train[['id', 'timestamp']].groupby('id').agg([np.min])

df_id.reset_index(level=0, inplace=True)

train = pd.merge(train, df_id, on='id', how='inner')

train = train.rename(columns={train.columns[len(train.columns)-1]: 'min_ts'})

train = train.loc[(train.min_ts > 1) & (train.y<y_max) & (train.y>y_min)].copy(deep=True)





features = ['b']+col

n = len(features)



learning_rate = 0.01

training_epochs = 1000

cost_history = np.empty(shape=[1],dtype=float)



X = tf.placeholder(tf.float32,[None,n])

Y = tf.placeholder(tf.float32,[None,1])

W = tf.Variable(tf.zeros([n,1]))



init = tf.global_variables_initializer()



y_ = tf.matmul(X, W)



cost = tf.add(tf.reduce_mean(tf.square(y_ - Y)), tf.reduce_mean(tf.square(W)))

training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)



sess = tf.Session()

sess.run(init)



for epoch in range(training_epochs):

    sess.run(training_step,feed_dict={X: train[features], Y: train[['y']].values})
while True:

    o.features[col] = im.transform(o.features[col])

    o.features[col] = sX.transform(o.features[col])

    o.features['b'] = 1

    

    o.target.y = sess.run(y_, feed_dict={X:o.features[features]})

    o.target.y = np.clip(o.target.y, y_min, y_max)

    

    o, reward, done, info = env.step(o.target)

    if done:

        print(info)

        break

    if o.features.timestamp[0] % 100 == 0:

        print(reward)