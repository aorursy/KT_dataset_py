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
import pandas as pd

import numpy as np

import tensorflow as tf

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df=pd.read_csv('../input/data.csv')
df.head()
df.drop('id',1,inplace=True)
df.drop('Unnamed: 32',1,inplace=True)
df.head()
df.describe()
df.isnull().sum()
plt.figure(figsize=(20,10))

sns.heatmap(df.corr(),annot=True)
from sklearn.model_selection import train_test_split

df_train,df_test = train_test_split(df, test_size=0.2, random_state=0,shuffle=True)
y_train=df_train['diagnosis']

y_test=df_test['diagnosis']
y_train.head(10)
y_train=pd.get_dummies(y_train)
y_train.head(10)
y_test.head(10)
y_test=pd.get_dummies(y_test)
y_test.head(10)
x_train=df_train.drop('diagnosis',1)

x_test=df_test.drop('diagnosis',1)
x_train.head()
y_train.shape
y_test.shape
x_train.shape
nodes_hl1 = 25

nodes_hl2 = 10





n_classes = 2

batch_size = 91
x = tf.placeholder('float',[None,30])

y = tf.placeholder('float')

x_train=x_train.values
y_train=y_train.values

y_test=y_test.values

x_test=x_test.values
x_train.shape
def neural_network_model(data):

    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([30, nodes_hl1])),

                      'biases':tf.Variable(tf.random_normal([nodes_hl1]))}



    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([nodes_hl1, nodes_hl2])),

                      'biases':tf.Variable(tf.random_normal([nodes_hl2]))}



    output_layer = {'weights':tf.Variable(tf.random_normal([nodes_hl2, n_classes])),

                      'biases':tf.Variable(tf.random_normal([n_classes]))}





    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])

    l1 = tf.nn.relu(l1)



    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])

    l2 = tf.nn.relu(l2)



    output = tf.add(tf.matmul(l2,output_layer['weights']), output_layer['biases'])

    

    



    return output
def train_neural_network(x):

    prediction = neural_network_model(x)

    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 150

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):

            epoch_loss = 0

            i=0

            while i <= (len(x_train)-batch_size):

                start=i

                end=i+batch_size

                batch_x=np.array(x_train[start:end])

                batch_y=np.array(y_train[start:end])

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})

                epoch_loss += c

                i=i+batch_size

            print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)    

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            print('Accuracy:',accuracy.eval({x:x_test, y:y_test}))
train_neural_network(x)