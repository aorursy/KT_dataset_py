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
import csv as csv

import numpy as np

import tensorflow as tf



def preprocess(filename):

    data_df = pd.read_csv(filename, header=0)

    data_df['Gender'] = data_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    # All missing Embarked -> just make them embark from most common place

    if len(data_df.Embarked[ data_df.Embarked.isnull() ]) > 0:

        data_df.Embarked[ data_df.Embarked.isnull() ] = data_df.Embarked.dropna().mode().values



    Ports = list(enumerate(np.unique(data_df['Embarked'])))    # determine all values of Embarked,

    Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index

    data_df.Embarked = data_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int



    # All the ages with no data -> make the median of all Ages

    median_age = data_df['Age'].dropna().median()

    if len(data_df.Age[ data_df.Age.isnull() ]) > 0:

        data_df.loc[ (data_df.Age.isnull()), 'Age'] = median_age



    # Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)

    data_df = data_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

    return data_df
# transform data to np.array

train_df = preprocess('../input/train.csv')

test_df = preprocess('../input/test.csv')



train_X = np.array(train_df.drop(['Survived'],axis=1))

train_X = train_X.T

train_Y = np.matrix(train_df['Survived'])

#train_Y = train_Y.T

#train_Y.reshape((train_Y.shape[0],1))

test_X = np.array(test_df)

test_X = test_X.T

print(train_X.shape)

print(train_Y.shape)

print(test_X.shape)
# build a neron network to be a classifier

X = tf.placeholder(tf.float32, shape=[7,None])

Y = tf.placeholder(tf.float32, shape=[1,None])



W1 = tf.Variable(tf.random_normal([6,7]))

b1 = tf.Variable(tf.random_normal([6,1]))

W2 = tf.Variable(tf.random_normal([5,6]))

b2 = tf.Variable(tf.random_normal([5,1]))

W3 = tf.Variable(tf.random_normal([1,5]))

b3 = tf.Variable(tf.random_normal([1,1]))



Z1 = tf.add(tf.matmul(W1,X),b1)

A1 = tf.nn.relu(Z1)

Z2 = tf.add(tf.matmul(W2,A1),b2)

A2 = tf.nn.relu(Z2)

Z3 = tf.add(tf.matmul(W3,A2),b3)

hypothesis = tf.sigmoid(Z3)

#hypothesis = hypothesis > 0.5

logits = tf.transpose(hypothesis, perm=[1, 0])

labels = tf.transpose(Y, perm=[1, 0])

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels = labels))

optimizer = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(cost)



predict = tf.cast(hypothesis>0.5,dtype = tf.float32)

training_accuracy = tf.reduce_mean(tf.cast(tf.equal(predict,Y),dtype=tf.float32))

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for step in range(50001):

        cost_val,_ = sess.run([cost,optimizer],feed_dict={X:train_X,Y:train_Y})

        if step %1000 ==0:

            print(step,cost_val)

    h,c,a = sess.run([hypothesis,predict,training_accuracy],feed_dict = {X:train_X,Y:train_Y})

    print('training accuracy:',a)

    test_predict = sess.run(predict,feed_dict={X:test_X})

    print(test_predict)



writer = csv.writer(open('predict.csv','wb'))

writer.writerow(['passageId','Survived'])

for i in range(test_predict.shape[1]):

    writer.writerow([i,test_predict[0][i]])