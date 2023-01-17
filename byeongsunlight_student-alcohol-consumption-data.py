# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas

import numpy

import matplotlib.pyplot as plt

import seaborn

import datetime

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data_math = pandas.read_csv("../input/student-alcohol-consumption/student-mat.csv")

data_math
data_portuguese = pandas.read_csv("../input/student-alcohol-consumption/student-por.csv")

data_portuguese
data_math['class'] = 'math'

data_portuguese['class'] = 'portuguese'

data = data_math.append(data_portuguese)
data.T.iloc[:,1:8]
data['class'].value_counts()


seaborn.factorplot('class',kind='count',data=data)
data['sex'].value_counts()


seaborn.factorplot('sex',kind='count',data=data)
data['school'].value_counts()


seaborn.factorplot('school',kind='count',data=data)
data['age'].value_counts()
seaborn.factorplot('age',kind='count',data=data)
columns = data.columns

numeric = []

character = []

for i in columns:

	if data[i].dtype == "int64":

		numeric.append(i)

	else:

		character.append(i)

numeric
dummy_data = pandas.get_dummies(data[character])

dummy_data
concat_data = pandas.concat([data[numeric],dummy_data],axis=1)

concat_data
correlations = concat_data.corr()

correlations
def get_redundant_pairs(data):

	pairs_to_drop = set()

	cols = data.columns

	for i in range(0, concat_data.shape[1]):

		for j in range(0, i+1):

			pairs_to_drop.add((cols[i],cols[j]))

	return pairs_to_drop



def get_top_abs_correlations(data, n=30):

	abs_corr = data.corr().abs().unstack()

	labels_to_drop = get_redundant_pairs(data)

	abs_corr = abs_corr.drop(labels=labels_to_drop).sort_values(ascending=False)

	return abs_corr[14:n]

get_top_abs_correlations(concat_data,30)
pandas.DataFrame({"Walc":correlations['Walc'], 'Dalc':correlations['Dalc']}).T
walc = correlations['Walc']

dalc = correlations['Dalc']
threshold_walc = walc[abs(walc) > 0.15]

threshold_dalc = dalc[abs(dalc) > 0.15]
threshold_walc

threshold_dalc
walc_index = threshold_walc.index.tolist()

dalc_index = threshold_dalc.index.tolist()
walc_index
dalc_index
walc_data = concat_data[walc_index]

dalc_data = concat_data[dalc_index]
walc_data
dalc_data
walc_correlations = walc_data.corr()

dalc_correlations = dalc_data.corr()
walc_correlations
dalc_correlations
concat_data['Alcohol'] = concat_data['Walc'] + concat_data['Dalc']
concat_data['Alcohol']
concat_data['Alcohol'].value_counts()
machine_pre_data = concat_data.copy()

machine_pre_data = shuffle(machine_pre_data)

machine_pre_data = machine_pre_data.reset_index()

del machine_pre_data['index']
machine_pre_data
X_data = machine_pre_data.columns.difference(['Walc', 'Dalc','Alcohol'])
X_data
X1 = machine_pre_data[X_data]

Y1 = machine_pre_data['Alcohol']
X1
Y1
import tensorflow as tf 

xy = numpy.loadtxt("../input/train-school/train.csv",delimiter=',',dtype=numpy.float32)

X_train = xy[:,0:-1]

Y_train = xy[:,[-1]]



Xtrain, Xtest, Ytrain, Ytest = train_test_split(X_train,Y_train,test_size=0.3, random_state=0)







n_class = 9

X = tf.placeholder(tf.float32, [None,61])

Y = tf.placeholder(tf.int32, [None,1])

Y_one_hot = tf.one_hot(Y,n_class)

Y_one_hot = tf.reshape(Y_one_hot,[-1, n_class])



#W = tf.Variable(tf.random_normal([61,n_class]))

W = tf.get_variable("W",shape=[61,n_class], initializer=tf.contrib.layers.xavier_initializer())

b = tf.Variable(tf.random_normal([n_class]))

logits = (tf.matmul(X,W) + b)

hypothesis  = tf.nn.softmax(logits)



cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot))



optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)





prediction = tf.argmax(hypothesis,1)

correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



with tf.Session() as sess:

	sess.run(tf.global_variables_initializer())



	for step in range(20000):

		sess.run(optimizer, feed_dict={X:Xtrain, Y:Ytrain})

		if step % 100 ==0:

			loss, acc = sess.run([cost,accuracy], feed_dict={X:Xtrain, Y:Ytrain})
"step",step,"loss",loss,"acc",acc
##pred = sess.run(prediction,feed_dict={X:Xtest})



##for p, y in zip(pred,Y_train.flatten()):

  ##  (p==int(y), p, int(y))