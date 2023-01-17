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
import numpy as np
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
#from __future__ import division 
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
mydata = np.genfromtxt("../input/dataaaaaaaaaa/data.csv", delimiter = ',' )

X = mydata [1: , 0:95]
Y = mydata [1: , 95:]

train_X = X[0:1200 , :]
test_X = X[1200: , :]
train_Y = Y[0:1200 , :]
test_Y = Y[1200: , :]
class_1 = test_Y[: , 0:1]
class_2 = test_Y[: , 1:2]
class_3 = test_Y[: , 2:]

learning_rate = 0.08
training_epochs = 200
cost_history = np.empty(shape=[1] , dtype = float)
n_dim = X.shape[1]
n_class = 3
model_path = "/home/itachi/Desktop/project"

n_hidden_1 = 20
n_hidden_2 = 20
n_hidden_3 = 20
n_hidden_4 = 20

x = tf.placeholder(tf.float32 , [None , n_dim])
w = tf.Variable(tf.zeros([n_dim , n_class]))
b = tf.Variable(tf.zeros([n_class]))
y_ = tf.placeholder(tf.float32 , [None , n_class])

def multilayer_perceptron(x, weights , biases):

	layer_1 = tf.add(tf.matmul(x , weights['h1']) , biases['b1'])
	layer_1 = tf.nn.tanh(layer_1)


	layer_2 = tf.add(tf.matmul(layer_1 , weights['h2']) , biases['b2'])
	layer_2 = tf.nn.tanh(layer_2)

	layer_3 = tf.add(tf.matmul(layer_2 , weights['h3']) , biases['b3'])
	layer_3 = tf.nn.tanh(layer_3)

	layer_4 = tf.add(tf.matmul(layer_3 , weights['h4']) , biases['b4'])
	layer_4 = tf.nn.tanh(layer_4)

	out_layer = tf.matmul(layer_4 , weights['out']) + biases['out']
	out_layer = tf.nn.relu(out_layer)

	return out_layer


weights = {
	'h1' : tf.Variable(tf.truncated_normal([n_dim , n_hidden_1])),
	'h2' : tf.Variable(tf.truncated_normal([n_hidden_1 , n_hidden_2])),
	'h3' : tf.Variable(tf.truncated_normal([n_hidden_2 , n_hidden_3])),
	'h4' : tf.Variable(tf.truncated_normal([n_hidden_3 , n_hidden_4])),
	'out' : tf.Variable(tf.truncated_normal([n_hidden_4 , n_class]))
}
biases = {
	'b1' : tf.Variable(tf.truncated_normal([n_hidden_1])),
	'b2' : tf.Variable(tf.truncated_normal([n_hidden_2])),
	'b3' : tf.Variable(tf.truncated_normal([n_hidden_3])),
	'b4' : tf.Variable(tf.truncated_normal([n_hidden_4])),
	'out' : tf.Variable(tf.truncated_normal([n_class]))
}

init = tf.global_variables_initializer()

saver = tf.train.Saver()


y = multilayer_perceptron(x , weights , biases)

cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y , labels = y_))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)


sess = tf.Session()
sess.run(init)

mse_history = []
accuracy_history = []

for epoch in range(training_epochs):
	sess.run(training_step , feed_dict= {x: train_X , y_: train_Y})
	cost = sess.run(cost_function , feed_dict = {x: train_X , y_: train_Y})
	cost_history = np.append(cost_history , cost)
	correct_prediction = tf.equal(tf.argmax(y , 1), tf.argmax(y_ , 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction , tf.float32))

	pred_y = sess.run(y , feed_dict = {x: test_X})
	mse = tf.reduce_mean(tf.square(pred_y - test_Y))
	mse_ = sess.run(mse)
	mse_history.append(mse_)
	accuracy = (sess.run(accuracy , feed_dict = {x: train_X , y_: train_Y}))
	accuracy_history.append(accuracy)

	print('epoch : ', epoch, '-', 'cost: ',cost,"-MSE: ",mse_, "-Train Accuracy: ",accuracy)


save_path = saver.save(sess , model_path)
print("Model saved in file: %s" %save_path)

plt.plot(mse_history, 'r')
plt.ylabel('Error rate')
plt.xlabel('Number of iterations')
plt.show()
plt.plot(accuracy_history)
plt.ylabel('Accuracy')
plt.xlabel('Number of iterations')
plt.show()

correct_prediction = tf.equal(tf.argmax(y , 1) , tf.argmax(y_ , 1))
#print y
accuracy = tf.reduce_mean(tf.cast(correct_prediction , tf.float32))
print("Test Accuracy: ", (sess.run(accuracy , feed_dict = {x: test_X , y_: test_Y})))


pred_y = sess.run(y , feed_dict = {x: test_X})
temp = pred_y[: , :]
temp = np.where(temp<0.5 , 0 , 1)
class_1_pred = temp[: , 0:1]
class_2_pred = temp[: , 1:2]
class_3_pred = temp[: , 2:]

"""#print 'Accuracy:', accuracy_score(class_1, class_1_pred)
print ('F1 score:', f1_score(class_1, class_1_pred))
print ('Recall:', recall_score(class_1, class_1_pred))
print ('Precision:', precision_score(class_1, class_1_pred))
print( '\n clasification report:\n', classification_report(class_1, class_1_pred))
print ('\n confussion matrix:\n',confusion_matrix(class_1, class_1_pred))
print ('F1 score:', f1_score(class_2, class_2_pred))


mse = tf.reduce_mean(tf.square(pred_y - test_Y))
print("MSE: %4" % sess.run(mse))

"""

