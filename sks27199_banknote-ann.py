# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../"))



# Any results you write to the current directory are saved as output.
import tensorflow as tf
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split
def read_dataset():

    df = pd.read_csv("../input/BankNote.csv")

    print(len(df.columns))

    X = df[df.columns[0:4]].values

    y = df[df.columns[4]]

    #Encode the dependent variable

    Y=pd.get_dummies(y)



    print(X.shape)

    return(X,Y)
def one_hot_encode(labels):

    n_labels = len(labels)

    n_unique_labels = len(np.unique(labels))

    one_hot_encode = np.zeros((n_labels, n_unique_labels))

    one_hot_encode[np.arange(n_labels), labels] = 1

    return one_hot_encode
#Read the dataset

X, Y = read_dataset()
#Shuffle the dataset to mix up the rows

X, Y = shuffle(X, Y, random_state =1)
# Convert the dataset into train and test part

train_x, test_x, train_y, test_y = train_test_split(X,Y,test_size=0.20, random_state=415)
#Inspect the shape of the training and testing

print(train_x.shape)

print(train_y.shape)

print(test_x.shape)

print(test_y.shape)

# ses = tf.Session()

# ini = tf.global_variables_initializer()

print(train_x)
# Define the important parameters and variables to work with the tensors

learning_rate = 0.01

training_epochs = 700

cost_history = np.empty(shape=[1],dtype = float)

n_dim = X.shape[1]

print("n_dim",n_dim)

n_class = 2

model_path = r"../"
#Define the number of the hidden layers and number of neurons for each layer

n_hidden_1 = 10

n_hidden_2 = 10

n_hidden_3 = 10

n_hidden_4 = 10



x = tf.placeholder(tf.float32, [None,n_dim])

w = tf.Variable(tf.zeros([n_dim, n_class]))

b = tf.Variable(tf.zeros([n_class]))

y_ = tf.placeholder(tf.float32, [None, n_class])

#Define the model

def multilayer_perceptron(x, weights, biases):

	# Hidden layer with RELU activation



	layer_1 = tf.add(tf.matmul(x, weights['h1']),biases['b1'])

	layer_1 = tf.nn.relu(layer_1)



	#Hidden layer with relu activation

	layer_2 = tf.add(tf.matmul(layer_1, weights['h2']),biases['b2'])

	layer_2 = tf.nn.relu(layer_2)



	#Hidden layer with relu activation

	layer_3 = tf.add(tf.matmul(layer_2, weights['h3']),biases['b3'])

	layer_3 = tf.nn.relu(layer_3)



	#Hidden layer with relu activation

	layer_4 = tf.add(tf.matmul(layer_3, weights['h4']),biases['b4'])

	layer_4 = tf.nn.relu(layer_4)



	#Output layer with sigmoid activation

	out_layer = tf.matmul(layer_4, weights['out'])+biases['out']

	out_layer = tf.nn.sigmoid(out_layer)

	return out_layer

#Define the weights and the biases for each layer



weights = {

	'h1':tf.Variable(tf.truncated_normal([n_dim, n_hidden_1])),

	'h2':tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),

	'h3':tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3])),

	'h4':tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4])),

	'out':tf.Variable(tf.truncated_normal([n_hidden_4, n_class]))

}



biases = {

	'b1':tf.Variable(tf.truncated_normal([n_hidden_1])),

	'b2':tf.Variable(tf.truncated_normal([n_hidden_2])),

	'b3':tf.Variable(tf.truncated_normal([n_hidden_3])),

	'b4':tf.Variable(tf.truncated_normal([n_hidden_4])),

	'out':tf.Variable(tf.truncated_normal([n_class])),

}

#Initialize the variables

init = tf.global_variables_initializer()



saver = tf.train.Saver()
#call your model defined

y = multilayer_perceptron(x, weights, biases)
#Define the cost function and optimizer

cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))

training_steps = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

sess = tf.Session()

sess.run(init)

#Calculate the cost and accuracy for epoch



mse_history = []

accuracy_history = []



for epoch in range(training_epochs):

	sess.run(training_steps, feed_dict={x:train_x, y_:train_y})

	cost = sess.run(cost_function, feed_dict={x:train_x, y_:train_y})

	cost_history = np.append(cost_history, cost)

	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



	#print("Accuracy:", (sess.run(accuracy, feed_dict={x:test_x, y:test_y})))

	pred_y = sess.run(y, feed_dict={x: test_x})

	mse = tf.reduce_mean(tf.square(pred_y - test_y))

	mse_ = sess.run(mse)

	mse_history.append(mse_)

	accuracy = (sess.run(accuracy, feed_dict={x:train_x, y_:train_y}))

	accuracy_history.append(accuracy)



	print('epoch : ', epoch, '-', 'cost : ',cost, " - MSE : ",mse_, "- Train Accuracy : ", accuracy)



save_path = saver.save(sess, model_path)

print("Model saved in file : %s" % save_path)
#plot mse and accuracy graph



plt.plot(mse_history, "r")

plt.show()

plt.plot(accuracy_history)

plt.show()



#Print the final accuracy



correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print("Test Accuracy : ", (sess.run(accuracy, feed_dict={x:test_x, y_:test_y})))



#print the final mean square error



pred_y = sess.run(y, feed_dict={x:test_x})

mse = tf.reduce_mean(tf.square(pred_y - test_y))

print("MSE : %.4f"%sess.run(mse))
