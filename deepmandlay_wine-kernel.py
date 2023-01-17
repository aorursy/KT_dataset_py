#import libraries
import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
df = pd.read_csv('../input/wine.csv')
#let's take a look at our dataset
df.head()
df.info()
df.describe()
x = df[df.columns[1:13]].values
x[:2]
#display first two values
y = df[df.columns[0]].values -1
y[:2]
#display first two values
#session
sess = tf.Session()
#let's convert Y into binary form
Y = tf.one_hot(indices = y, depth=3, on_value = 1., off_value = 0., axis = 1 ,name = "a").eval(session=sess)
#indices = data = y
#dept = 3 as we have 3 different types of wine
#on_value = 1 : to indicate which type of wine
#axis = 1 as its an array
Y[:2]
#display first two values i.e. 0, 1
#normalize our dataset for faster calculation
scaler = preprocessing.StandardScaler() 
X = scaler.fit_transform(x) 
#transform data into train & test
#train dataset contains 80% of data, test consist of 20% data 
train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size=0.20, random_state=101)
train_x.shape,train_y.shape,test_x.shape,test_y.shape
#142 rows, 12 columns
#142 rows, 3 columns values from Y to binary form
#define and initialize the variables to work with the tensors
# w(t+1) = w(t) - learning_rate * gradient of error term 
#learning rate : update limit of each weight in each iteration
learning_rate = 0.1
#epoch : number of iteration for training our dataset
#low epoch for less load
#take alteast 1000 and see our result i.e. cost
#if cost value is constant / increasing after sometime, see the lowset value and that is our global minima
epoch = 1000
#to store cost value
cost_history = np.empty(shape=[1],dtype=float)
#dimension of x
no_dim = x.shape[1]
no_dim
#class of Y, we had encoded it 3 : 0,1,
no_class = 3
#[none] : it can be of any number of rows
#n_dim : number of input features / columns
#b : bias term
x = tf.placeholder(tf.float32,[None,no_dim])
#number of columns of 1st matrix should be equal to no of rows of 2nd matrix
w = tf.Variable(tf.zeros([no_dim,no_class]))
b = tf.Variable(tf.zeros([no_class]))
#initialize all variables.
init = tf.global_variables_initializer()
#define the cost function
y_ = tf.placeholder(tf.float32,[None,no_class])
#matrix multiplication of x ,w
y = tf.nn.softmax(tf.matmul(x, w)+ b)
#cross entropy : y_ * tf.log(y)
#reduce_sum : summation of all errors
#reduce_mean : taking mean of all values
cross_entropy = tf.reduce_mean(-tf.reduce_sum((y_ * tf.log(y)),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy) 
#initialize the session
sess = tf.Session()
sess.run(init)
#we need to store loss at each iteration, starting with empty list
mse_history = []
accuracy_history =[]
for epoch in range(epoch):
    sess.run(train_step,feed_dict=({x: train_x, y_: train_y}))
    cost= sess.run (cross_entropy, feed_dict={x: train_x, y_: train_y})
    cost_history = np.append(cost_history,cost)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    pred_y = sess.run(y, feed_dict={x: test_x})
    mse = tf.reduce_mean(tf.square(pred_y - test_y))
    mse_ = sess.run(mse)
    mse_history.append(mse_)
    accuracy = (sess.run(accuracy, feed_dict={x: test_x, y_: test_y}))
    accuracy_history.append(accuracy)
    print('epoch : ', epoch,  ' - ', 'cost: ', cost, '-', 'accuracy :',accuracy)
    
print("Accuracy : ",accuracy)
