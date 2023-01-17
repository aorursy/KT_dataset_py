
#import the required libraries
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import  shuffle
from sklearn.model_selection import train_test_split
sns.set(style="darkgrid")
%matplotlib inline
import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
#read our csv file
df = pd.read_csv("../input/sonar.csv")
#information about dataset
df.head(3)
df.info()
df.describe()
#total number of columns in our dataset
len(df.columns)
#1 to 60 are input parameters
x = df[df.columns[1:60]].values
x
#61 is output
y = df[df.columns[60]]
y
#normalize our dataset for faster, less complex computation. 
#normalize our dataset
def normalize(value):
    mean = np.mean(value, axis = 0)
    std = np.std(value, axis = 0)
    normalize = (value - mean)/ std
    return normalize
normalize = normalize(x)
#as our output has 2 variables we need to convert it, by encoding
#encode the depedent variable, single it has more than one class
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
y
#x is 2d array, y is 1d array we need to convert y into 2d array
#2d array conversion for Y
def array_convert(labels):
    no_labels = len(labels)
    unique_no_labels = len(np.unique(labels))
    array = np.zeros((no_labels, unique_no_labels))
    array[np.arange(no_labels), labels] = 1
    return array
Y = array_convert(y)
Y
#all of our values are converted
x,Y,y,normalize
#we can see that our data is linear order, we need to shuffle it and then we can use it for testing and training purpose
#Transform the data in training and testing
x,Y = shuffle(x,Y,random_state=1)
#test_size = 0.20 : it indicates that 80% for training & 20% for testing
train_x,test_x,train_y,test_y = train_test_split(x,Y,test_size=0.20, random_state=42)
#define and initialize the variables to work with the tensors
# w(t+1) = w(t) - learning_rate * gradient of error term 
#learning rate : update limit of each weight in each iteration
learning_rate = 0.1
#epoch : number of iteration for training our dataset
#low epoch for less load
#take alteast 1000 and see our result i.e. cost
#if cost value is constant / increasing after sometime, see the lowset value and that is our global minima
epoch = 500

#to store cost value
cost_history = np.empty(shape=[1],dtype=float)
#dimension of x
no_dim = x.shape[1]
no_dim
#class of Y, we had encoded it 2 : 0,1
no_class = 2
#[none] : it can be of any number of rows
#n_dim : number of input features / columns
#b : bias term
x = tf.placeholder(tf.float32,[None,no_dim])
W = tf.Variable(tf.zeros([no_dim,no_class]))
b = tf.Variable(tf.zeros([no_class]))
#initialize all variables.
#mandatory to initalize our variable
init = tf.global_variables_initializer()
#define the cost function
y_ = tf.placeholder(tf.float32,[None,no_class])
#matrix multiplication of x ,W
y = tf.nn.softmax(tf.matmul(x, W)+ b)
#cross entropy : y_ * tf.log(y)
#we can also use predefined function, but for better explanation
#reduce_sum : summation of all errors
#reduce_mean : taking mean of all values
cost_function = tf.reduce_mean(-tf.reduce_sum((y_ * tf.log(y)),reduction_indices=[1]))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)
#initialize the session
sess = tf.Session()
sess.run(init)
#we need to store loss at each iteration, starting with empty list
mse_history = []
#calculate the cost for each epoch
#epoch is number of iterations
for epoch in range(epoch):
    sess.run(training_step,feed_dict={x:train_x,y_:train_y})
    cost = sess.run(cost_function,feed_dict={x: train_x,y_: train_y})
    cost_history = np.append(cost_history,cost)
    pred_y = sess.run(y, feed_dict={x: test_x})
    #when cost is reducing at each iteration, it indicates we are doing a good job and model is performing better
    print('epoch : ', epoch,  ' - ', 'cost: ', cost)


    mse = tf.reduce_mean(tf.square(pred_y - test_y))
    mse_history.append(sess.run(mse))

#we can see that our cost value is decreasing at every iteration

#print the final mean square error
plt.plot(mse_history, 'ro-')
plt.show()
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Accuracy: ",(sess.run(accuracy, feed_dict={x: test_x, y_: test_y})))
#accuray can be increased by changing learning rate, epoch
# In[15]:

plt.plot(range(len(cost_history)),cost_history)
plt.axis([0,epoch,0,np.max(cost_history)])
plt.show()
