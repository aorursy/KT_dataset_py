# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import tensorflow as tf
# Simple tensorflow example

a = tf.Variable(3,name="a")
b = tf.Variable(5,name='b')
s = a+b
print(s)
init = tf.global_variables_initializer()

# to evaluate the variables
with tf.Session() as sess: # session
    init.run()             # initialization
    print ( s.eval() )     # evaluation
# MNIST 

train_path = '../input/train.csv'
test_path = '../input/test.csv'
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

train.head()
train.iloc[:,0].values[:10]
import seaborn as sns
sns.countplot(train.iloc[:,0].values)
train.iloc[2,0:][1]
import matplotlib.pyplot as plt


def display_sample(num):
    plt.imshow(train.iloc[num,1:].values.reshape([28,28]),  cmap=plt.get_cmap('gray_r'))
               
display_sample(10)

# (x_train.iloc[0].values.reshape([28,28]))

import pandas as pd
print(train.shape)
train = pd.get_dummies(train, columns=['label'])
print(train.shape)
pd.set_option('display.max_columns', None)
train.head()
# Seprate X and y
train_X = train.iloc[:, :-10].values
train_y = train.iloc[:, -10:].values
test_t=test.values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_X = sc.fit_transform(train_X)
train_t = sc.transform(test_t)
#Split trainning and testing case，make sure every digit have equal chance in both testing and trainning set.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(train_X, train_y,
                                                    test_size=0.2,
                                                    stratify=train_y)
x_train.shape
y_test.shape
# Create a graph
g1 = tf.Graph()

with g1.as_default():
    X = tf.placeholder(tf.float32, [None,784]) #cols are fixed
    w = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))
    
    Ylogits = tf.matmul(X,w)+b
    
    Y  = tf.nn.softmax(Ylogits)
    Y_ = tf.placeholder(tf.float32,[None,10])
    
    # loss function
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits,labels=Y_))
    
    #Evaluate the model
    correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    optimizer = tf.train.GradientDescentOptimizer(0.5)#0.5 is 
    train_step = optimizer.minimize(cross_entropy)
    
# define the function to calculate
def runmodel():
    init = tf.global_variables_initializer()
    sess.run(init)
    
    batch_size=100 # each time how many case input to NN for trainning
    epoch=20 # how many time the NN view the whole data set
    iterations=int(x_train.shape[0]/batch_size)
    
    batchnumber=0
   # mini batch
    for e in range(epoch):
        for i in range(iterations):
            batchnumber= batchnumber+1
            batch_start_idx = (i * batch_size) % (x_train.shape[0] - batch_size)
            batch_end_idx = batch_start_idx + batch_size
            batch_X = x_train[batch_start_idx:batch_end_idx]
            batch_Y = y_train[batch_start_idx:batch_end_idx]
            train_data = {X:batch_X,Y_:batch_Y}
            # train
            sess.run(train_step, feed_dict=train_data)
        print ("Epoch"+str(e+1))
        print ("batch: "+ str(batchnumber+1))
        ans=sess.run(accuracy,feed_dict={X:x_test,Y_:y_test}) # evaluate the testing dataset.
        print(ans)
# run the first ANN
with tf.Session( graph = g1) as sess:
    runmodel()