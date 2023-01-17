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
from __future__ import absolute_import 

from __future__ import division

from __future__ import print_function

import numpy as np 

import tensorflow as tf
# Define a hello tensor as a tf.constant

hello=tf.constant("Hello,tensorflow!")

print(hello)



#Define a TF session

sess=tf.Session()



# Run the hello tensor under the session 

print(sess.run(hello))
# Deine cosstant Scalar 

ex_tensor=tf.constant(3)

tf.shape(ex_tensor)

tf.rank(ex_tensor)

print(ex_tensor)
# Defining a List

ex_tensor=tf.constant([1,2,3])

#print(tf.shape(ex_tensor))

#tf.rank(ex_tensor)

print(ex_tensor)
#Defining a matrix 

ex_tensor=tf.constant([[1,2,3],[4,5,6]])

tf.shape(ex_tensor)

tf.rank(ex_tensor)

print(ex_tensor)
#Defining a 3d array 

ex_tensor=tf.constant([[[1,2,3]],[[7,8,9]]])

tf.shape(ex_tensor)

tf.rank(ex_tensor)

print(ex_tensor)
a=tf.constant(10)

b=tf.constant(20)

#a+b

with tf.Session() as sess:

    result=sess.run(a+b)   

result
const=tf.constant(10)
fill_mat=tf.fill((4,4),10) # Tensor filled matrix 

fill_mat
myzeros=tf.zeros((4,4)) # Created 4x4 Zero tensor 
myones=tf.ones((4,4))# Creates 4x4 one tensor
myrandn=tf.random_normal((4,4),mean=0,stddev=1.0) # Creates a random normal distribution tensor with mean 0 and standard deviation 1
myrandu=tf.random_uniform((4,4),minval=0,maxval=1) #Creates a unifrom distribution tensor with minimum value as 0 and maximum value as 1
# Prenting all the results 

my_ops=[const,fill_mat,myzeros,myones,myrandn,myrandu]

sess=tf.InteractiveSession() # INteractive Session helps to get the result out of the Session

for op in my_ops:

    print(sess.run(op))

    print('\n')
a=tf.constant([[1,2],[3,4]])

a.get_shape()
b=tf.constant([[10],[100]])

b.get_shape()
result=tf.matmul(a,b)

sess.run(result)
result.eval()
a=tf.constant([3.0,2.0],dtype=tf.float32)

b=tf.constant([4.0,1.0])  #also tf.float32 implicitly 

total=a+b

print(a)

print(b)

print(total)
sess=tf.Session()

print(sess.run(total))
#Build a graph 

x=tf.placeholder(tf.float32)

y=tf.placeholder(tf.float32)

z=x+y



#Define session for executing the graph 

# feed_dict argument of the run method to feed concrete values 

print(sess.run(z,feed_dict={x:3,y:4.5}))

print(sess.run(z,feed_dict={x:[1,3],y:[2,5]}))
import matplotlib.pyplot as plt 

import tensorflow as tf

import numpy as np 

import pandas as pd

from sklearn.preprocessing import LabelEncoder 

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split
def read_dataset():

    df=pd.read_csv('../input/sonar.all-data.csv')

   # print(len(df.columns))

    X=df[df.columns[0:60]].values

    y=df[df.columns[60]]

#Encode the independent variable 

    encoder=LabelEncoder()

    encoder.fit(y)

    y=encoder.transform(y)

    Y=one_hot_encode(y)

    print(X.shape)

    return(X,Y)
def one_hot_encode(labels):

    n_labels=len(labels)

    n_unique_labels=len(np.unique(labels))

    one_hot_encode=np.zeros((n_labels,n_unique_labels))

    one_hot_encode[np.arange(n_labels),labels]=1

    return one_hot_encode  

    
X,Y=read_dataset()
X,Y=shuffle(X,Y,random_state=1)
train_x,test_x,train_y,test_y=train_test_split(X,Y,test_size=0.2,random_state=415)
print(train_x.shape)

print(train_y.shape)

print(test_x.shape)
learning_rate=0.3

training_epochs=1000

cost_history=np.empty(shape=[1],dtype=float)

n_dim=X.shape[1]

print('n_dim',n_dim)

n_class=2

#model_path=""
n_hidden_1=60

n_hidden_2=60

n_hidden_3=60

n_hidden_4=60



x=tf.placeholder(tf.float32,[None,n_dim])

W=tf.Variable(tf.zeros([n_dim,n_class]))

b=tf.Variable(tf.zeros([n_class]))

y_=tf.placeholder(tf.float32,[None,n_class])
def multilayer_preceptron(x,weights,biases):

    # Hidden layer with Relu activised 

    

    layer_1=tf.add(tf.matmul(x,weights['h1']),biases['b1'])

    layer_1=tf.nn.sigmoid(layer_1)

    

    # Hidden layer with Sigmoid activation 

    layer_2=tf.add(tf.matmul(layer_1,weights['h2']),biases['b2'])

    layer_2=tf.nn.sigmoid(layer_2)

    

     # Hidden layer with Sigmoid activation 

    layer_3=tf.add(tf.matmul(layer_2,weights['h3']),biases['b3'])

    layer_3=tf.nn.sigmoid(layer_3)

    

     # Hidden layer with Relu activation 

    layer_4=tf.add(tf.matmul(layer_3,weights['h4']),biases['b4'])

    layer_4=tf.nn.sigmoid(layer_4)

    

    # Output layer with Linear Activation 

    out_layer=tf.matmul(layer_4,weights['out']) + biases['out']

    return out_layer
weights={

    'h1':tf.Variable(tf.truncated_normal([n_dim,n_hidden_1])),

    'h2':tf.Variable(tf.truncated_normal([n_hidden_1,n_hidden_2])),

    'h3':tf.Variable(tf.truncated_normal([n_hidden_2,n_hidden_3])),

    'h4':tf.Variable(tf.truncated_normal([n_hidden_3,n_hidden_4])),

    'out':tf.Variable(tf.truncated_normal([n_hidden_4,n_class])),

}



biases={

    'b1':tf.Variable(tf.truncated_normal([n_hidden_1])),

    'b2':tf.Variable(tf.truncated_normal([n_hidden_2])),

    'b3':tf.Variable(tf.truncated_normal([n_hidden_3])),

    'b4':tf.Variable(tf.truncated_normal([n_hidden_4])),

    'out':tf.Variable(tf.truncated_normal([n_class])),

}
init=tf.global_variables_initializer()

saver=tf.train.Saver()
y=multilayer_preceptron(x,weights,biases)
cost_function=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_))

training_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)



sess=tf.Session()

sess.run(init)
mse_history=[]

accuracy_history=[]

sess.run(init)



for epoch in range(training_epochs):

    sess.run(training_step,feed_dict={x:train_x,y_:train_y})

    cost=sess.run(cost_function,feed_dict={x:train_x,y_:train_y})

    cost_history=np.append(cost_history,cost)

    correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))

    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    pred_y=sess.run(y,feed_dict={x:test_x})

    mse=tf.reduce_mean(tf.square(pred_y-test_y))

    mse_=sess.run(mse)

    mse_history.append(mse_)

    accuracy=(sess.run(accuracy,feed_dict={x:train_x,y_:train_y}))

    accuracy_history.append(accuracy)

    

    print('epoch:',epoch,' - ','cost:',cost,'- MSE: ',mse_,"-Train Accuracy:",accuracy)

    

save_path=saver.save(sess,model_path)

print('Model saved in file: %s'% save_path)
plt.plot(mse_history,'r')

plt.show()

plt.plot(accuracy_history)

plt.show()
correct_pridiction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))

accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

print("Test Accuracy:",(sess.run(accuracy,feed_dict={x:test_x,y_:test_y})))
pred_y=sess.run(y,feed_dict={x: test_x})

mse=tf.reduce_mean(tf.square(pred_y-test_y))

print("MSE: %.4f" % sess.run(mse))