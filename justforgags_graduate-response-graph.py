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
tf.reset_default_graph()
train=pd.read_csv('../input/Admission_Predict.csv')
train.columns = train.columns.astype(str)
train.head()
train=train.drop(['Serial No.'],axis=1)
admit=train['Chance of Admit ']
train=train.drop(['Chance of Admit '],axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train, admit, test_size=0.33, random_state=42)
X_train.shape
y_train.shape
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
num_neurons=100
batch_size=200

X=tf.placeholder(dtype=tf.float32)
Y=tf.placeholder(dtype=tf.float32)

W=tf.Variable(tf.zeros(shape=(7,1)),dtype=tf.float32)
b=tf.Variable(tf.random_normal(shape=(batch_size,1)),dtype=tf.float32)
Z=tf.matmul(X,W)+b
print(Y)
#a=tf.sigmoid(Z)
error=tf.reduce_mean(tf.square(Y-Z)) #[200] - []
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)
training=optimizer.minimize(error)
init=tf.global_variables_initializer()

admit.shape
from sklearn.model_selection import train_test_split
with tf.Session() as sess:
    sess.run(init)
    num_epochs=20000
    for i in range(num_epochs):
        rand_in=np.random.randint(0,X_train.shape[0]-batch_size-1)
        X_in=X_train[rand_in:rand_in+batch_size]
        
        admits=y_train[rand_in:rand_in+batch_size]
        #print(X_in)
        #print(admits)
        #print(X_in.shape,admits.shape)
        sess.run(training,feed_dict={X:X_in,Y:admits})
        #print(W.eval(sess))
        
        if i%1000==0:
            print("Mean square error at epoch {} is {}".format(i,sess.run(error,feed_dict={X:X_in,Y:admits})))
            
    model_m,model_b=sess.run([W,b])
mean_bias=np.mean(model_b)
mean_bias
y_predicts_test=np.matmul(X_test,model_m)+mean_bias
y_predicts_test.shape
y_predicts_test.shape
import matplotlib.pyplot as plt

plt.plot(y_test[:60],[i for i in range(60)],'b')
plt.plot(y_predicts_test[:60],[i for i in range(60)],'r')
plt.legend(['Actual','Predicted'])
train.columns
X_vect=[316,96,1,4,4,9.9,1]
y_predicts_my_score=np.matmul(X_vect,model_m)
print('Chances of getting in university is {}'.format(y_predicts_my_score))
X_vect=[340,120,1,5,5,5.10,1]
y_predicts_my_score=np.matmul(X_vect,model_m)
print('Chances of getting in university is {}'.format(y_predicts_my_score))
X_vect=[340,120,3,5,5,5.10,1]
y_predicts_my_score=np.matmul(X_vect,model_m)
print('Chances of getting in university is {}'.format(y_predicts_my_score))
