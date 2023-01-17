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
import tensorflow as tf

node1=tf.constant(3.0,tf.float32)

node2=tf.constant(4.0)

print(node1,node2)
sess=tf.Session()

print(sess.run([node1,node2]))

sess.close()
with tf.Session() as sess:

    output=sess.run([node1,node2])

    print(output)


#Build a graph

a=tf.constant(5.0)

b=tf.constant(6.0)



c=a*b



#Launch the graph in a session 

sess=tf.Session()



#Evaluate the tensor 'c'

print(sess.run(c))



sess.close()



a=tf.placeholder(tf.float32)

b=tf.placeholder(tf.float32)



adder_node=a + b

sess=tf.Session()



print(sess.run(adder_node,{a:[1,3],b:[2,4]}))
import tensorflow as tf 



# Model Parameters 

W=tf.Variable([.3],tf.float32)   # Variable with initial value 0.3

b=tf.Variable([-.3],tf.float32)   # Variable with initial value -0.3



# Inputs and Outputs 

x=tf.placeholder(tf.float32)



linear_model= W*x + b



y=tf.placeholder(tf.float32)



# Loss Function 



squared_delta=tf.square(linear_model-y)

loss=tf.reduce_sum(squared_delta)



init=tf.global_variables_initializer() #to initalize all the variables in a tensor flow program



sess=tf.Session()

sess.run(init)

print(sess.run(linear_model,{x:[1,2,3,4],y:[0,-1,-2,-3]}))
# Model Parameters 

W=tf.Variable([.3],tf.float32)   # Variable with initial value 0.3

b=tf.Variable([-.3],tf.float32)   # Variable with initial value -0.3



# Inputs and Outputs 

x=tf.placeholder(tf.float32)



linear_model= W*x + b



y=tf.placeholder(tf.float32)



# Loss Function 



squared_delta=tf.square(linear_model-y)

loss=tf.reduce_sum(squared_delta)



# Optimize

optimizer=tf.train.GradientDescentOptimizer(0.01)

train=optimizer.minimize(loss)



init=tf.global_variables_initializer()  #to initalize all the variables in a tensor flow program



sess=tf.Session()

sess.run(init)



for i in range (500):

    sess.run(train,{x:[1,2,3,4],y:[0,-1,-2,-3]})

#print(sess.run(linear_model,{x:[1,2,3,4],y:[0,-1,-2,-3]}))



print(sess.run([W,b]))

import numpy as np

import matplotlib.pyplot as plt 

import pandas as pd 

import warnings

warnings.filterwarnings('ignore') 
dataset=pd.read_csv('../input/Churn_Modelling.csv')

dataset.head()
X=dataset.iloc[:,3:13].values

#X

y=dataset.iloc[:,13].values
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelencoder_X_1=LabelEncoder()

X[:,1] = labelencoder_X_1.fit_transform(X[:,1]) #Encoding the values of column Country

labelencoder_X_2=LabelEncoder()

X[:,2] = labelencoder_X_2.fit_transform(X[:,2])

onehotencoder=OneHotEncoder(categorical_features=[1])

X=onehotencoder.fit_transform(X).toarray()

X=X[:,1:]

#X
from sklearn.model_selection import train_test_split   #cross_validation doesnt work any more

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0) 

#X_train
from sklearn.preprocessing import StandardScaler 

sc_X=StandardScaler()

X_train=sc_X.fit_transform(X_train)

X_test=sc_X.fit_transform(X_test)

#X_train
import keras 

from keras.models import Sequential 

from keras.layers import Dense 
classifier=Sequential()
classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#classifier.fit(X_train,y_train, batch_size=10, nb_epoch=100)

classifier.fit(X_train, y_train,batch_size=10,nb_epoch=100)
y_pred=classifier.predict(X_test)

y_pred=(y_pred>0.5)

#y_pred
from sklearn.metrics import confusion_matrix  #Class has capital at the begining function starts with small letters 

cm=confusion_matrix(y_test,y_pred)

import seaborn as sns

import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.title("Test for Test Dataset")

plt.xlabel("predicted y values")

plt.ylabel("real y values")

plt.show()