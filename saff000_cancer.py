# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score
data = pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")
data = data.drop(['Unnamed: 32','id'],axis=1)
sns.countplot(x='diagnosis',data=data)
x = data.iloc[:,1:32].values
y = data['diagnosis'].values
encode = {
    'M' : 1,
    'B' : 0
}
for i in range(569) :
  y[i] = encode[y[i]]
y = y.reshape(1,569)
y = y.T
temp = x.transpose()
''''
Min-Max Scaling 
'''
for i in range(30) : 
  max = np.amax(temp[i])
  min = np.amin(temp[i])
  for j in range(569) :
    temp[i][j] = (temp[i][j]-min) / ( max - min)
x = temp.transpose()
x = x.astype('float32')
y = y.astype('float32')
'''
initialization of weights and biased
'''
input_dim = x.shape[1]
output_dim = 1
hidden_neurons = 60
alpha = 0.0000001
n_samples = 569
costs = []

w1 = np.random.randn(input_dim,hidden_neurons)
b1 = np.zeros((1,hidden_neurons))
w2 = np.random.randn(hidden_neurons,hidden_neurons)
b2 = np.zeros((1,hidden_neurons))
w3 = np.random.randn(hidden_neurons,output_dim)
b3 = np.zeros((1,1))

def sigmoid(x) :
#   print((1 + np.exp(-x)))
  return (1 / (1 + np.exp(-x)))
  
def relu(x) :
  return np.maximum(x,0)

def derivative_relu(x) :
  x[x<0] = 0
  x[x>0] = 1
  return x


for _ in range(50000) :
    '''
    feed forward
    '''
    z1 = np.dot(x,w1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1,w2) + b2
    a2 = relu(z2)
    z3 = np.dot(a2,w3) + b3
    a3 = sigmoid(z3)
    
#     cost = - (y*np.log(a3)) + ((1-y)*np.log(1-a3))
    cost = np.mean(((a3-y)**2))
    costs.append(cost)
    
    '''
    back propogate
    '''
    #ouput
    dz3 = 2 * (a3 - y) * ( sigmoid(z3)- (1-sigmoid(z3)) ) 
    dw3 = (np.dot(a2.T,dz3))/569
    db3 = (np.sum(dz3,axis=0,keepdims=True))/569

    #hidden layer 2
    dz2 = np.dot(dz3,w3.T) * derivative_relu(z2)
    dw2 = (np.dot(a1.T,dz2))/569
    db2 = (np.sum(dz2,axis=0))/569

    #hidden layer 1
    dz1 = np.dot(dz2,w2.T) * derivative_relu(z1)
    dw1 = (np.dot(x.T,dz1))/569
    db1 = (np.sum(dz1,axis=0))/569
    
    '''
    update
    '''
    w3-=(alpha*dw3)
    w2-=(alpha*dw2)
    w1-=(alpha*dw1)
    
    b3-=(alpha*db3)
    b2-=(alpha*db2)
    b1-=(alpha*db1)
costs
z1 = np.dot(x,w1) + b1 
a1 = relu(z1)
z2 = np.dot(a1,w2) + b2
a2 = relu(z2)
z3 = np.dot(a2,w3) + b3
a3 = sigmoid(z3)    
'''
prediction time
'''
y_pred = []
for i in a3 :
    if i< 0.5 :
        y_pred.append(0)
    else :
        y_pred.append(1)
y_true = y.tolist()
accuracy_score(y_true,y_pred)*100
import tensorflow as tf
inputs = tf.keras.layers.Input(shape=(30,))
ouput =  tf.keras.layers.Dense(64,activation='relu')(inputs)
output = tf.keras.layers.Dense(64,activation='relu')(inputs)
output =  tf.keras.layers.Dense(1,activation='sigmoid')(inputs)
model =  tf.keras.models.Model(inputs,output)
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x,y,epochs=50)
'''
prediction time
'''
a3 = model.predict(x)
y_pred = []
for i in a3 :
    if i< 0.5 :
        y_pred.append(0)
    else :
        y_pred.append(1)
y_true = y.tolist()
accuracy_score(y_true,y_pred)*100