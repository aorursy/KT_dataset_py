# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import math
train_data=pd.read_csv("../input/titanic/train.csv")
train_data.head()
test_data=pd.read_csv("../input/titanic/test.csv")
test_data.head()
train_data

train_data=train_data.fillna(train_data.mean())
train_data['Cabin']=train_data['Cabin'].fillna(0)

X_train=train_data[['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare']]
X_train.set_index('PassengerId',inplace=True)
Y_train=train_data[['Survived']]
for index in range(X_train.shape[0]):
    if X_train['Sex'].iloc[index]=='female':
        X_train['Sex'].iloc[index]=1
    else:
        X_train['Sex'].iloc[index]=0
X_train
def initialize_parameters(X):
    W=np.random.randn(X.shape[1],1)
    b=np.zeros([X.shape[0],1])
    return W,b
import numpy as np
def sigmoid(Z):
    a=np.zeros(Z.shape)
    a=1/(1+np.exp(-Z))
    return a
import numpy as np
def cost_function(X, Y, W, b):
    m=len(Y)
    Z=np.zeros([X.shape[0],1])
    Z=np.dot(X,W)+b
    yhat=sigmoid(Z)
    L=(-Y*np.log(yhat))-((1-Y)*np.log(1-yhat))
    J=(np.sum(L))/m
    return J, yhat
import numpy as np
def update_parameters(W, b, a, Y, X, learning_rate=1.2):
    dZ=a-Y
    dW=np.dot(X.T,dZ)
    W=W-learning_rate*dW
    db=dZ
    b=b-learning_rate*db
    return W, b
def model(X, Y, numiter=1200):
    W, b = initialize_parameters(X)
    cost, yhat = cost_function(X, Y, W, b)
    print("Cost after 1st iteration =", cost)
    for i in range(numiter):
        Wnew, bnew = update_parameters(W, b, yhat, Y, X)
        costnew = cost_function(X, Y, Wnew, bnew)
        print("Cost after"+str(i)+"th iteration =", costnew)
    return None
model(X_train, Y_train)
