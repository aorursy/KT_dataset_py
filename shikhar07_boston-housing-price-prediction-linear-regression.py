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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.datasets import load_boston
boston=load_boston()
boston.keys()
print (boston.DESCR)
X=boston.data

y=boston.target
X.shape
#Data Normalization

u=np.mean(X,axis=0)

std=np.std(X,axis=0)
X=(X-u)/std
X[:4,:4]
#Adding one extra column of ones to the X-dataset.

ones=np.ones((X.shape[0],1))

X=np.hstack((ones,X))
X[:4,:4]
print (X.shape,y.shape)

#Custom Linear Regression

def hypothesis(x,theta):

    y_=0.0

    n=x.shape[0]

    for i in range(n):

        y_+=theta[i]*x[i]

    return y_

def gradient(X,y,theta):

    m,n=X.shape

    grad=np.zeros((n,))

    for j in range(n):

        for i in range(m):

            y_=hypothesis(X[i],theta)

            grad[j]+=(y_-y[i])*X[i][j]

    return grad/m

def error(X,y,theta):

    m=X.shape[0]

    e=0.0

    for i in range(m):

        y_=hypothesis(X[i],theta)

        e+=(y_-y[i])**2

    return (e/m)

def grad_desc(X,y,lr=0.01,max_epochs=400):

    m,n=X.shape

    theta=np.zeros((n,))

    error_list=[]

    for r in range(max_epochs):

        e=error(X,y,theta)

        error_list.append(e)

        grad=gradient(X,y,theta)

        for j in range(n):

            theta[j]-=lr*grad[j]

    return theta, error_list

#def prediction(p):

 #   y_=0.0

  #  t=theta

   # for i in range(p.shape[0]):

    #    for j in range(p.shape[1]):

     #       y_+=np.dot(p[i][j],t[i][j])

    #return y_

    

theta,error_list=grad_desc(X,y)
theta
plt.plot(error_list)

plt.show()
#Prediction

y_=[]

for i in range(X.shape[0]):

    pred=hypothesis(X[i],theta)

    y_.append(pred)

y_=np.array(y_)
print (y_[:3],y[:3])
y_.shape
import seaborn as sns
y.shape,X.shape
error_list=np.array(error_list)

error_list
sns.set_style('darkgrid')

plt.scatter(y,y_)

plt.xlabel('Prices')

plt.title('Given Prices vs Predicted Prices')

plt.show()
def r2_score(y,y_):

    num=np.sum((y-y_)**2)

    den=np.sum((y-y.mean())**2)

    score=(1-num/den)

    return score*100
score=r2_score(y,y_)

score