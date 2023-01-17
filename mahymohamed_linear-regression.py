# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Reading the train data:

X = pd.read_csv('../input/online-news-classification/train.csv')
print(X.shape)
X.head()
# Setting the target

Y = (np.mat(X['data_channel'])).T
print(Y.shape)
Y
# Dropping the target from the train Data and prepping the train matrix
X.drop('data_channel' , axis = 1, inplace = True)
X.drop('tag' , axis = 1, inplace = True)
X = np.mat(X)
n,m = X.shape
X0 = np.ones((n,1))
X = np.hstack((X0,X))
n,m = X.shape
# Making the Weights matrix

theta = np.random.randint(0 ,1000,(m,1))
theta =np.mat(theta)
print(theta.shape)
theta
# Function that returns the Hypothesis

def hypothesis(theta , x): 
    h = np.dot(x, theta)
    return h
# function that returns the cost for each run of the loop
# where in M: number of rows, H: hypothesis, Y: the target
def compute_cost( m , h , y ):
    
    cost = 1/(2 * m) * sum(h-y)
    return cost
#Updating the weights using gradient descent, using a learning rate of 0.005

def gradient_descent():
    
    thetaNew = theta - (0.005) * (X.T.dot(h - Y))
    return thetaNew
# fitting everything together:

i= 0
numOfIterations = 20
m = len(Y)
while i <= numOfIterations:
    h = hypothesis(theta ,X)
    c = compute_cost( m , h , Y )
    newTheta = gradient_descent()
    i = i + 1
    
test  = pd.read_csv('../input/online-news-classification/test.csv')
test.drop('tag' , axis = 1, inplace = True)
test.drop('Id' , axis = 1, inplace = True)
test.insert(loc=0, column ='name', value =[1] *test.shape[0])
newX = test
newX.head()
newY = hypothesis(newTheta, newX)
print(newY)