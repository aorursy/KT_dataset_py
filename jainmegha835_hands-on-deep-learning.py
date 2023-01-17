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
def sigmoid(x):

    return 1/(1+np.exp(-x))

sigmoid(3)
def tanh(x):

    numerator = 1-np.exp(-2*x)

    denominator = 1+np.exp(-2*x)

    return numerator/denominator

tanh(3)
def ReLU(x):

    if x<0:

        return 0

    else:

        return x

    

ReLU(3)
def leakyReLU(x,alpha=0.01):

    if x<0:

        return (alpha*x)

    else:

        return x

leakyReLU(3,0.01)
import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

    
X=np.array([[0,1],[1,0],[1,1],[0,0]])

y=np.array([[1],[1],[0],[0]])
num_input=2

num_hidden=5

num_output=1
Wxh = np.random.randn(num_input,num_hidden)

bh = np.zeros((1,num_hidden))
Why=np.random.randn(num_hidden,num_output)

by=np.zeros((1,num_output))
def sigmoid(z):

    return 1/(1+np.exp(-z))
def sigmoid_derivative(z):

    

    return np.exp(-z)/((1+np.exp(-z))**2)
def forward_prop(X,Wxh,Why):

    z1 = np.dot(X,Wxh) + bh

    a1 = sigmoid(z1)

    z2 = np.dot(a1,Why) + by

    y_hat = sigmoid(z2)

    return z1,a1,z2,y_hat
def backword_prop(y_hat, z1, a1, z2):

    delta2 = np.multiply(-(y-y_hat),sigmoid_derivative(z2))

    dJ_dWhy = np.dot(a1.T, delta2)

    delta1 = np.dot(delta2,Why.T)*sigmoid_derivative(z1)

    dJ_dWxh = np.dot(X.T, delta1)

    return dJ_dWxh, dJ_dWhy

def cost_function(y,y_hat):

    J=0.5*sum((y-y_hat)**2)

    return J
alpha = 0.01

num_iterations = 5000
cost =[]

for i in range(5000):

    z1,a1,z2,y_hat = forward_prop(X,Wxh,Why)

    dJ_dWxh, dJ_dWhy = backword_prop(y_hat, z1, a1, z2)

#update weights

    Wxh = Wxh -alpha * dJ_dWxh

    Why = Why -alpha * dJ_dWhy

#compute cost

    c = cost_function(y, y_hat)

    cost.append(c)

plt.grid()

plt.plot(range(5000),cost)

plt.title('Cost Function')

plt.xlabel('Training Iterations')

plt.ylabel('Cost')