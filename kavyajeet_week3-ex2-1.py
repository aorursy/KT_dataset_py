import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/ex2data1.csv')

data.head()
# extracting the variables

X = data.iloc[:,0:2]

Y = data.iloc[:,2]
# preprocessing

# introducting intercept column

ones = np.ones((len(data),1))

X = np.concatenate((ones,X),axis=1)

n,m = np.shape(X)

Y = np.array(Y).reshape(n,1)



np.shape(X), np.shape(Y)
data0 = data[data['label']==0]

data1 = data[data['label']==1]



plt.scatter(data0['x'],data0['y'])

plt.scatter(data1['x'],data1['y'])

plt.xlabel('exam1')

plt.ylabel('exam2')
def sigmoid(x):

    return 1/(1+np.exp(-x))
theta = np.zeros(shape=(m,1))

def error(theta,x,y):

    h1 = y.T@np.log(sigmoid(x@theta)) + (1-y).T@np.log(1-sigmoid(x@theta))

    return -(1/n)*h1

error(theta,X,Y)
def gradient_descent(theta,X,y):

    return (1/n)*(X.T@(sigmoid(X@theta)-y))

gradient_descent(theta,X,Y)
from scipy.optimize import fmin_tnc

temp = fmin_tnc(func = error, x0 = theta.flatten(),fprime = gradient_descent, args = (X, Y.flatten()))

theta_fit = temp[0]

theta_fit
error(theta_fit,X,Y)
# prediction

X_i = np.array([1,60,60])

sigmoid(X_i@theta_fit)
# for plotting purpose we will convert the probability value to int

def convert_value(p):

    if p<=0.5:

        return 0

    else:

        return 1

X_i = np.array([1,90,90])

pi = sigmoid(X_i@theta_fit)

convert_value(pi)
xs = np.linspace(min(data['x']),max(data['y']),num=10)

ys = np.linspace(min(data['x']),max(data['y']),num=10)



X_grid,Y_grid = np.meshgrid(xs,ys)

Z_grid = np.zeros(shape=(len(ys),len(xs)))

for i in range(len(xs)):

    for j in range(len(ys)):

        value = np.array([1,xs[i],ys[j]])@theta_fit

        Z_grid[j,i] = convert_value(value)

        

from matplotlib.colors import ListedColormap

fig = plt.figure(figsize=(15,10))

plt.contourf(X_grid,Y_grid,Z_grid,cmap=ListedColormap(('blue','orange')),alpha=0.4)

plt.scatter(data0['x'],data0['y'],label='No Admission')

plt.scatter(data1['x'],data1['y'],label='Got Admission')

plt.xlabel('exam1')

plt.ylabel('exam2')

plt.title('Linear Logistic Regression')

plt.legend()
prediction = X@theta_fit

y_pred = []

for pred in prediction:

    y_pred.append(convert_value(pred))

y_pred = np.array(y_pred).reshape(-1,1)

correct = np.sum(Y==y_pred)

accuracy = correct/len(Y)*100

print('accuracy is %.2f%%' % accuracy)