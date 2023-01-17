# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import pylab as pl

import seaborn as sns



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("../input/FuelConsumptionCo2.csv")



# take a look at the dataset

df.head()

# summarize the data

df.describe()
#selecting relavent data

cdf = df[['ENGINESIZE', 'CO2EMISSIONS']]

cdf.head(5)
type(cdf)
sns.set(style='whitegrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [14, 5]})

ax = sns.countplot(x="ENGINESIZE", data=cdf)
sns.barplot(x = "ENGINESIZE", y = "CO2EMISSIONS", data = cdf, ci = False)
sns.scatterplot(x = "ENGINESIZE", y = "CO2EMISSIONS", data = cdf, ci = False)
msk = np.random.rand(len(df)) < 0.8

train = cdf[msk]

test = cdf[~msk]
train.head(2)
type(train)
type(train['ENGINESIZE'])
X = train['ENGINESIZE'].values.reshape(-1,1) 

# eshape(-1,1) tells python to convert the array into a matrix with one coloumn. “-1” tells python to figure out the rows by itself. 

# .values: extracts a numpy array with the values of your pandas Series object and then reshapes it to a 2D array.
type(X)
noOfTrainEx = X.shape[0] # no of training examples

print("noOfTrainEx: ",noOfTrainEx)

noOfWeights = X.shape[1]+1 # no of features+1 => weights

print("noOfWeights: ", noOfWeights)
ones = np.ones([noOfTrainEx, 1]) # create a array containing only ones 

X = np.concatenate([ones, X],1) # cocatenate the ones to X matrix

theta = np.ones((1, noOfWeights)) #np.array([[1.0, 1.0]])
y = train['CO2EMISSIONS'].values.reshape(-1,1) # create the y matrix
print(X.shape)

print(theta.shape)

print(y.shape)
# Setting hyper parameter values

# notice small alpha value

alpha = 0.01

iters = 3000
## Creating cost function

def computeCost(X, y, theta):

    h = X @ theta.T

    error = h-y

    loss = np.power(error, 2) 

    J = np.sum(loss)/(2*noOfTrainEx)

    return J
computeCost(X, y, theta) #Computing cost now produces very high cost
## Gradient Descent funtion

def gradientDescent(X, y, theta, alpha, iters):

    cost = np.zeros(iters)

    for i in range(iters):

        theta = theta - (alpha/len(X)) * np.sum((X @ theta.T - y) * X, axis=0)

        cost[i] = computeCost(X, y, theta)

        if i % 100 == 0: # just look at cost every ten loops for debugging

            print(i, 'iteration, cost:', cost[i])

    return (theta, cost)
g, cost = gradientDescent(X, y, theta, alpha, iters)  
print(g, cost)
#plot the cost

fig, ax = plt.subplots()  

ax.plot(np.arange(iters), cost, 'r')  

ax.set_xlabel('Iterations')  

ax.set_ylabel('Cost')  

ax.set_title('Error vs. Training Epoch')
axes = sns.scatterplot(x = "ENGINESIZE", y = "CO2EMISSIONS", data = cdf, ci = False)

x_vals = np.array(axes.get_xlim()) 

y_vals = g[0][0] + g[0][1]* x_vals #the line equation

plt.plot(x_vals, y_vals, '--')
from sklearn import linear_model

regr = linear_model.LinearRegression()

train_x = np.asanyarray(train[['ENGINESIZE']])

train_y = np.asanyarray(train[['CO2EMISSIONS']])

regr.fit (train_x, train_y)

# The coefficients

print ('Coefficients: ', regr.coef_)

print ('Intercept: ',regr.intercept_)
axes = sns.scatterplot(x = "ENGINESIZE", y = "CO2EMISSIONS", data = cdf, ci = False)

x_vals = np.array(axes.get_xlim()) 

y_vals = regr.intercept_[0] + regr.coef_[0][0]* x_vals #the line equation

plt.plot(x_vals, y_vals, '--')
from sklearn.metrics import r2_score



test_x = np.asanyarray(test[['ENGINESIZE']])

test_y = np.asanyarray(test[['CO2EMISSIONS']])

test_y_hat = regr.predict(test_x)



print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))

print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))

print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )