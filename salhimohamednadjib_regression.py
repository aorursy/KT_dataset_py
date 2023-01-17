# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('../input/bike-sharing-dataset-for-linear-regression/bike_sharing_data.txt')
#show data details

print('data = \n' , data.head(10))

print('******************')

print('data.describe = \n' , data.describe())

print('*******************')

#draw data

data.plot(kind='scatter' , x='Population' , y='Profit' , figsize =(5,5))
#adding a new column called ones before the data 

data.insert(0, 'Ones' , 1)

print('new data = \n' , data.head(10))

### Separate X(training data) from y(target variable)

cols = data.shape[1]

X = data.iloc[ : , 0:2]

y = data.iloc[ : , 2:3]

print('X data = \n' , X.head(10))

print('y data = \n' , y.head(10))
# Convert from data frames to numpy matrices

X = np.matrix(X.values)

y = np.matrix(y.values)

theta = np.matrix(np.array([0,0]))

print('X \n' , X)

print('X.shape = ' ,X.shape)

print('theta \n' , theta)

print('theta.shape = ' , theta.shape)

print('y \n', y)

print('y.shape = ' , y.shape)





data.shape
#cost function 

def computeCost(X , y , theta):

    z = np.power(((X * theta.T) - y), 2)

    print('z \n' , z)

    print('m ', len(X))

    return np.sum(z) / (2 * len(X))



print('computeCost(X , y , theta ) = ' , computeCost(X , y , theta))
## GD function 

def gradientDescent(X, y, theta, alpha, iters ):

    temp = np.matrix(np.zeros(theta.shape))

    parameters = int(theta.ravel().shape[1])

    cost = np.zeros(iters)

    

    for i in range(iters):

        error = (X * theta.T) - y

        

        for j in range(parameters):

            term = np.multiply(error, X[:,j])

            temp[0,j] = theta[0,j] - ((alpha/len(X)) * np.sum(term))

            

        theta = temp

        cost[i] = computeCost(X, y, theta)

        

    return theta, cost
#initialize variables for learning rate and iterations

alpha = 0.01

iters = 1000
#perform gradient descent to 'fit' the model parameters

g, cost = gradientDescent(X, y , theta, alpha, iters)

print('g = ' , g)

print('cost = ' , cost[0:50])

print('computeCost = ' , computeCost(X , y , g))
# get best fit line 



x = np.linspace(data.Population.min(), data.Population.max(), 100)

print('x \n', x)

print('g \n', g)



f = g[0, 0] + (g[0, 1] * x)

print('f \n' , f)

#draw the line

fig, ax = plt.subplots(figsize=(5,5))

ax.plot(x, f, 'r' , label='Prediction')

ax.scatter(data.Population, data.Profit, label='training Data')

ax.legend(loc=2)

ax.set_xlabel('Population')

ax.set_ylabel('Profit')

ax.set_title('Predicted Profit vs. Population size')
# draw error graph 

fig, ax = plt.subplots(figsize=(5,5))

ax.plot(np.arange(iters), cost, 'g')

ax.set_xlabel('Iterations')

ax.set_ylabel('Cost')

ax.set_title('Error vs.training Epoch')




