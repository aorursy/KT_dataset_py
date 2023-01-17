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

pd.set_option('display.max_rows',None)



import numpy as np

import warnings

warnings.filterwarnings('ignore')



import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.model_selection import train_test_split
data = pd.read_csv('../input/headbrain/headbrain.csv')

data.head()
x = data['Head Size(cm^3)']

y = data['Brain Weight(grams)']

n = len(y)
train_x, test_x, train_y, test_y = train_test_split(x,y, test_size=0.3)

train_x.shape , train_y.shape
plt.scatter(train_x, train_y)

plt.xlabel('Head Size(cm^3)')

plt.ylabel('Brain Weight(grams)')

plt.show()
mean_x = np.mean(train_x)

mean_y = np.mean(train_y)

num = 0

denom = 0



# for i in range(n):

#     num = num + (x[i] - mean_x)*(y[i] - mean_y)

#     denom = denom + ((x[i] - mean_x))**2

    

num = np.dot(np.subtract(train_x,mean_x), np.subtract(train_y,mean_y))

denom = np.dot(np.subtract(train_x,mean_x), np.subtract(train_x,mean_x))



m = num/denom

c = mean_y - (m*mean_x)

print(m,c)
min_x = np.min(train_x)-100

max_x = np.max(train_x)+100

x_dummy = np.linspace(min_x,max_x,1000)

y_dummy = m * x_dummy + c



plt.scatter(train_x,train_y,color='g')

plt.plot(x_dummy,y_dummy,color='r')

plt.title('Simple Linear Regression')

plt.xlabel('Head size cm^3')

plt.ylabel('Brain weight in grams')
sum_pred = 0

sum_act = 0



for xi,yi in zip(train_x, train_y):

    y_pred = (m * xi + c)

    sum_pred += (y_pred - mean_y)**2

    sum_act += (yi - mean_y)**2



# r2 = 1-(sum_pred/sum_act)

r2 = sum_pred/sum_act

print(r2)



# Here we can observe that we got R**2> 0.5 . so we have good model
def predict(x):

    return m*x+c



print(predict(4177))
x = data['Head Size(cm^3)'].values

y = data['Brain Weight(grams)'].values

n = len(y)



x = x.reshape((len(x),1)) # Converting into 2d array

y = y.reshape((len(y),1))



train_x, test_x, train_y, test_y = train_test_split(x,y, test_size=0.3)



train_x.shape , train_y.shape
from sklearn import linear_model



reg = linear_model.LinearRegression(normalize=True)

reg.fit(train_x, train_y) # accepts 2d array

y_predict = reg.predict(test_x)
df = pd.DataFrame({'Actual': test_y.flatten(), 'Predicted': y_predict.flatten()})

df.head()
plt.scatter(test_x,test_y,color='g')

plt.plot(test_x,y_predict,color='r')

plt.title('Simple Linear Regression')

plt.xlabel('Head size cm^3')

plt.ylabel('Brain weight in grams')
from sklearn.metrics import mean_absolute_error



accuracy2 = mean_absolute_error(test_y, y_predict)

accuracy2
from sklearn.metrics import r2_score



accuracy = r2_score(test_y, y_predict)

print(accuracy)



weights = reg.coef_

intercept = reg.intercept_

print(weights, intercept)
x = data['Head Size(cm^3)']

y = data['Brain Weight(grams)']

x.shape, y.shape
def gradient_descent(x, y, m, c, alpha, iterations, n):



    # Performing Gradient Descent 

    for i in range(iterations): 

        y_guess = m*x + c  # The current predicted value of Y

        cost = 1/n * np.sum((y - y_guess)**2) # Cost function to check convergence of theta

        D_m = (-2/n) * np.sum(x * (y- y_guess))  # Derivative wrt m

        D_c = (-2/n) * np.sum(y - y_guess)  # Derivative wrt c

        m = m - alpha * D_m  # Update m

        c = c - alpha * D_c  # Update c

        costs.append(cost)

    return m,c, costs
n = len(x)

m = 0

c = 0

costs = []

alpha = 0.000000009 # The learning Rate

iterations = 30 # The number of iterations to perform gradient descent



m,c, costs = gradient_descent(x, y, m, c, alpha, iterations, n)
plt.plot(costs)

plt.ylabel('cost')

plt.xlabel('iterations (per hundreds)')

plt.title('Cost reduction over time')

plt.show()


y_guess = m*x+c



plt.scatter(x,y)

plt.xlabel('Head Size(cm^3)')

plt.ylabel('Brain Weight(grams)')

plt.plot([min(x), max(x)], [min(y), max(y)], color='red')

plt.show()
from sklearn.metrics import r2_score



accuracy = r2_score(y, y_guess)

print(accuracy)
def predict(x_):

    return m*x_+c
print(predict(4747))
data[data['Head Size(cm^3)'] == 4747]