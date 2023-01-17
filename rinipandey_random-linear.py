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
# simple linear regression using library function
from sklearn.model_selection import train_test_split 
train = pd.read_csv('../input/random-linear-regression/train.csv')
test = pd.read_csv('../input/random-linear-regression/test.csv')
train.dropna(inplace = True)
test.dropna(inplace = True)
x_train = train['x']
y_train = train['y']
x_test = test['x']
y_test = test['y']

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

x_train = x_train.reshape(-1,1)
x_test = x_test.reshape(-1,1)
train.describe()
test.describe()
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score, mean_squared_error

clf = LinearRegression(normalize=True)
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
print(r2_score(y_test,y_pred))
# simple linear regression without library function
mean_x = np.mean(x_train)
mean_y = np.mean(y_train)

N = len(x_train)

s_xx = 0
s_xy = 0

for i in range(N):
    s_xy += (x_train[i] - mean_x)*(y_train[i] - mean_y)
    s_xx += (x_train[i]-mean_x)**2
b1 = s_xy/s_xx
b0 = mean_y - b1*mean_x
print(b0)
print(b1)
sst = 0
ssr = 0
N = len(x_test)
for i in range(N):
    y_pred = b1*x_test[i]
    sst += (y_test[i] - mean_y)**2
    ssr += (y_test[i] - y_pred)**2
r2 = 1 - (ssr/sst)
print(r2)
import matplotlib.pyplot as plt 

y_prediction = b0 + b1 * x_test

y_plot = []
for i in range(100):
    y_plot.append(b0 + b1 * i)
plt.figure(figsize=(10,10))
plt.scatter(x_test,y_test,color='red',label='Scatter Plot')
plt.plot(range(len(y_plot)),y_plot,color='black',label = 'regression line')
plt.xlabel('X in testing data-set')
plt.ylabel('Y in testing data-set')
plt.legend()
plt.show()
# Lasso regression
from sklearn import linear_model
reg = linear_model.Lasso(alpha=10)
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
# ridge regression
from sklearn import linear_model
reg = linear_model.Ridge(alpha=10)
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
# Elastic-net regression
from sklearn import linear_model
reg = linear_model.ElasticNet(random_state=0)
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
# bridge regression
from sklearn import linear_model
reg = linear_model.BayesianRidge()
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.xlabel('x')
plt.ylabel('y')
plt.show()