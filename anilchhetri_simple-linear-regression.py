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
train = pd.read_csv('/kaggle/input/random-linear-regression/train.csv')

test = pd.read_csv('/kaggle/input/random-linear-regression/test.csv')
train.head()
import matplotlib.pyplot as plt
_ = plt.boxplot(train.x)
#finding the outliers Value

train[train['x'] > 500]
train.iloc[213]
#removing the outliers

train.drop(213, axis=0, inplace=True)
fig, ax = plt.subplots(1,2, sharey=True)

ax[0].set_title('BoxPlot of X variables')

ax[0].grid(True)

ax[0].set(xticklabels=['X'])

_ = ax[0].boxplot(train.x)



ax[1].set(title='boxplot of Y variables', xticklabels=['y'])

ax[1].grid(True)

_ = ax[1].boxplot(train.y)



#checking for null values

train.isnull().sum()
plt.scatter(train.x, train.y)

plt.xlabel('x Variables (features)')

plt.ylabel('y variables (target variable)')

plt.title('Scatter plot of X and Y variables')

ax = plt.gca()

ax.spines['right'].set_visible(False)

ax.spines['top'].set_visible(False)
train.corr()
from sklearn.linear_model import LinearRegression
slr = LinearRegression()

slr.fit(train[['x']], train['y'])

slr.score(train[['x']], train['y'])
slr.score(test[['x']], test['y'])
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
y_train_predict = slr.predict(train[['x']])

y_test_predict = slr.predict(test[['x']])
plt.scatter(train['x'], train['y'])

plt.plot(train['x'], y_train_predict, 'r', linewidth=2)

plt.title('Train y and predicted values')

plt.ylabel('Y value')

plt.xlabel('x Value')

ax = plt.gca()

ax.spines['right'].set_visible(False)

ax.spines['top'].set_visible(False)
plt.scatter(test['x'], test['y'])

plt.plot(test['x'], y_test_predict, 'r', linewidth=2)

plt.title('Test y and predicted values')

plt.ylabel('Y value')

plt.xlabel('x Value')

ax = plt.gca()

ax.spines['right'].set_visible(False)

ax.spines['top'].set_visible(False)
print('The MAE of Train Data',mean_absolute_error(train['y'], y_train_predict))

print('The MSE of Train Data',mean_squared_error(train['y'], y_train_predict))

print('The R-squred of Train Data',r2_score(train['y'], y_train_predict))
print('The MAE of Test Data',mean_absolute_error(test['y'], y_test_predict))

print('The MSE of Test Data',mean_squared_error(test['y'], y_test_predict))

print('The R-squred of Test Data',r2_score(test['y'], y_test_predict))