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
import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score

plt.style.use('ggplot')
train_dirty = pd.read_csv('/kaggle/input/train.csv')

test_dirty = pd.read_csv('/kaggle/input/test.csv')
train_dirty.head()
test_dirty.head()
# identify any missing values in train_dirty and test_dirty

print(train_dirty.isnull().sum())

print(test_dirty.isnull().sum())
# calculate mean of y in train_dirty data frame and fill NaN value

avg_y = train_dirty['y'].mean()

train_dirty['y'].replace(np.nan, avg_y, inplace = True)

train_dirty.isnull().sum()
# shape of data

train_dirty.shape, test_dirty.shape
# use seaborn residplot to see if a linear relationship exists

sns.scatterplot(train_dirty['x'], train_dirty['y'])

plt.show()
# remove outlier (max value of x or y)

max_x = train_dirty['x'].idxmax()

train_dirty.drop(max_x, inplace = True)
sns.scatterplot(train_dirty['x'], train_dirty['y'])

plt.show()
df_train = pd.DataFrame(data=train_dirty)

x_train = np.array(train_dirty['x']).reshape(-1,1) # reshape data 

y_train = train_dirty['y']

x_test = np.array(test_dirty['x']).reshape(-1,1) # reshape data

y_test = test_dirty['y']



x_train.shape, y_train.shape, x_test.shape, y_test.shape
# residual plot 

sns.residplot(x_train, y_train)

plt.show()
# scatter plot

sns.scatterplot(df_train['x'], df_train['y'])

plt.show()
df_train.corr()
# create a linear model and fit to training data

lm = LinearRegression().fit(x_train, y_train)
# estimate value of y uisng x_test

yhat = lm.predict(x_test)
plt.scatter(x_test, y_test, color='r', marker='.')

plt.plot(x_test, yhat, color ='b', lw=2)

plt.xlabel('X')

plt.ylabel('Y')

plt.show()
# rsquared using LinearRegression.score()

r2 = lm.score(x_test,y_test)



# rsquared using r2_score from sklearn.metrics library

#r2_skl = r2_score(y_test, yhat)



# mean squared error

mse = mean_squared_error(y_test, yhat)

mse



# root mean squared error

rmse = np.sqrt(mse)

rmse



print('R2: ', r2)

print('MSE: ', mse)

print('RMSE: ', rmse)