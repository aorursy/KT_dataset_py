# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import seaborn as sns

%matplotlib inline



from sklearn.metrics import r2_score



from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression

import statsmodels.api as sm  



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('/kaggle/input/random-linear-regression/train.csv')

df.head()
print(df.isnull().sum())

print(df[df['y'].isnull()])

df.describe()
# removing null column value this also treats outlier

df=df[~(df['y'].isnull())]

df.describe()
df.boxplot(column=["x"])

df.plot.scatter(x=['x'],y=['y'])
y_train=df['y']

x_train=df['x']

x_train = sm.add_constant(x_train)

lm = sm.OLS(y_train,x_train).fit() 
y_train_pred=lm.predict(x_train)
fig = plt.figure()

sns.distplot((y_train - y_train_pred), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18)    
lm.summary()
df_test=pd.read_csv('/kaggle/input/random-linear-regression/test.csv')

x_test=df_test['x']

x_test = sm.add_constant(x_test)
y_test_pred=lm.predict(x_test)
y_test=df_test['y']

fig = plt.figure()

sns.distplot((y_test - y_test_pred), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18)  


r2_score(y_test, y_test_pred)
# Plotting y_test and y_pred to understand the spread.

fig = plt.figure()

plt.scatter(y_test,y_test_pred)

fig.suptitle('y_test vs y_test_pred', fontsize=20)              # Plot heading 

plt.xlabel('y_test', fontsize=18)                          # X-label

plt.ylabel('y_pred', fontsize=16)    
