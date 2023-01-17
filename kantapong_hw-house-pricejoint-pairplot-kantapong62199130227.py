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

import numpy as np

from sklearn import metrics

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# Importing the Boston Housing dataset

from sklearn.datasets import load_boston

boston = load_boston()
# Initializing the dataframe

df = pd.DataFrame(boston.data)
df.head()
#Adding the feature names to the dataframe

df.columns = boston.feature_names

df.head()
df['PRICE'] = boston.target 
df.shape
df.columns
df.dtypes
df.nunique()
df.isnull().sum()
# See rows with missing values

df[df.isnull().any(axis=1)]
# Viewing the data statistics

df.describe()
# Finding out the correlation between the features

corr = df.corr()

corr.shape
plt.figure(figsize=(10,10))

sns.heatmap(corr, cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':10}, cmap='inferno')
# Spliting target variable and independent variables

X = df.drop(['PRICE'], axis = 1)

y = df['PRICE']
df.dtypes
pair = sns.pairplot(df[['CRIM','INDUS','NOX','AGE','PRICE']], palette = 'hls',size=2)

pair.set(xticklabels=[]);
joint = sns.jointplot(data=df,x='AGE', y='PRICE',kind='reg',color="#0033cc")
print(X.shape)
print(y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=2019)
X_train.shape
X_test.shape
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
y_pred
y_test
#Fitting the regular linear regression model to the training data set

import statsmodels.api as sm



X_train_sm = X

X_train_sm = sm.add_constant(X_train_sm)



lm_sm = sm.OLS(y,X_train_sm.astype(float)).fit()



# lm_sm.params
print(lm_sm.summary())
lm_sm.summary()