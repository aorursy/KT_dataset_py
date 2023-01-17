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
df=pd.read_csv('/kaggle/input/80-cereals/cereal.csv')

df.head()
df.shape
df.isnull().sum()
numerical=df.select_dtypes(include=['int64','float64'])

numerical.head()
categorical=df.select_dtypes(include='object')

categorical.head()
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
sns.distplot(df['rating'])
sns.barplot(data=df,x='rating',y='sugars')
df.columns
sns.barplot(data=df,x='protein',y='rating')
sns.barplot(data=df,x='vitamins',y='rating')
sns.barplot(df['fat'],df['rating'])
df.columns
plt.figure(figsize=(15,10))

sns.barplot(df['sodium'],df['rating'])
sns.barplot(df['fiber'],df['rating'])
plt.figure(figsize=(15,10))

sns.barplot(df['carbo'],df['rating'])
corr=df.corr()
plt.figure(figsize=(15,10))

sns.heatmap(corr, cbar = True,  square = True, annot=True,cmap= 'coolwarm')
categorical.head()
dummy=pd.get_dummies(df[['name','mfr','type']])

column_name=df.columns.values.tolist() 

column_name.remove('name') 

column_name.remove('mfr')

column_name.remove('type')

data1=df[column_name].join(dummy) 
data1.head()
from sklearn.model_selection import train_test_split

y = data1['rating']

X = data1.drop('rating', axis=1)



# setting up testing and training sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=27)

print (X_train.shape)

print (y_train.shape)

print (X_test.shape)

print (y_test.shape)
from sklearn.linear_model import LinearRegression

lm=LinearRegression()

import statsmodels.api as sm

lm = sm.OLS(y_train, X_train).fit()

lm.summary()