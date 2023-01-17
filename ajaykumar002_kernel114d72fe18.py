import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df=pd.read_csv('/kaggle/input/tvradionewspaperadvertising/Advertising.csv')
df.head()
df.tail()
df
df.isnull().sum()
df.dtypes
df.describe()
df.corr()
plt.figure(figsize=(15,8))

sns.heatmap(df.corr(),annot=True,cmap='Blues')
df.head()
X = df.drop(['Sales'],axis=1) 
X
y = df[['Sales']]
y
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

from sklearn.preprocessing import MinMaxScaler

import statsmodels.api as sm

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
lr = LinearRegression()
lr.fit(X_train,y_train)
pred = lr.predict(X_test)
pred
r2_score(y_test,pred)
X_train
X_train_sm = sm.add_constant(X_train)
lr = sm.OLS(y_train,X_train_sm).fit()
lr.summary()
X_train.drop(['Newspaper'],axis=1,inplace=True)

X_test.drop(['Newspaper'],axis=1,inplace=True)
X_train
X_train_sm = sm.add_constant(X_train)

lr = sm.OLS(y_train,X_train_sm).fit()

lr.summary()
lr = LinearRegression()

lr.fit(X_train,y_train)

pred = lr.predict(X_test)

r2_score(y_test,pred)
sns.distplot(y_test-pred)
plt.scatter(y_test,pred)