import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

%matplotlib inline 
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df=pd.read_csv('/kaggle/input/focusing-on-mobile-app-or-website/Ecommerce Customers')
df.head()
df.isna().sum()
sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=df)
sns.jointplot(x='Avg. Session Length',y='Yearly Amount Spent',data=df)
sns.pairplot(df)
df.columns
X=df[ ['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]

y=df['Yearly Amount Spent']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,y_train)
print(lr.coef_)
predit=lr.predict(X_test)
plt.scatter(predit,y_test)

plt.xlabel('Predited value')

plt.ylabel('Amount spent')
sns.distplot(y_test-predit)
pd.DataFrame(lr.coef_, X.columns, columns = ['Coefficient'])
print(lr.intercept_)
from sklearn.metrics import r2_score

print('R Square',r2_score(y_test,predit)*100,'%')
from sklearn import metrics

print ("Mean absolute error",metrics.mean_absolute_error(y_test, predit))

print ('Mean squared error',metrics.mean_squared_error(y_test, predit))

print ('Root Mean squared error',np.sqrt (metrics.mean_squared_error(y_test, predit)))