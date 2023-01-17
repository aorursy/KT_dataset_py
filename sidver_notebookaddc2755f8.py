import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/insurance/insurance.csv')

df.head()
df.isnull().sum()
df.describe()
print(df['sex'].nunique())

print(df['smoker'].nunique())

print(df['region'].nunique())
import matplotlib.pyplot as plt

import seaborn as sns
sns.boxplot(x='age',data=df)
sns.boxplot(x='bmi',data=df)
i = df[df['bmi']>45].index

df.drop(i,inplace=True)
sns.boxplot(x='bmi',data=df)
sns.pairplot(df)
sns.heatmap(df.corr())
X = df.iloc[:,:6].values

y = df.iloc[:,6].values
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

X[:,1] = le.fit_transform(X[:,1])

X[:,4] = le.fit_transform(X[:,4])

X[:,5] = le.fit_transform(X[:,5])
X
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=41)
from sklearn.linear_model import Lasso

from sklearn.model_selection import GridSearchCV

ls = Lasso()

parameters = {'alpha':[0.00001,0.001,0.01,0.0001,1,5,10,100]}

lsreg = GridSearchCV(ls,parameters,scoring='neg_mean_squared_error',cv=5)

lsreg.fit(X_train,y_train)
y_pred = lsreg.predict(X_test) 
from sklearn import metrics

mse = metrics.mean_squared_error(y_test,y_pred)

mae = metrics.mean_absolute_error(y_test,y_pred)

mr = metrics.r2_score(y_test,y_pred)
print(mse)

print(mae)

print(mr)
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators=300)

rfr.fit(X_train,y_train)
y_pred = rfr.predict(X_test) 
from sklearn import metrics

mse = metrics.mean_squared_error(y_test,y_pred)

mae = metrics.mean_absolute_error(y_test,y_pred)

mr = metrics.r2_score(y_test,y_pred)
print(mse)

print(mae)

print(mr)
from sklearn.linear_model import LinearRegression

le = LinearRegression()

le.fit(X_train,y_train)

y_pred = le.predict(X_test)
from sklearn import metrics

mse = metrics.mean_squared_error(y_test,y_pred)

mae = metrics.mean_absolute_error(y_test,y_pred)

mr = metrics.r2_score(y_test,y_pred)
print(mse)

print(mae)

print(mr)