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
df = pd.read_csv('/kaggle/input/insurance/insurance.csv')
df.head()
df.region.value_counts()
df.isnull().sum()
df.bmi.mean()
import matplotlib.pyplot as plt
import seaborn as sns
sns.barplot(x='smoker',y='charges',data=df)
sns.catplot(x='charges',y='smoker',data=df)
sns.scatterplot(x='bmi',y='charges',data=df)
df.groupby(['sex','smoker']).size()
sns.factorplot(x='sex',data=df,kind='count',hue='smoker')
sns.barplot(x='region',y='charges',data=df)
sns.scatterplot(x='age',y='bmi',data=df)
df.describe()
df.corr()
sns.heatmap(df.corr(),annot=True)
df.head()
df = pd.get_dummies(df,drop_first=True)
df.head()
X = df.drop('charges',axis=1)
y=df[['charges']]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 100)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
y_pred = lin_reg.predict(X_test)
r2_score(y_test,y_pred)
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_train)

regressor_lin_2 = LinearRegression()
regressor_lin_2.fit(X_poly,y_train)
y_pred_poly = regressor_lin_2.predict(poly.fit_transform(X_test))
r2_score(y_test,y_pred_poly)
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0,max_depth=5)
regressor.fit(X_train,y_train)

y_pred_dt = regressor.predict(X_test)
r2_score(y_test,y_pred_dt)
from sklearn.ensemble import RandomForestRegressor
regressor_rf = RandomForestRegressor(n_estimators=1000,random_state=0)
regressor_rf.fit(X_train,y_train)
y_pred_rf = regressor_rf.predict(X_test)
r2_score(y_test,y_pred_rf)
