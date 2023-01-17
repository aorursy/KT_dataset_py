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
data=pd.read_csv("../input/climate-change/climate_change.csv");
data.head()
data.shape
data.info()
data.describe()
data.columns
data.isnull().sum()
data.columns
X=data.drop(['Month'],axis=1)
X=X.drop(['Year'],axis=1)
X=X.drop(['Temp'],axis=1)
y=data['Temp']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
import seaborn as sns
import matplotlib.pyplot as plt
corr=data.corr()
fig,ax=plt.subplots(figsize=(15,10))
g=sns.heatmap(corr,ax=ax,annot=True)
ax.set_title('Correlation between variables')
sns.relplot(x='CO2',y='Temp',kind='line',data=data)
sns.relplot(x='CH4',y='Temp',kind='line',data=data)
sns.relplot(x='N2O',y='Temp',kind='line',data=data)
sns.relplot(x='CFC-11',y='Temp',kind='line',data=data)
sns.relplot(x='CFC-12',y='Temp',kind='line',data=data)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
model=LinearRegression()
model.fit(X_train,y_train)
model.coef_
y_pred=model.predict(X_test)
mean_squared_error(y_test,y_pred)
r2_score(y_test,y_pred)
print(model.score(X_train,y_train))
print(model.score(X_test,y_test))
from sklearn.tree import DecisionTreeRegressor
model=DecisionTreeRegressor(max_depth=5)
model.fit(X_train,y_train)
print(model.score(X_train,y_train))
print(model.score(X_test,y_test))
y_pred=model.predict(X_test)
print(mean_squared_error(y_test,y_pred))
print(r2_score(y_test,y_pred))
