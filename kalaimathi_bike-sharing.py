# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
bike_data=pd.read_csv('/kaggle/input/bike-sharing-dataset/hour.csv')
plt.title('Season Vs Count')
g=sns.barplot(bike_data['season'],bike_data['cnt'])
g.set(xticklabels=['spring','sumer','fall','winter'])
fig_dims = (15,4)
fig, ax = plt.subplots(figsize=fig_dims)
bike_data.groupby(['yr','mnth'])['cnt'].sum().plot(kind='bar',ax=ax)
plt.xlabel(' Month')
plt.ylabel('Count')
plt.title('Monthwise count for 2011,2012')

sns.pairplot(bike_data,x_vars=['holiday','weekday','workingday'],y_vars='cnt')

plt.title('WindSpeed vs Count')
sns.scatterplot(x=bike_data['windspeed'],y=bike_data['cnt'])
sns.pairplot(bike_data,x_vars=['temp','atemp','hum'],y_vars='cnt')
plt.title('weathersit Vs Count')
sns.barplot(bike_data['weathersit'],bike_data['cnt'])
#This plot shows correlation between the features
fig_dims = (15,5)
fig, ax = plt.subplots(figsize=fig_dims)
sns.heatmap(bike_data.corr(),annot=True,cmap='winter',linewidths=0.25,linecolor='magenta',ax=ax)

X=bike_data.iloc[:,2:13].values
y=bike_data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y)
from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor()
dtr.fit(X_train,y_train)

y_pred=dtr.predict(X_test)
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
print("R square :",r2_score(y_test,y_pred))
print("MAE :",mean_absolute_error(y_test,y_pred))
print("MSE :",mean_squared_error(y_test,y_pred))
print("RMSE :",mean_squared_error(y_test,y_pred)**0.5)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X,y)
y_pred2=lr.predict(X_test)

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
print("R square :",r2_score(y_test,y_pred2))
print("MAE :",mean_absolute_error(y_test,y_pred2))
print("MSE :",mean_squared_error(y_test,y_pred2))
print("RMSE :",mean_squared_error(y_test,y_pred2)**0.5)