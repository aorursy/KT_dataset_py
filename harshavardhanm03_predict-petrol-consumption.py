# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

df=pd.read_excel("../input/measurements2.xlsx")

print(df.head())

# Any results you write to the current directory are saved as output.
import seaborn as sns
sns.heatmap(df.isnull())
null_values=df.isnull().sum().sort_values(ascending=False)
ax=sns.barplot(null_values.index,null_values.values)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
import matplotlib.pyplot as plt
plt.show()
df.drop(['refill gas','refill liters','specials'],axis=1,inplace=True)
sns.heatmap(df.isnull())
temp_inside_mean=np.mean(df['temp_inside'])
print(temp_inside_mean)
df['temp_inside'].fillna(temp_inside_mean,inplace=True)
sns.heatmap(df.isnull())
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
l=LinearRegression()
x=df.drop(['consume','gas_type'],axis=1)

y=df['consume']
l.fit(x,y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
l.fit(x_train,y_train)
y_pred=l.predict(x_test)
print(l.coef_,l.intercept_)
from sklearn import metrics
print(metrics.mean_squared_error(y_test,y_pred))
print(metrics.mean_absolute_error(y_test,y_pred))
print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
""""from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['gas_type'] = le.fit_transform(df['gas_type'])"""
dum1=pd.get_dummies(df['gas_type'])
print(dum1)
df=pd.concat([df,dum1],axis=1)
df.drop('gas_type',axis=1,inplace=True)
x1=df.drop('consume',axis=1)
y1=df['consume']
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
l=LinearRegression()
x_train,x_test,y_train,y_test=train_test_split(x1,y1,test_size=0.3,random_state=42)
l.fit(x_train,y_train)
y_pred_1=l.predict(x_test)
print(y_pred_1)
from sklearn import metrics
print(np.sqrt(metrics.mean_squared_error(y_test,y_pred_1)))