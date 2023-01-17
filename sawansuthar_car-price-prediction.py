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
df = pd.read_csv("/kaggle/input/used-car-dataset-ford-and-mercedes/hyundi.csv")
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
print(df.shape)
df.head(20)

df["year"].value_counts()
df["year"] = 2020-df["year"]
sns.pairplot(df,x_vars=[ 'model','year','transmission','mileage', 'fuelType', 'tax(£)', 'mpg', 'engineSize'],y_vars=['price'])
plt.figure(figsize=(38,8))
plt.subplot(231)
sns.countplot(df["model"],hue = df["fuelType"])
plt.legend(loc='upper center')
plt.subplot(232)
sns.countplot(df["fuelType"])
plt.subplot(233)
sns.countplot(df["transmission"],hue = df["fuelType"])
plt.subplot(234)
sns.countplot(df["year"],hue = df["fuelType"])



sns.countplot(df["engineSize"],hue = df["fuelType"])
plt.figure(figsize=(10,10))
sns.distplot(df["mpg"])
plt.figure(figsize=(10,10))
sns.heatmap(df.corr())
df["mpg"].unique()
X = df.drop(axis=1,columns=['price'])
Y = df.price
X = pd.get_dummies(X,columns = ['model','year','transmission','fuelType'])
plt.figure(figsize=(20,15))
sns.heatmap(X.corr())
X.corrwith(Y).plot(kind="barh",grid=True,xlim = (-1,1),xticks = np.linspace(start=-1,stop=1,num=21),figsize=(10,15),fontsize=12)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
sc_1 = StandardScaler()
y = Y.values
y = sc_1.fit_transform(y.reshape(-1,1))
x_train,x_test,y_train,y_test = train_test_split(X,y,train_size=0.8)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
print(lr.score(x_train,y_train))
print(lr.score(x_test,y_test))
      
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(x_train,y_train)
print(rfr.score(x_train,y_train))
print(rfr.score(x_test,y_test))