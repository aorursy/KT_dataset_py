from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression 

from sklearn.tree import DecisionTreeRegressor 

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder 

from sklearn.metrics import mean_squared_error as mse
import seaborn as sns
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

nRowsRead = 1259 # specify 'None' if want to read whole file

# MagicBricks.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows

df1 = pd.read_csv('/kaggle/input/MagicBricks.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = 'MagicBricks.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
df1.head(5)
df1.describe()
sns.countplot(df1['BHK'])
df1['BHK'].value_counts()
df1[df1['BHK']==6]
df1.drop([721,345,163,164,261,352,353,585],inplace=True)  ##### these are indexes of rows, being removed from dataset
sns.countplot(df1['Bathroom'])
df1[df1['Bathroom']==6]
df1[df1['Bathroom']==7]
df1.drop([225,495,527,659,676,681,1211,248,1029],inplace=True)
df1.Furnishing.value_counts().plot.bar()
plt.figure(figsize=(14,7))

sns.boxplot(x=df1.Furnishing,y=df1.Price)
df1.isnull().sum()
df1.Parking.fillna(0,inplace=True)
sns.countplot(df1.Parking)
df1[df1.Parking==39]
df1['Parking'].replace([39,114],1,inplace=True)

df1['Parking'].replace([5,9,10],4,inplace=True)
sns.countplot(df1.Status)
sns.countplot(df1['Type'])
sns.boxplot(x=df1.Transaction,y=df1.Price)
plt.figure(figsize=(14,7))

sns.scatterplot(x=df1.Area,y=df1.Price)
df1.drop('Per_Sqft',axis=1,inplace=True)
df1.isnull().sum()
df1.Bathroom.fillna(df1.Bathroom.median(),inplace=True)

df1.Type.fillna('Apartment',inplace=True)

df1.Furnishing.fillna('Semi-Furnished',inplace=True)
df1.Locality.unique()
df1.drop('Locality',axis=1,inplace=True)
df1 = pd.get_dummies(df1)

Y = df1.Price

X = df1.drop('Price',axis=1)
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2,random_state = 42)
x_train
lr = LinearRegression()

lr.fit(x_train,y_train)   ########### traing model

pred = lr.predict(x_test) ########### Getting predictions
pred
from math import sqrt
print(sqrt(mse(y_test,pred)))  ######## Root Mean Square Error 
lr = DecisionTreeRegressor()

lr.fit(x_train,y_train)

pred = lr.predict(x_test)

print(sqrt(mse(y_test,pred)))
sns.lineplot(x=df1.Area,y=df1.Price)
plt.figure(figsize=(14,7))

sns.scatterplot(x=df1.Area,y=df1.Price)
p = np.array(df1[df1.Area>5000].index)
df1.drop(p,inplace=True)  ##### these are indexes of rows, being removed from dataset
plt.figure(figsize=(14,7))

sns.regplot(x="Area", y="Price", data=df1)
Y = df1.Price

X = df1.drop('Price',axis=1)

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2,random_state = 42)

lr = DecisionTreeRegressor()

lr.fit(x_train,y_train)

pred = lr.predict(x_test)

print(sqrt(mse(y_test,pred)))
Y = df1.Price

X = df1.drop('Price',axis=1)

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2,random_state = 42)

lr = LinearRegression()

lr.fit(x_train,y_train)

pred = lr.predict(x_test)

print(sqrt(mse(y_test,pred)))