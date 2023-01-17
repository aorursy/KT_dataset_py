import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

#import geopandas as gpd

import sklearn



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('../input/world-happiness/2015.csv')

data.head()
data.describe()
data.columns
data.info()
f,ax=plt.subplots(figsize=(10,10))

sns.heatmap(data.corr(),annot=True)#,linewidths=5,fmt='.1f',ax=ax)
#data_correlation = data.corr()

#data_correlation
plt.figure(figsize=(10,6))

sns.distplot(a=data['Happiness Score'])#, kde=False)
#plt.figure(figsize=(10,6))

#sns.distplot(a=data['Economy (GDP per Capita)'])#, kde=False)
plt.figure(figsize=(17,6))

sns.scatterplot(x=data['Happiness Score'], y=data['Economy (GDP per Capita)'], hue=data['Region'])
plt.figure(figsize=(17,6))

sns.scatterplot(x=data['Happiness Score'], y=data['Family'], hue=data['Region'])
plt.figure(figsize=(17,6))

sns.scatterplot(x=data['Happiness Score'], y=data['Health (Life Expectancy)'], hue=data['Region'])
plt.figure(figsize=(10,6))

plt.title("Average hapiness score for different regions")

sns.barplot(x=data['Happiness Score'], y=data['Region'])
pd.unique(data.Region)
data_p = pd.get_dummies(data,columns=['Region'])

data_p.info()

data_p.describe()
data.columns
#data_p.drop(['Country','Happiness Rank'], axis=1, inplace=True)
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
X = data['Economy (GDP per Capita)'].values.reshape(-1,1)

      # , 'Freedom', 'Trust (Government Corruption)',

      # 'Generosity', 'Dystopia Residual'].values

Y = data ['Happiness Score']
X.shape, Y.shape
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.3, random_state=0)
x_train.shape, y_train.shape
x_test.shape, y_test.shape
linear_reg = LinearRegression(normalize=True).fit(x_train,y_train)

linear_reg
print("Training score:" ,linear_reg.score(x_train,y_train))
y_pred = linear_reg.predict(x_test)
from sklearn.metrics import r2_score

print ("Testing score:", r2_score(y_test, y_pred))
plt.figure(figsize=(8,8))

plt.scatter(x_test,y_test, c="black")

plt.plot(x_test,y_pred, c="blue", linewidth=2)

plt.xlabel("Economy (GDP per capita)")

plt.ylabel("Hapiness Score")
X = data.drop(["Country","Region","Happiness Rank", "Happiness Score"], axis=1)
X.shape
X.head()
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.3, random_state=0)
x_train.shape
linear_reg = LinearRegression(normalize=True).fit(x_train,y_train)

linear_reg
print("Training score:" ,linear_reg.score(x_train,y_train))
y_pred = linear_reg.predict(x_test)
print ("Testing score:", r2_score(y_test, y_pred))
from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import Lasso



regressor = DecisionTreeRegressor(random_state=0).fit(x_train,y_train)

regressor

Lasso_reg = Lasso(alpha = 0.1).fit(x_train,y_train)

Lasso_reg
print("Training score for Decision Tree Regressor:" ,regressor.score(x_train,y_train))

print("Training score for Lasso Regressor:" ,Lasso_reg.score(x_train,y_train))
y_pred_DT = regressor.predict(x_test)

y_pred_L = Lasso_reg.predict(x_test)
print ("Testing score for Decision Tree Regressor:", r2_score(y_test, y_pred_DT))

print ("Testing score for Lasso Regressor:", r2_score(y_test, y_pred_L))