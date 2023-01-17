# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#load the dataset
dataset = pd.read_csv("/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
#first 5 rows
dataset.head()
#check if there is any null value
dataset.isnull().sum()
#datatypes
dataset.dtypes
#dropping the null values
dataset = dataset.drop(['name', 'host_name', 'last_review', 'reviews_per_month'], axis = 1)
#checking the dataset
dataset.isnull().sum()
#description of the dataset
dataset.describe().T
#different types of rooms
dataset.room_type.unique()
#different neighbourhoods
dataset.neighbourhood.unique()
#number of different neighbourhoods
d = dataset.neighbourhood.unique()
len(d)
#different neighbourhood groups
dataset.neighbourhood_group.unique()
#count plot for neighbourhood grouos
sns.countplot(dataset.neighbourhood_group, data = dataset)
#countplot for types of room
sns.countplot(dataset.room_type, data = dataset)
#room type based on the neighbourhood
sns.countplot(x = 'room_type', hue = 'neighbourhood_group', data = dataset)
#distplot to see price
#distplot for availability
#distplot for minimum nights
df1 = dataset[dataset.price < 500]
df2 = dataset[dataset.availability_365 < 370]
df3 = dataset[dataset.minimum_nights < 200]
f, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
sns.despine(left=True)
sns.distplot(df1.price, ax = axes[0])
sns.distplot(df2.availability_365, ax = axes[1])
sns.distplot(df3.minimum_nights, ax=axes[2])
plt.setp(axes, yticks=[])
plt.tight_layout()
#each neighbourhood group price variation
f, axes = plt.subplots(3,2, figsize=(7, 7), sharex=True)
sns.despine(left=True)
sns.distplot(df1[(df1.neighbourhood_group == 'Brooklyn')]['price'],color='k', axlabel = 'Brooklyn Price', ax = axes[0,0])
sns.distplot(df1[(df1.neighbourhood_group == 'Manhattan')]['price'],color='k', axlabel = 'Manhattan Price', ax = axes[0,1])
sns.distplot(df1[(df1.neighbourhood_group == 'Queens')]['price'],color='k', axlabel = 'Queens Price', ax = axes[1,0])
sns.distplot(df1[(df1.neighbourhood_group == 'Staten Island')]['price'],color='k', axlabel = 'Staten Island Price', ax = axes[1,1])
sns.distplot(df1[(df1.neighbourhood_group == 'Bronx')]['price'],color='k', axlabel = 'Bronx Price', ax = axes[2,0])

#price in each neighbourhood group
sns.set(style="ticks", palette="pastel")
sns.boxplot(x = dataset.neighbourhood_group, y = dataset.price,  data = dataset)
sns.despine(offset=10, trim=True)
#price distribution wrt minimum nights
sns.jointplot(y = dataset.price, x = dataset.minimum_nights)
#price and availability
sns.jointplot(y = dataset.price, x = dataset.availability_365)

#availability wrt price and neighbourhood groups
sns.scatterplot(x = dataset.availability_365, y = dataset.price, hue = dataset.neighbourhood_group, data = dataset)
#price with respect to neighbourhood group
sns.scatterplot(x = dataset.price, y = dataset.neighbourhood_group, data = dataset)
#violin plot for neighbourhood group with price and room type
sns.catplot(x="neighbourhood_group", y="price", hue="room_type",
            kind="violin", data=df1)

#top 5 neighbouhoods
plt.figure(figsize = (6,6))
df4 = dataset.neighbourhood.value_counts().head(5)
sns.barplot(x = df4.index, y = df4.values)
#top 5 host id
plt.figure(figsize = (6,6))
df5 = dataset.host_id.value_counts().head(5)
sns.barplot(x = df5.index, y = df5.values)
#latitude and longitude
plt.figure(figsize = (10,10))
sns.scatterplot(dataset.longitude, dataset.latitude, hue = dataset.neighbourhood_group)
plt.ioff()
#based on room type
plt.figure(figsize = (10,10))
sns.scatterplot(dataset.longitude, dataset.latitude, hue = dataset.room_type)
plt.ioff()
#check the dataset
dataset
#check datatypes
dataset.dtypes
#label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dataset.neighbourhood_group = le.fit_transform(dataset.neighbourhood_group)
dataset.neighbourhood = le.fit_transform(dataset.neighbourhood)
dataset.room_type = le.fit_transform(dataset.room_type)
#check correlation with respect to price
dataset.corr()['price'].sort_values
#visual display of correlation
plt.figure( figsize = (10,10))
sns.heatmap(dataset.corr(), annot = True)
#split the dataset into X and y
X = dataset.drop(['price'], axis = 1).values
y = dataset.price.values
#p value calculation
#backward elimination model used to determine the best features
import statsmodels.api as sm
X = np.append(arr = np.ones((48895,1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0,1,2,3,4,5,6,7,8,9,10,11]]
reg_OLS = sm.OLS(y,X_opt).fit()
reg_OLS.summary()
#removing 8 - minimum_nights
X_opt = X[:, [0,1,2,3,4,5,6,7,9,10,11]]
reg_OLS = sm.OLS(y,X_opt).fit()
reg_OLS.summary()
#removing 1 - host_id
X_opt = X[:, [0,2,3,4,5,6,7,9,10,11]]
reg_OLS = sm.OLS(y,X_opt).fit()
reg_OLS.summary()
#best features
#updating the values of X
X = X_opt[:,1:]
#split into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#Multivariate Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
#predict the value
y_pred = regressor.predict(X_test)
#check metrics
from sklearn.metrics import r2_score, mean_squared_error
r2_score(y_test, y_pred)
#mse value
mean_squared_error(y_test, y_pred)
#SVM
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)
#predict the value
y_pred = regressor.predict(X_test)
#r2 score
r2_score(y_test, y_pred)
#mse value
mean_squared_error(y_test, y_pred)
#random forest 
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 50)
regressor.fit(X_train, y_train)
#predict the value
y_pred = regressor.predict(X_test)
r2_score(y_test, y_pred)
#mse value
mean_squared_error(y_test, y_pred)
#regression plot of random forest
sns.regplot(y = y_test, x = y_pred, color = 'blue')
