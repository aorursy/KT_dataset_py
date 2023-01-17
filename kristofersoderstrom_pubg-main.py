# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

import os

print(os.listdir("../input"))

#loading additional dependencies 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #plots and graphs

import seaborn as sns #additional functionality and visualization

from math import sqrt

import random

random.seed(30) #seed for reproducibility

#dependencies for preprocessing and modelling data

from sklearn import preprocessing 

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error,r2_score

import xgboost as xgb

#load data and create dataframe 

train_data = pd.read_csv('../input/train_V2.csv')

#summarize information 

train_data.info()

print("database shape:",train_data.shape)

before = train_data.shape

print("missing data?",train_data.isnull().values.any())

print("deleting missing values...")# dataframe has missing values, we will drop them because of time constraints. Usually not desirable since missing information can actually provide with important insights.

train_data = train_data.dropna()

print("missing data?",train_data.isnull().values.any())

after = train_data.shape

#print("using random sample (1% of data) to speed up computation...")

#train_data = train_data.sample(n=None, frac=0.01, replace=False, weights=None, random_state=None, axis=None)

print("database shape:",train_data.shape)

print("Dropped rows:",before[0]-after[0])

train_data.head()
#developing a heatmap with example from https://seaborn.pydata.org/examples/many_pairwise_correlations.html

sns.set(style="white") # set a style for the graph

corr = train_data.corr() # compute correlation matrix

f, ax = plt.subplots(figsize=(15,15)) #set size

cmap = sns.diverging_palette(220,10,as_cmap=True) #define a custom color palette

sns.heatmap(corr,annot=False,cmap=cmap,square=True,linewidths=0.5) #draw graph

plt.show()
n = 10

f, ax = plt.subplots(figsize=(15,15))

cols = train_data.corr().nlargest(n,"winPlacePerc")["winPlacePerc"].index

corr = np.corrcoef(train_data[cols].values.T)

sns.heatmap(corr, annot=True,cmap=cmap,square=True,linewidths=0.5,xticklabels=cols.values,yticklabels=cols.values)

plt.show()
sns.set_style("white")

f, ax = plt.subplots(figsize=(8,5))

sns.scatterplot(x="walkDistance",y="winPlacePerc",data=train_data)

plt.show()
sns.set_style("white")

f, ax = plt.subplots(figsize=(8,5))

sns.lineplot(x="walkDistance", y="winPlacePerc", data=train_data)

plt.show()
sns.set(style="white")

cols = ['winPlacePerc','walkDistance', 

        'boosts','weaponsAcquired',

        'damageDealt','heals']

sns.pairplot(train_data[cols])

plt.show()
train_data["matchType"].unique() # we can see the different types of matches in the game
#we start by diving the training data between features and targets

x_train = train_data.iloc[:,3:-1]

y_train = train_data.iloc[:,-1].values

print("traning data has the shape:",x_train.shape)

#one hot encoding matchType to include in analysis, it has 16 different types which might reflect specific characteristics between the match types

x_train = pd.get_dummies(x_train, prefix = ["matchType"])

print("x_train shape after one hot encoding",x_train.shape)

print("y_train shape",y_train.shape)



#we will normalize data to facilitate learning

print("normalizing data...")

#x_train = preprocessing.StandardScaler().fit_transform(x_train)

x_train = preprocessing.scale(x_train)



print("validation split to 20%")#

X_train,X_val,y_train,y_val = train_test_split(x_train,y_train,

                                               test_size=0.2,

                                               random_state=30)

print("fitting linear regression...")

reg = LinearRegression()

fit = reg.fit(X_train,y_train)

y_predicted = reg.predict(X_val)

print('Train R2',reg.score(X_train,y_train))

print('Val R2', r2_score(y_val,y_predicted))

print('Val RMSE', sqrt(mean_squared_error(y_val, y_predicted)))
reg = xgb.XGBRegressor()                        

print("fitting xgboost regression")

fit = reg.fit(X_train,y_train)

y_predicted = reg.predict(X_val)

print('Train R2',reg.score(X_train,y_train))

print('Val R2', r2_score(y_val,y_predicted))

print('Val RMSE', sqrt(mean_squared_error(y_val, y_predicted)))
#boosts and heals 

print("correlation between passive items and finishing placement:")

train_data["_passiveItems"] = train_data["boosts"]+train_data["heals"]

print(np.corrcoef(train_data["_passiveItems"],train_data["winPlacePerc"])) #corrcoefficient

#correlation graph

sns.set_style("white")

f, ax = plt.subplots(figsize=(8,5))

sns.scatterplot(x="_passiveItems",y="winPlacePerc",data=train_data,legend="full")

plt.show()
#total distance

print("correlation total distance travelled and finishing placement:")

train_data["_totalDistance"] = train_data["walkDistance"]+train_data["rideDistance"]+train_data["swimDistance"]



print(np.corrcoef(train_data["_totalDistance"],train_data["winPlacePerc"])) #corrcoefficient

#correlation graph

sns.set_style("white")

f, ax = plt.subplots(figsize=(8,5))

sns.lineplot(x="_totalDistance",y="winPlacePerc",data=train_data,legend="full")

plt.show()
train_data["matchId"].describe()
#skipped

#adding matchId

#x_train = train_data.iloc[:,2:-1]

#y_train = train_data.iloc[:,-1].values

#print("traning data has the shape:",x_train.shape)



#x_train = pd.get_dummies(x_train, prefix = ["matchId","matchType"])

#print("x_train shape after one hot encoding", x_train.shape)

#print("y_train shape", y_train.shape)

train_data_feat = train_data.drop(columns=["boosts","heals","walkDistance",

                                           "rideDistance","swimDistance",

                                           "Id","groupId","matchId"])



#once again we construct our x and y sets, one hot encode matchType and normalize the data

x_train = train_data_feat.drop(columns=["winPlacePerc"])

y_train = train_data[["winPlacePerc"]].values

print("traning data has the shape:",x_train.shape)

#one hot encoding matchType to include in analysis

x_train = pd.get_dummies(x_train, prefix = ["matchType"])

print("x_train shape after one hot encoding", x_train.shape)

print("y_train shape", y_train.shape)



#we will normalize data to facilitate learning

print("normalizing data...")

from sklearn import preprocessing 

x_train = preprocessing.scale(x_train)



#training-validation split

X_train,X_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.2,random_state=30)

print("fitting xgboost regression...")



reg = xgb.XGBRegressor()  

fit = reg.fit(X_train,y_train)

y_predicted = reg.predict(X_val)

print('Train R2',reg.score(X_train,y_train))

print('Val R2', r2_score(y_val,y_predicted))

print('Val RMSE', sqrt(mean_squared_error(y_val, y_predicted)))