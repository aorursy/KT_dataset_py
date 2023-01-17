import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import accuracy_score, cohen_kappa_score  #Classification metrics

from sklearn.metrics import mean_squared_error, r2_score #Regression metrics



from sklearn.ensemble import RandomForestRegressor , RandomForestClassifier

from sklearn.preprocessing import StandardScaler



from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

import os

diamond = pd.read_csv("../input/diamonds.csv")
diamond.info()
diamond.head()
diamond=diamond.drop('Unnamed: 0',axis=1)
diamond.head()
diamond.isnull().sum()
sns.distplot(diamond.carat , color = 'red')
sns.distplot(diamond.depth , color = 'green')
sns.distplot(diamond.table , color = 'yellow')
sns.distplot(diamond.price , color = 'blue')
plt.figure(1,figsize=[10,7])

plt.subplot(221)

sns.distplot(diamond.x)



plt.subplot(222)

sns.distplot(diamond.y)



plt.subplot(223)

sns.distplot(diamond.z)



plt.show()
diamond.cut.unique()  # count 5
diamond.color.unique()   # count 7
diamond.clarity.unique()   # count 8
diamond.head()
plt.scatter(diamond.carat , diamond.price)

plt.xlabel('Carat')

plt.ylabel('Price')

plt.show()
plt.scatter(diamond.depth , diamond.price)

plt.xlabel('Depth')

plt.ylabel('Price')

plt.show()
plt.scatter(diamond.table , diamond.price)

plt.xlabel('Table')

plt.ylabel('Price')

plt.show()
plt.figure(1,figsize=[10,7])

plt.subplot(221)

plt.scatter(diamond.x , diamond.price)

plt.xlabel('x')

plt.ylabel('Price')



plt.subplot(222)

plt.scatter(diamond.y , diamond.price)

plt.xlabel('y')

plt.ylabel('Price')



plt.subplot(223)

plt.scatter(diamond.z , diamond.price)

plt.xlabel('z')

plt.ylabel('Price')

plt.show()
sns.boxplot(diamond.cut , diamond.price)
diamond.groupby('cut').price.median()
diamond.groupby('cut').price.mean()
sns.boxplot(diamond.color , diamond.price)

plt.show()

# color vs price
sns.boxplot(diamond.clarity , diamond.price)

plt.show()
plt.figure(figsize=[7,6])

sns.heatmap(diamond.corr() , annot = True)

plt.show()
diamond.isnull().sum()
X=diamond.drop('price',axis = 1)

y=diamond.price
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
dummyXtrain = pd.get_dummies(X_train)

dummyXtest = pd.get_dummies(X_test)
dummyXtrain.head()
lr = LinearRegression()
lr.fit(dummyXtrain,y_train)
y_pred = lr.predict(dummyXtest)
mean_squared_error(y_test,y_pred)
lr.score(dummyXtest,y_test)
rfr = RandomForestRegressor()
rfr.fit(dummyXtrain,y_train)
y_pred = rfr.predict(dummyXtest)
rfr.score(dummyXtest,y_test)   #r2_score
mean_squared_error(y_test,y_pred)
diamond.head()
X=diamond.drop('cut',axis=1)
y=diamond.cut
diamond.cut.unique()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=100)
dummyXtrain = pd.get_dummies(X_train)

dummyXtest = pd.get_dummies(X_test)
knn = KNeighborsClassifier()
knn.fit(dummyXtrain,y_train)
y_pred = knn.predict(dummyXtest)
accuracy_score(y_test,y_pred)
dtree = DecisionTreeClassifier()
dtree.fit(dummyXtrain,y_train)
y_pred = dtree.predict(dummyXtest)
accuracy_score(y_test,y_pred)
rfr = RandomForestClassifier()
rfr.fit(dummyXtrain,y_train)
y_pred = rfr.predict(dummyXtest)
accuracy_score(y_test,y_pred)