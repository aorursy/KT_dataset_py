import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
import matplotlib.pyplot as plt
from itertools import cycle
from scipy import interp

import os

#####
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model # lasso, ridge
from sklearn import tree

import seaborn as sns
train = pd.read_csv("../input/train.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")

test = pd.read_csv("../input/test.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
train.shape
train.head()
train.describe()
# 1o - renda per capta media
r = pd.DataFrame({'rpc': train['median_income']/train['population']})

trainf= train.join(r)
corr = trainf.corr()

# plot the heatmap
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)
train_b = trainf.drop(columns = ['median_age', 'total_bedrooms', 'population', 'households'])
test_b1 = test.drop(columns = ['median_age', 'total_bedrooms', 'population', 'households'])
train_b.head()
plt.figure(1,figsize = (15,3))
plt.subplot(161)
train_b['longitude'].hist(bins = 10).plot()
plt.title('longitude')
plt.subplot(162)
train_b['latitude'].hist(bins = 10).plot()
plt.title('latitude')
plt.subplot(163)
train_b['total_rooms'].hist(bins = 10).plot()
plt.title('total_rooms')
plt.subplot(164)
train_b['median_income'].hist(bins = 10).plot()
plt.title('median_income')
plt.subplot(165)
train_b['rpc'].hist(bins = 3).plot()
plt.title('rpc')
plt.subplot(166)
train_b['median_house_value'].hist(bins = 10).plot()
plt.title('median_house_value')

# Divisão da base de treino em features escolhidas e labels
Ftrain = train.drop(columns = ['median_house_value', 'Id'])
Ltrain = train["median_house_value"]

Xtest = test.drop(columns = ['Id'])
# knn regressor
knnr = KNeighborsRegressor(n_neighbors=3)
scores = cross_val_score(knnr, Ftrain, Ltrain, cv=10)
scores.mean()
#Lasso
lasso = linear_model.Lasso(alpha = 0.1)
scores = cross_val_score(lasso, Ftrain, Ltrain, cv=10)
scores.mean()
#Ridge
ridge = linear_model.Ridge(alpha=1.0)
scores = cross_val_score(ridge, Ftrain, Ltrain, cv=10)
scores.mean()
#Decision tree
tr = tree.DecisionTreeRegressor(random_state=0)
scores = cross_val_score(tr, Ftrain, Ltrain, cv=10)
scores.mean()
Ftrain2 = train_b.drop(columns = ['Id', 'median_house_value'])
Ltrain2 = train_b['median_house_value']

#teste
r = pd.DataFrame({'rpc': test['median_income']/test['population']})

test_b = test_b1.join(r)
Xtest_b = test_b.drop(columns = ['Id'])
# knn regressor
knnr = KNeighborsRegressor(n_neighbors=3)
scores = cross_val_score(knnr, Ftrain2, Ltrain2, cv=10)
scores.mean()
#Lasso
lasso = linear_model.Lasso(alpha = 0.1)
scores = cross_val_score(lasso, Ftrain2, Ltrain2, cv=10)
scores.mean()
#Ridge
ridge = linear_model.Ridge(alpha=1.0)
scores = cross_val_score(ridge, Ftrain2, Ltrain2, cv=10)
scores.mean()
#Decision tree
tr = tree.DecisionTreeRegressor(random_state=0)
scores = cross_val_score(tr, Ftrain2, Ltrain2, cv=10)
scores.mean()
pred = tr.fit(Ftrain2, Ltrain2).predict(Xtest_b)
pred.shape
predf = []
for i in range(0,6192):
    predf.append([test['Id'][i],pred[i]])
    
for i in range(0,5):
    print(predf[i])
out = pd.DataFrame(predf, columns = ['Id', 'median_house_value'])
out.to_csv("pred.csv", index = False)
out