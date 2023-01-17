import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns 

import re

import time

import warnings

import sqlite3

from sqlalchemy import create_engine # database connection

import csv

import os

warnings.filterwarnings("ignore")

import datetime as dt

import numpy as np

from nltk.corpus import stopwords

from sklearn.decomposition import TruncatedSVD

from sklearn.preprocessing import normalize

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.manifold import TSNE

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics.classification import accuracy_score, log_loss

from sklearn.feature_extraction.text import TfidfVectorizer

from collections import Counter

from scipy.sparse import hstack

from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import SVC

from collections import Counter, defaultdict

from sklearn.calibration import CalibratedClassifierCV

from sklearn.naive_bayes import MultinomialNB

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

import math

from sklearn.metrics import normalized_mutual_info_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import SGDClassifier

from mlxtend.classifier import StackingClassifier

from sklearn import model_selection

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import precision_recall_curve, auc, roc_curve

import os

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

print(train.head())

print('**'* 50)

print(test.head())
print(train.info())

print('**'* 50)

print(test.info())
n_train = train.shape[0]

n_test = test.shape[0]

y = train['SalePrice'].values

print(train['SalePrice'].value_counts())

#print(y.value_counts())

data = pd.concat((train, test)).reset_index(drop=True)

data.drop(['SalePrice'], axis=1, inplace=True)
print(data.head())

print(data.shape)
plt.figure(figsize=(30,10))

sns.heatmap(train.corr(),cmap='coolwarm',annot = True)

plt.show()

sns.pairplot(train, palette='rainbow')
counts, bin_edges = np.histogram(train['YearBuilt'], bins=10, density = True)

pdf = counts/(sum(counts))

print(pdf);

print(bin_edges);

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf);

plt.plot(bin_edges[1:], cdf)

plt.show();
sns.lmplot(x='YearBuilt',y='SalePrice',data=train)

counts, bin_edges = np.histogram(train['YearBuilt'], bins=10, 

                                 density = True)

pdf = counts/(sum(counts))

print(pdf);

print(bin_edges);

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf);

plt.plot(bin_edges[1:], cdf)









plt.show();
plt.figure(figsize=(16,8))

sns.boxplot(x='GarageCars',y='SalePrice',data=train)

plt.show()
counts, bin_edges = np.histogram(train['GarageCars'], bins=10, 

                                 density = True)

pdf = counts/(sum(counts))

print(pdf);

print(bin_edges);

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf);

plt.plot(bin_edges[1:], cdf)

plt.show();
plt.figure(figsize=(16,8))

sns.barplot(x='GarageArea',y = 'SalePrice',data=train, estimator=np.mean)

plt.show()
counts, bin_edges = np.histogram(train['GarageArea'], bins=10, 

                                 density = True)

pdf = counts/(sum(counts))

print(pdf);

print(bin_edges);

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf);

plt.plot(bin_edges[1:], cdf)

plt.show();
plt.figure(figsize=(16,8))

sns.barplot(x='FullBath',y = 'SalePrice',data=train)

plt.show()
counts, bin_edges = np.histogram(train['FullBath'], bins=10, 

                                 density = True)

pdf = counts/(sum(counts))

print(pdf);

print(bin_edges);

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf);

plt.plot(bin_edges[1:], cdf)

plt.show();
sns.lmplot(x='1stFlrSF',y='SalePrice',data=train)



sns.boxplot(x='1stFlrSF',y='SalePrice',data=train)
data = data[['LotArea','Street', 'Neighborhood','Condition1', 'Condition2','BldgType','HouseStyle','OverallCond', 'Heating','CentralAir','Electrical','1stFlrSF','2ndFlrSF','BsmtHalfBath','FullBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','GarageCars','GarageArea','PoolArea']]
data.info()
data['BsmtHalfBath'] = data['BsmtHalfBath'].fillna(data['BsmtHalfBath'].mean())

data['GarageCars'] = data['GarageCars'].fillna(data['GarageCars'].mean())

data['GarageArea'] = data['GarageArea'].fillna(data['GarageArea'].mean())

#data['Electrical']=data['Electrical'].fillna(' ')



# Categorical boolean mask

categorical_feature_mask = data.dtypes==object

# filter categorical columns using mask and turn it into alist

categorical_cols = data.columns[categorical_feature_mask].tolist()

print(categorical_cols)

print("number of categorical features ",len(categorical_cols))



# i in range(len(categorical_cols)):

 # data[i]=data[i].fillna(' ')

#data['Electrical'] = data['Electrical'].fillna(' ')



data = pd.get_dummies(data, columns=categorical_cols)
data.info()
data.shape
train =data[:n_train]

test = data[n_train:]

print(train.info())

print(test.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.1, random_state=101)
# we are going to scale to data



y_train= y_train.reshape(-1,1)

y_test= y_test.reshape(-1,1)

print(X_train.info())

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

sc_y = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.fit_transform(X_test)

y_train = sc_X.fit_transform(y_train)

y_test = sc_y.fit_transform(y_test)
from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train,y_train)

print(lm)
# This function plots the confusion matrices given y_i, y_i_hat.

def plot_confusion_matrix(test_y, predict_y):

    C = confusion_matrix(test_y, predict_y)

    # C = 9,9 matrix, each cell (i,j) represents number of points of class i are predicted class j

    

    A =(((C.T)/(C.sum(axis=1))).T)

    #divid each element of the confusion matrix with the sum of elements in that column

    

    # C = [[1, 2],

    #     [3, 4]]

    # C.T = [[1, 3],

    #        [2, 4]]

    # C.sum(axis = 1)  axis=0 corresonds to columns and axis=1 corresponds to rows in two diamensional array

    # C.sum(axix =1) = [[3, 7]]

    # ((C.T)/(C.sum(axis=1))) = [[1/3, 3/7]

    #                           [2/3, 4/7]]



    # ((C.T)/(C.sum(axis=1))).T = [[1/3, 2/3]

    #                           [3/7, 4/7]]

    # sum of row elements = 1

    

    B =(C/C.sum(axis=0))

    #divid each element of the confusion matrix with the sum of elements in that row

    # C = [[1, 2],

    #     [3, 4]]

    # C.sum(axis = 0)  axis=0 corresonds to columns and axis=1 corresponds to rows in two diamensional array

    # C.sum(axix =0) = [[4, 6]]

    # (C/C.sum(axis=0)) = [[1/4, 2/6],

    #                      [3/4, 4/6]] 

    plt.figure(figsize=(20,4))

    

    labels = [1,2]

    # representing A in heatmap format

    cmap=sns.light_palette("blue")

    plt.subplot(1, 3, 1)

    sns.heatmap(C, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predicted Class')

    plt.ylabel('Original Class')

    plt.title("Confusion matrix")

    

    plt.subplot(1, 3, 2)

    sns.heatmap(B, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predicted Class')

    plt.ylabel('Original Class')

    plt.title("Precision matrix")

    

    plt.subplot(1, 3, 3)

    # representing B in heatmap format

    sns.heatmap(A, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predicted Class')

    plt.ylabel('Original Class')

    plt.title("Recall matrix")

    

    plt.show()
# print the intercept

print(lm.intercept_)
print(lm.coef_)

predictions = lm.predict(X_test)

predictions= predictions.reshape(-1,1)

#plot_confusion_matrix(y_test, predictions)
plt.figure(figsize=(15,8))

plt.scatter(y_test,predictions)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')

plt.show()

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

#print(log_loss(y_test, predictions))

from sklearn import ensemble

from sklearn.utils import shuffle

from sklearn.metrics import mean_squared_error, r2_score
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,

          'learning_rate': 0.01, 'loss': 'ls'}

clf = ensemble.GradientBoostingRegressor(**params)



clf.fit(X_train, y_train)
clf_pred=clf.predict(X_test)

clf_pred= clf_pred.reshape(-1,1)
print('MAE:', metrics.mean_absolute_error(y_test, clf_pred))

print('MSE:', metrics.mean_squared_error(y_test, clf_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, clf_pred)))
plt.figure(figsize=(15,8))

plt.scatter(y_test,clf_pred, c= 'brown')

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')

plt.show()

plt.plot(y_test,clf_pred, c= 'blue')

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')

plt.show()
from sklearn.tree import DecisionTreeRegressor

dtreg = DecisionTreeRegressor(random_state = 100)

dtreg.fit(X_train, y_train)

dtr_pred = dtreg.predict(X_test)

dtr_pred= dtr_pred.reshape(-1,1)
print('MAE:', metrics.mean_absolute_error(y_test, dtr_pred))

print('MSE:', metrics.mean_squared_error(y_test, dtr_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, dtr_pred)))
plt.figure(figsize=(15,8))

plt.scatter(y_test,dtr_pred,c='green')

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')

plt.show()

plt.plot(y_test,clf_pred, c= 'blue')

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')

plt.show()
from sklearn.svm import SVR

svr = SVR(kernel = 'rbf')

svr.fit(X_train, y_train)

svr_pred = svr.predict(X_test)

svr_pred= svr_pred.reshape(-1,1)
print('MAE:', metrics.mean_absolute_error(y_test, svr_pred))

print('MSE:', metrics.mean_squared_error(y_test, svr_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, svr_pred)))
plt.figure(figsize=(15,8))

plt.scatter(y_test,svr_pred, c='red')

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')

plt.show()

plt.plot(y_test,clf_pred, c= 'blue')

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')

plt.show()
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators = 1500, random_state = 0)

rfr.fit(X_train, y_train)

rfr_pred= rfr.predict(X_test)

rfr_pred = rfr_pred.reshape(-1,1)
print('MAE:', metrics.mean_absolute_error(y_test, rfr_pred))

print('MSE:', metrics.mean_squared_error(y_test, rfr_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, rfr_pred)))
plt.figure(figsize=(15,8))

plt.scatter(y_test,rfr_pred, c='orange')

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')

plt.show()

plt.plot(y_test,clf_pred, c= 'blue')

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')

plt.show()
error_rate=np.array([metrics.mean_squared_error(y_test, predictions),metrics.mean_squared_error(y_test, clf_pred),metrics.mean_squared_error(y_test, dtr_pred),metrics.mean_squared_error(y_test, svr_pred),metrics.mean_squared_error(y_test, rfr_pred)])





print(min(metrics.mean_squared_error(y_test, predictions),min(metrics.mean_squared_error(y_test, clf_pred),min(metrics.mean_squared_error(y_test, svr_pred),metrics.mean_squared_error(y_test, dtr_pred)))))
plt.figure(figsize=(16,5))

print(error_rate)

plt.plot(error_rate)

plt.scatter(error_rate,range(1,6))

seed = 7

# prepare models

models = ['SVM','RANDOM_FOREST','LR','BGT','DT']



a = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
test_id = a['Id']

print(test_id.shape)

#making dataframe 

a = pd.DataFrame(test_id, columns=['Id'])
test = sc_X.fit_transform(test)
test.shape
test_prediction_svr=svr.predict(test)

test_prediction_svr= test_prediction_svr.reshape(-1,1)
test_prediction_svr
test_prediction_svr =sc_y.inverse_transform(test_prediction_svr)

test_prediction_svr
test_pred_svr = pd.DataFrame(test_prediction_svr, columns=['SalePrice'])

#test_pred_svr
test_pred_svr.head()
result = pd.concat([a,test_pred_svr], axis=1)
result.head()
result.to_csv('../submission.csv',index=False)