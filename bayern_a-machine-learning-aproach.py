# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")

number_of_train_examples = len(train_data)
y = train_data["SalePrice"]#label

#we have to concat train and test and making common preproccesing
dataset = pd.concat(objs = [train_data,test_data],axis = 0)
dataset = dataset.drop(["SalePrice","Id"], axis=1)
dataset.shape
#Data cleaning.
dataset.isnull().mean()
#Maybe delete some features with a lot of misssing (>=80% missing data)
dataset = dataset[dataset.columns[dataset.isnull().mean() < 0.8]]
#Fix the missing data of X
dataset = dataset.interpolate()                              #numerical missing data
dataset = dataset.fillna("Missing")                          #categorial missing data

dataset = pd.get_dummies(dataset)
dataset.shape
X , X_test = dataset[:number_of_train_examples] , dataset[number_of_train_examples:]
X.shape

from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectFromModel

lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X)
X_new_test = model.transform(X_test)#make the same for test data
X = pd.DataFrame(X_new)
X_test = pd.DataFrame(X_new_test)
print(X.shape)
print(X_test.shape)
X = X.drop(X.columns[((X==0).mean()>0.8)],axis=1)
X_test = X_test.drop(X_test.columns[((X_test==0).mean()>0.8)],axis=1)#make the same for test data
print(X.shape)
print(X_test.shape)
from sklearn import preprocessing
x = X.values #returns a numpy array
xt = X_test.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
xt_scaled = min_max_scaler.fit_transform(xt)#make the same for test data
X = pd.DataFrame(x_scaled)
X_test = pd.DataFrame(xt_scaled)
#set values to some parameters
#d = how polynomial we want to be our regression model, so we try some values of degree d
d = [2,3,4,5]
#l = how slow or fast we want learn our algorithm, we define some values of learning rate
l = [0.0000001,0.0000005,0.000001,0.000005,0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.02,0.04,0.08,0.16,0.32,0.64,1.24,2.48,5.12,10.24]
from sklearn.model_selection import train_test_split
X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.3, random_state=42)
X_cv.shape
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

error_train = []
error_cv = []
for degree in d:    
    clf = make_pipeline(PolynomialFeatures(degree), SGDRegressor(max_iter=150))
    clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)
    y_cv_pred = clf.predict(X_cv)
    error_train.append(mean_squared_error(y_train_pred, y_train))
    error_cv.append(mean_squared_error(y_cv_pred, y_cv))

import matplotlib.pyplot as plt
ax = plt.axes(yscale='log')
ax.grid();
plt.plot(d, error_train, label='train error')
plt.plot(d, error_cv, label='cross validation error')
plt.ylabel('error')
plt.xlabel('degree')
plt.legend(loc = 'best')
plt.show()
from sklearn.metrics import mean_squared_error

error_train = []
error_cv = []
best_degree = 5
polClf = PolynomialFeatures(best_degree)
X_new_train = polClf.fit_transform(X_train,y_train)
X_new_cv = polClf.fit_transform(X_cv,y_cv)
for learn_rate in l:
    clf = SGDRegressor(alpha=learn_rate, max_iter=150)
    clf.fit(X_new_train, y_train)
    y_train_pred = clf.predict(X_new_train)
    y_cv_pred = clf.predict(X_new_cv)
    error_train.append(mean_squared_error(y_train_pred, y_train))
    error_cv.append(mean_squared_error(y_cv_pred, y_cv))
    
ax = plt.axes(xscale='log',yscale='log')
ax.grid();
plt.plot(l, error_train , label='train error')
plt.plot(l, error_cv, label='cross validation error')
plt.ylabel('error')
plt.xlabel('learning rate')
plt.legend(loc = 'best')
plt.show()
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

degree = 5
learning_rate = 0.0001

model = make_pipeline(PolynomialFeatures(degree), SGDRegressor(alpha=learning_rate,max_iter=150))
model.fit(X, y)
guess = model.predict(X_test)

# my_submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': guess})
# my_submission.to_csv('submission.csv', index=False)