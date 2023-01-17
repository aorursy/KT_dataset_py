import time

import random

import datetime

import pandas as pd

import matplotlib.pyplot as plt

import statistics

import numpy as np

from scipy import stats

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression

from sklearn.metrics import explained_variance_score, mean_absolute_error,mean_squared_error, median_absolute_error, r2_score

from sklearn.svm import SVR

from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import cross_val_score

import seaborn

from IPython.display import Image
data = pd.read_csv('../input/concrete/Concrete.csv')

data.head()
print(len(data))
data.columns
data.isnull().sum()
#Plotting Scatter plots between the response and exploratory variables

plt.figure(figsize=(15,10.5))

plot_count=1

for feature in list(data.columns)[:-1]:

    plt.subplot(3,3,plot_count)

    plt.scatter(data[feature],data['csMPa'])

    plt.xlabel(feature.replace('_','_').title())

    plt.ylabel('Concrete Strength')

    plot_count+=1

plt.show()
#Calculating pair wise correlation

pd.set_option('display.width',100)

pd.set_option('precision',3)

correlations = data.corr(method='pearson')

print(correlations)
data_ = data[(data.T!=0).any()]

seaborn.pairplot(data_,vars=data.columns,kind='reg')

plt.show()
def split_train_test(data,feature,train_index=0.7):

    train,test = train_test_split(data,test_size=1-train_index)

    if type(feature)==list:

        X_train = train[feature].as_matrix()

        y_train = train['csMPa'].as_matrix()

        X_test  = test[feature].as_matrix()

        y_test  = test['csMPa'].as_matrix()

    else:

        X_train = [[x] for x in list(train[feature])]

        y_train = [[x] for x in list(train['csMPa'])]

        

        X_test  = [[x] for x in list(test[feature])]

        y_test  = [[x] for x in list(test['csMPa'])]

    

    return X_train,y_train, X_test, y_test
#Linear Regression

plt.figure(figsize=(15,7))

plot_count=1



for feature in['cement','slag','flyash','water','superplasticizer','coarseaggregate']:

    data_tr = data[['csMPa',feature]]

    data_tr = data_tr[(data_tr.T!=0).all()]

    X_train,y_train,X_test,y_test = split_train_test(data_tr,feature)

    regr = LinearRegression()

    regr.fit(X_train,y_train)

    y_pred = regr.predict(X_test)

    plt.subplot(2,3,plot_count)

    plt.scatter(X_test,y_test,color='black')

    plt.plot(X_test,y_pred,color='blue',linewidth=3)

    plt.xlabel(feature.replace('_',' ').title())

    plt.ylabel('Concrete strength')

    print(feature,r2_score(y_test,y_pred))

    plot_count+=1

plt.show()

