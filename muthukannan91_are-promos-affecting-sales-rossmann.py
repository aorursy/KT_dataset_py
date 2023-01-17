# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn import ensemble
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression,ridge_regression
import statsmodels.api as sm
from sklearn.tree import DecisionTreeRegressor
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

givenTrain = pd.read_csv("../input/train.csv",parse_dates = ['Date'])
givenTest = pd.read_csv("../input/test.csv",parse_dates = ['Date'])
givenStore = pd.read_csv("../input/store.csv")
givenTrain[givenTrain.StateHoliday=="0"] = 0 
givenTrain.isnull().sum()
givenTest.isnull().sum()
givenStore.isnull().sum()
givenTrain[(givenTrain['Open'] == 0) & (givenTrain['Sales'] != 0)]
openStores = givenTrain[givenTrain['Open'] == 1]

openStores.drop(columns='Date',axis=1,inplace =True)
openStores = pd.get_dummies(openStores)
storeData = pd.get_dummies(givenStore)
storeData.dtypes
mergeData = pd.merge(openStores, storeData, how="inner")
mergeData.isna().sum()
cleanData= mergeData.drop(mergeData[['CompetitionDistance','CompetitionOpenSinceMonth','CompetitionOpenSinceYear','Promo2SinceWeek','Promo2SinceYear']],axis= 1)
cleanData.isna().sum()
X = cleanData.drop(cleanData[['Sales', 'Store','Open']],axis = 1) # Store and Open have no explanatory power and sales is the target variable.
y = cleanData['Sales'] 
corr = cleanData.corr()
sns.heatmap(corr)
X1 = cleanData.drop(cleanData[['Sales', 'Store','Open','PromoInterval_Feb,May,Aug,Nov','StoreType_a','Assortment_a','StateHoliday_a']],axis = 1)
y = cleanData['Sales']
X2 = sm.add_constant(X1)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())
def unRestrictedModel(cleanData):
    predictiveVariables = cleanData.drop(cleanData[['Sales', 'Store','Open','PromoInterval_Feb,May,Aug,Nov','StoreType_a','Assortment_a','StateHoliday_a']],axis = 1)
    toPredict = cleanData['Sales']
    X_train, X_test, y_train, y_test = train_test_split(predictiveVariables, toPredict, test_size=0.33, random_state=42)
    lm = LinearRegression()
    lfit = lm.fit(X_train,y_train)
    yPredict = lfit.predict(X_test)
    return np.sqrt(mean_absolute_error(y_test,yPredict))
#unRestrictedModel(cleanData)
    
def RestrictedModel(cleanData): #Dropping promo
    predictiveVariables = cleanData.drop(cleanData[['Promo','Sales', 'Store','Open','PromoInterval_Feb,May,Aug,Nov','StoreType_a','Assortment_a','StateHoliday_a']],axis = 1)
    toPredict = cleanData['Sales']
    X_train, X_test, y_train, y_test = train_test_split(predictiveVariables, toPredict, test_size=0.33, random_state=42)
    lm = LinearRegression()
    lfit = lm.fit(X_train,y_train)
    yPredict = lfit.predict(X_test)
    return np.sqrt(mean_absolute_error(y_test,yPredict))
#RestrictedModel(cleanData)
randomNess = 0
for i in range(1,101):
    f=(i/100)
    df =cleanData.sample(frac = f ,replace=True)
    errorFullModel = unRestrictedModel(df)
    errorNonPromoModel = RestrictedModel(df)
    if errorFullModel > errorNonPromoModel:
        randomNess = randomNess + 1
le= len(cleanData)
print(randomNess)
pVal = randomNess/100
pVal
# Gradient Boosting Regressor with promos included in explanatory variables
X = cleanData.drop(cleanData[['Sales', 'Store','Open','PromoInterval_Feb,May,Aug,Nov','StoreType_a','Assortment_a','StateHoliday_a']],axis = 1)
Y = cleanData['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.02, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)
clf.fit(X_train, y_train)
mae = mean_absolute_error(y_test, clf.predict(X_test))
print("Mean Absolute Error with Promos included: %.4f" % mae)

# Gradient Boosting Regressor after removing promos in explanatory variables
X = cleanData.drop(cleanData[['Promo','Sales', 'Store','Open','PromoInterval_Feb,May,Aug,Nov','StoreType_a','Assortment_a','StateHoliday_a']],axis = 1)
Y = cleanData['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.02, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)
clf.fit(X_train, y_train)
mae = mean_absolute_error(y_test, clf.predict(X_test))
print("Mean Absolute Error without Promos included: %.4f" % mae)
rf = RandomForestRegressor(n_estimators=100)
X = cleanData.drop(cleanData[['Sales', 'Store','Open','PromoInterval_Feb,May,Aug,Nov','StoreType_a','Assortment_a','StateHoliday_a']],axis = 1)
Y = cleanData['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)
rf.fit(X,y)
mae = mean_absolute_error(y_test, rf.predict(X_test))
print("MAE with Promos included: %.4f" % mae)

rf1 = RandomForestRegressor(n_estimators=100)
X = cleanData.drop(cleanData[['Promo','Sales', 'Store','Open','PromoInterval_Feb,May,Aug,Nov','StoreType_a','Assortment_a','StateHoliday_a']],axis = 1)
Y = cleanData['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)
rf1.fit(X,y)
mae1 = mean_absolute_error(y_test, rf1.predict(X_test))
print("MAE without Promos included: %.4f" % mae1)