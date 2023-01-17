import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)
import os

print(os.listdir("../input"))
df = pd.read_csv('../input/preprocessed-house-price-pred-adv-regg-tech/preprocessed_houseprice_dataframe.csv')

df.head(3)
df.shape
column_names = df.columns

print(column_names)
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

print(train.shape)

print(test.shape)
num_features = ['MSSubClass','LotFrontage','LotArea','OverallQual','OverallCond','MasVnrArea','BsmtFinSF1','BsmtFinSF2',

                'BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath',

                'FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars',

                'GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal']
plt.figure(figsize = (25,20))

for i in range(len(num_features)):

    plt.subplot(8,4,i+1)

    ax = sns.distplot(df[num_features[i]])

    ax.legend(["skewness:{:.2f}".format(df[num_features[i]].skew())], fontsize = "xx-large")
for i in num_features:

    df[i] = np.log(df[i]+1)
plt.figure(figsize = (25,20))

for i in range(len(num_features)):

    plt.subplot(8,4,i+1)

    ax = sns.distplot(df[num_features[i]])

    ax.legend(["skewness:{:.2f}".format(df[num_features[i]].skew())], fontsize = "xx-large")
SalePrice = np.log(train['SalePrice']+1)
X_train = df[:len(train)]

X_test = df[len(train):]

y_train = SalePrice

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(X_train)
X_train_sc = sc.transform(X_train)

X_test_sc = sc.transform(X_test)
X_train_sc
X_train_sc =  pd.DataFrame(X_train_sc, columns = column_names)

X_test_sc =  pd.DataFrame(X_test_sc, columns = column_names)
X_train_sc.head(3)
from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import make_scorer, r2_score
def test_model(model, X_train = X_train, y_train = y_train):

    cv = KFold(n_splits = 3, shuffle = True, random_state = 45)

    r2 = make_scorer(r2_score)

    r2_val_score = cross_val_score(model, X_train, y_train, cv = cv, scoring = r2)

    score = [r2_val_score.mean()]

    return score
#scaled function

def test_model_sc(model, X_train = X_train_sc, y_train = y_train):

    cv = KFold(n_splits = 3, shuffle = True, random_state = 45)

    r2 = make_scorer(r2_score)

    r2_val_score = cross_val_score(model, X_train, y_train, cv = cv, scoring = r2)

    score_sc = [r2_val_score.mean()]

    return score_sc
import sklearn.linear_model as linear_model

LR = linear_model.LinearRegression()

test_model(LR)
test_model_sc(LR)
#ridge

LRRidge = linear_model.Ridge()

test_model(LRRidge)
test_model_sc(LRRidge)
#lasso

LRLasso = linear_model.Lasso()

test_model(LRLasso)
test_model_sc(LRLasso)
#svm

from sklearn.svm import SVR

svr_reg = SVR(kernel = 'rbf')

test_model(svr_reg)
test_model_sc(svr_reg)
#decision tree

from sklearn.tree import DecisionTreeRegressor

dt_reg = DecisionTreeRegressor()

test_model(dt_reg)
test_model_sc(dt_reg)
#random forest

from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators = 1000)

test_model(rf_reg)
test_model_sc(rf_reg)
#bagging and boosting

from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor

br_reg = BaggingRegressor(n_estimators = 1000)

gbr_reg = GradientBoostingRegressor(n_estimators = 1000, learning_rate = 0.1, loss = 'ls')
test_model(br_reg)
test_model(gbr_reg)
#xgboost

import xgboost

xgb_reg = xgboost.XGBRegressor()

test_model(xgb_reg)
test_model_sc(xgb_reg)
LRRidge.fit(X_train, y_train)

y_pred = np.exp(LRRidge.predict(X_test)).round(2)

y_pred
submit_test2 = pd.concat([test['Id'], pd.DataFrame(y_pred)], axis = 1)

submit_test2.columns = ['Id', 'SalePrice']

submit_test2.shape
submit_test2.to_csv('submit_test2.csv', index = False)
#checking correlation

plt.figure(figsize = (16,5))

ax = sns.barplot(train.corrwith(train.SalePrice).index, train.corrwith(train.SalePrice))

ax.tick_params(labelrotation = 90)