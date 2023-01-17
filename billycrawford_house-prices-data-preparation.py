import numpy as np
import pandas as pd 
import os
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn import linear_model
from sklearn import metrics
from sklearn import kernel_ridge
from sklearn import ensemble
from sklearn import neighbors
from tensorflow import keras
import tensorflow as tf
sns.set(style="darkgrid")
train_df = pd.read_csv('../input/train.csv')
train_df.head()
test_df = pd.read_csv('../input/test.csv')
test_df.head()
train_df = train_df.rename(index=str, columns={"1stFlrSF": "FirstFlrSF", "2ndFlrSF": "SecondFlrSF"})
test_df = test_df.rename(index=str, columns={"1stFlrSF": "FirstFlrSF", "2ndFlrSF": "SecondFlrSF"})
train_df.describe()
corr = train_df.corr()
mask = np.zeros_like(corr, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True
sns.set(style="white")

plt.figure(figsize = (20,12))
sns.heatmap(corr, mask = mask);
test_df['SalePrice'] = np.nan

dat = pd.concat([train_df,test_df])
dat.head()
dat.dtypes
dat.isna().sum()[dat.isna().sum()>0]
list1 = list(dat.select_dtypes('number').columns)
list2 = list(dat.columns[dat.isna().sum()>0])
numeric_missing = list(set(list1).intersection(list2))
character_missing = list(set(list2).difference(list1))
for col in character_missing:
    dat[col] = dat[col].astype(str)
numeric_missing

dat[numeric_missing].isna().sum()/len(dat)
numeric_missing = [x for x in numeric_missing if x !='LotFrontage' and x != 'SalePrice']
for x in numeric_missing:
    na_indices = dat[x].isna()
    dat[x].loc[na_indices] = np.mean(dat[x].loc[~na_indices])

plt.figure(figsize = (10,8))
ax = plt.scatter(train_df['LotFrontage'],train_df['SalePrice'])
indices = ~train_df['LotFrontage'].isna()
model = smf.ols(formula = 'SalePrice~LotFrontage', data =train_df.loc[indices])
results = model.fit()
print(results.summary())
dat['MSSubClass'] = 'class_' + dat['MSSubClass'].astype(str)

dat_dummies = pd.get_dummies(dat)
indices = ~dat_dummies['LotFrontage'].isna()
dat_dummies.head()
train_df = dat_dummies.loc[~dat['LotFrontage'].isna()].drop(['SalePrice'], axis = 1)
test_df = dat_dummies.loc[dat['LotFrontage'].isna()].drop(['SalePrice'], axis = 1)

train, val = ms.train_test_split(train_df, test_size=0.2)

X_train = np.array(train.drop(['LotFrontage'], axis = 1))
X_val = np.array(val.drop(['LotFrontage'], axis = 1))
X_test = np.array(test_df.drop(['LotFrontage'], axis = 1))

Y_train = np.array(train['LotFrontage'])
Y_val = np.array(val['LotFrontage'])
model = linear_model.LinearRegression()
model.fit(X_train,Y_train)

np.sqrt(metrics.mean_squared_error(Y_val,model.predict(X_val)))
model = linear_model.RidgeCV()
model.fit(X_train,Y_train)

np.sqrt(metrics.mean_squared_error(Y_val,model.predict(X_val)))
model = kernel_ridge.KernelRidge()
model.fit(X_train,Y_train)

np.sqrt(metrics.mean_squared_error(Y_val,model.predict(X_val)))
model = linear_model.LassoCV(cv = 5)
model.fit(X_train,Y_train)

np.sqrt(metrics.mean_squared_error(Y_val,model.predict(X_val)))
model = ensemble.RandomForestRegressor(n_estimators = 50)
model.fit(X_train,Y_train)

np.sqrt(metrics.mean_squared_error(Y_val,model.predict(X_val)))
param_grid = {
                 'n_estimators': [5, 10, 15, 20, 50],
                 'max_depth': [2, 5, 7, 9]
             }

model = ensemble.RandomForestRegressor()
grid_search = ms.GridSearchCV(model,param_grid, cv = 5)
grid_search.fit(X_train,Y_train)

np.sqrt(metrics.mean_squared_error(Y_val,grid_search.predict(X_val)))
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}

model = ensemble.GradientBoostingRegressor(**params)
model.fit(X_train,Y_train)

np.sqrt(metrics.mean_squared_error(Y_val,model.predict(X_val)))
model = linear_model.RidgeCV()
model.fit(X_train,Y_train)

dat_dummies.loc[dat['LotFrontage'].isna(),'LotFrontage'] = model.predict(X_test)
dat_dummies['LotFrontage'] = dat_dummies['LotFrontage'].astype(float)
dat_dummies.isna().sum()[dat_dummies.isna().sum()>0]
dat_dummies.loc[~dat_dummies['SalePrice'].isna(),'SalePrice'] = np.log(dat_dummies.loc[~dat_dummies['SalePrice'].isna(),'SalePrice'])
dat_dummies.to_csv('all_data.csv', index = False)