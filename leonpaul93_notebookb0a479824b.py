import pandas as pd

import os

import time

import pickle

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error

from scipy import stats

from random import seed

import datetime

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

print('Library Import done...')
df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")

print("the training dataset has " +str(df_train.shape[0])+" rows and "+str(df_train.shape[1])+" columns")

print("the testing dataset has " +str(df_test.shape[0])+" rows and "+str(df_test.shape[1])+" columns")
df_train.head(10)
target_variable = df_train['SalePrice']

target_variable.shape[0]
target_variable.describe()
sns.distplot(target_variable)

#plt.savefig('SalePrice_skewed.png', dpi=300, bbox_inches='tight')
print("Skewness: %f" % target_variable.skew())

print("Kurtosis: %f" % target_variable.kurt())
sns.distplot(np.log(target_variable))

plt.savefig('SalePrice_norm.png', dpi=300, bbox_inches='tight')
print("Skewness: %f" % np.log(target_variable.skew()))

print("Kurtosis: %f" % np.log(target_variable.kurt()))
df_train.columns
df_full = df_train[df_train.columns.difference(['SalePrice'])].append(df_test, ignore_index = False)

df_full.shape
df_num = df_full.select_dtypes(include = ['int64','float64'])

df_car = df_full.select_dtypes(include = ['object'])

print(df_full.shape[1])

print(df_num.shape[1])

print(df_car.shape[1])
df_num.columns
df_train.plot.scatter(x = 'GrLivArea', y = 'SalePrice')

plt.savefig('GrLivArea.png', dpi=300, bbox_inches='tight')

df_train.plot.scatter(x = 'TotalBsmtSF', y = 'SalePrice')

plt.savefig('TotalBsmtSF.png', dpi=300, bbox_inches='tight')

df_train.plot.scatter(x = 'GarageCars', y = 'SalePrice')

plt.savefig('GarageCars.png', dpi=300, bbox_inches='tight')

df_train.plot.scatter(x = 'OverallQual', y = 'SalePrice')

plt.savefig('OverallQual.png', dpi=300, bbox_inches='tight')

df_train.plot.scatter(x = 'YearBuilt', y = 'SalePrice')

plt.savefig('YearBuilt.png', dpi=300, bbox_inches='tight')

df_train.plot.scatter(x = 'FullBath', y = 'SalePrice')

plt.savefig('FullBath.png', dpi=300, bbox_inches='tight')

df_train.plot.scatter(x = 'PoolArea', y = 'SalePrice')

plt.savefig('PoolArea.png', dpi=300, bbox_inches='tight')
df_train.plot.scatter(x = '1stFlrSF', y = 'GrLivArea')

#plt.savefig('Scatter_GarageCars.png', dpi=300, bbox_inches='tight')

df_train.plot.scatter(x = '2ndFlrSF', y = 'GrLivArea')

#plt.savefig('Scatter_1.png', dpi=300, bbox_inches='tight')

df_train.plot.scatter(x = 'GarageCars', y = 'GarageArea')

#plt.savefig('Scatter_1.png', dpi=300, bbox_inches='tight')
cor_matrix = df_train.corr()

f, ax = plt.subplots(figsize=(20, 15))

sns.heatmap(cor_matrix, vmax=0.8, square=True, annot=False, linewidth = 5)

plt.savefig('Heatmap_1.png', dpi=300, bbox_inches='tight')
k = 15 #Looking for the top 15 features highly correlated with SalePrice

cols = cor_matrix.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df_train[cols].values.T)

f, ax = plt.subplots(figsize=(20, 15))

sns.set(font_scale=1.5)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols.values, xticklabels=cols.values)

plt.savefig('Heatmap_2.png', dpi=300, bbox_inches='tight')

plt.show()
df_car.columns
var = 'BsmtQual'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(28, 15))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

plt.savefig('BsmtQual.png', dpi=300, bbox_inches='tight')
var = 'CentralAir'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(28, 15))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

plt.savefig('CentralAir.png', dpi=300, bbox_inches='tight')
var = 'ExterCond'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(28, 15))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

plt.savefig('ExterCond.png', dpi=300, bbox_inches='tight')
var = 'GarageQual'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(28, 15))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

plt.savefig('GarageQual.png', dpi=300, bbox_inches='tight')
var = 'GarageCond'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(28, 15))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

plt.savefig('GarageCond.png', dpi=300, bbox_inches='tight')
var = 'KitchenQual'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(28, 15))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

plt.savefig('KitchenQual.png', dpi=300, bbox_inches='tight')
var = 'Neighborhood'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(28, 15))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

plt.savefig('Neighborhood.png', dpi=300, bbox_inches='tight')
var = 'FullBath'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(28, 15))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

plt.savefig('FullBath.png', dpi=300, bbox_inches='tight')
#Imputing missing LotFrontage values

vec_lf = df_full.loc[df_full['LotFrontage'].isnull()].index.tolist()

df_full.ix[vec_lf,'LotFrontage'] = 70

df_full['LotFrontage'].isnull().sum()
total = df_full.isnull().sum().sort_values(ascending=False)

percent = (df_full.isnull().sum()/df_full.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data[missing_data['Total'] >= 1]
var = 'BsmtExposure'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(28, 15))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
var = 'BsmtFinType1'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(28, 15))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
var = 'BsmtFinType2'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(28, 15))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
var = 'BsmtCond'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(28, 15))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
var = 'BsmtQual'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(28, 15))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
var = 'MasVnrArea'

df_train.plot.scatter(x = var, y = 'SalePrice')

var = 'MasVnrType'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(28, 15))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
df_full['LotFrontage'].shape
total1 = df_full.isnull().sum().sort_values(ascending=False)

percent1 = (df_full.isnull().sum()/df_full.isnull().count()).sort_values(ascending=False)

missing_data1 = pd.concat([total1, percent1], axis=1, keys=['Total', 'Percent'])

missing_data1[missing_data1['Total'] > 0]
df_full['LotFrontage'].isnull().sum()
missing_data1[missing_data1['Total'] > 1]
df_full = df_full.drop((missing_data[missing_data['Total'] > 5]).index,1)
df_full['LotFrontage'].shape
df_full.isnull().sum()
total1 = df_full.isnull().sum().sort_values(ascending=False)

percent1 = (df_full.isnull().sum()/df_full.isnull().count()).sort_values(ascending=False)

missing_data1 = pd.concat([total1, percent1], axis=1, keys=['Total', 'Percent'])

missing_data1[missing_data1['Total'] > 0]
df_train = df_full[:df_train.shape[0]]

df_train['SalePrice'] = target_variable

df_train.head(10)
df_test = df_full[df_train.shape[0]:]

df_test.shape

df_test.head(10)
#df_train['LotFrontage']

#df_test['LotFrontage']
total1 = df_train.isnull().sum().sort_values(ascending=False)

percent1 = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data1 = pd.concat([total1, percent1], axis=1, keys=['Total', 'Percent'])

missing_data1[missing_data1['Total'] > 0]
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
total1 = df_train.isnull().sum().sort_values(ascending=False)

percent1 = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data1 = pd.concat([total1, percent1], axis=1, keys=['Total', 'Percent'])

missing_data1[missing_data1['Total'] > 0]
df_train.shape
############################################################################################################################
df_test.shape
total2 = df_test.isnull().sum().sort_values(ascending=False)

percent2 = (df_test.isnull().sum()/df_test.isnull().count()).sort_values(ascending=False)

missing_data1 = pd.concat([total2, percent2], axis=1, keys=['Total', 'Percent'])

missing_data1[missing_data1['Total'] > 0]
df_full['MSZoning'].describe()
vec = df_test.loc[df_test['MSZoning'].isnull()].index.tolist()

df_test.ix[vec,'MSZoning']

#df_test[df_test.loc[df_test['MSZoning'].isnull()].index.tolist()]
df_test.ix[vec,'MSZoning'] = 'RL'

df_test.ix[vec,'MSZoning']
df_full['Functional'].describe()
vec = df_test.loc[df_test['Functional'].isnull()].index.tolist()

df_test.ix[vec,'Functional']

#df_test[df_test.loc[df_test['MSZoning'].isnull()].index.tolist()]
df_test.ix[vec,'Functional'] = 'Typ'

df_test.ix[vec,'Functional']
df_full['Utilities'].describe()
vec = df_test.loc[df_test['Utilities'].isnull()].index.tolist()

df_test.ix[vec,'Utilities']

#df_test[df_test.loc[df_test['MSZoning'].isnull()].index.tolist()]
df_test.ix[vec,'Utilities'] = 'AllPub'

df_test.ix[vec,'Utilities']
df_full['BsmtHalfBath'].describe()
vec = df_test.loc[df_test['BsmtHalfBath'].isnull()].index.tolist()

df_test.ix[vec,'BsmtHalfBath']

#df_test[df_test.loc[df_test['MSZoning'].isnull()].index.tolist()]
df_test.ix[vec,'BsmtHalfBath'] = 0

df_test.ix[vec,'BsmtHalfBath']
df_test.shape
df_full['BsmtFullBath'].describe()
vec = df_test.loc[df_test['BsmtFullBath'].isnull()].index.tolist()

df_test.ix[vec,'BsmtFullBath']

#df_test[df_test.loc[df_test['MSZoning'].isnull()].index.tolist()]
df_test.ix[vec,'BsmtFullBath'] = 0

df_test.ix[vec,'BsmtFullBath']
df_test.shape
df_full['BsmtFinSF2'].describe()
vec = df_test.loc[df_test['BsmtFinSF2'].isnull()].index.tolist()

df_test.ix[vec,'BsmtFinSF2']

#df_test[df_test.loc[df_test['MSZoning'].isnull()].index.tolist()]
df_test.ix[vec,'BsmtFinSF2'] = 0

df_test.ix[vec,'BsmtFinSF2']
df_test.shape
total2 = df_test.isnull().sum().sort_values(ascending=False)

percent2 = (df_test.isnull().sum()/df_test.isnull().count()).sort_values(ascending=False)

missing_data1 = pd.concat([total2, percent2], axis=1, keys=['Total', 'Percent'])

missing_data1[missing_data1['Total'] > 0]
df_full['SaleType'].describe()
vec = df_test.loc[df_test['SaleType'].isnull()].index.tolist()

df_test.ix[vec,'SaleType']

#df_test[df_test.loc[df_test['MSZoning'].isnull()].index.tolist()]
df_test.ix[vec,'SaleType'] = 'WD'

df_test.ix[vec,'SaleType']
df_test.shape
df_full['Exterior1st'].describe()
vec = df_test.loc[df_test['Exterior1st'].isnull()].index.tolist()

df_test.ix[vec,'Exterior1st']

#df_test[df_test.loc[df_test['MSZoning'].isnull()].index.tolist()]
df_test.ix[vec,'Exterior1st'] = 'VinylSd'

df_test.ix[vec,'Exterior1st']
df_test.shape
df_full['KitchenQual'].describe()
vec = df_test.loc[df_test['KitchenQual'].isnull()].index.tolist()

df_test.ix[vec,'KitchenQual']

#df_test[df_test.loc[df_test['MSZoning'].isnull()].index.tolist()]
df_test.ix[vec,'KitchenQual'] = 'TA'

df_test.ix[vec,'KitchenQual']
df_test.shape
df_full['Exterior2nd'].describe()
vec = df_test.loc[df_test['Exterior2nd'].isnull()].index.tolist()

df_test.ix[vec,'Exterior2nd']

#df_test[df_test.loc[df_test['MSZoning'].isnull()].index.tolist()]
df_test.ix[vec,'Exterior2nd'] = 'VinylSd'

df_test.ix[vec,'Exterior2nd']
total2 = df_test.isnull().sum().sort_values(ascending=False)

percent2 = (df_test.isnull().sum()/df_test.isnull().count()).sort_values(ascending=False)

missing_data1 = pd.concat([total2, percent2], axis=1, keys=['Total', 'Percent'])

missing_data1[missing_data1['Total'] > 0]
df_test.shape
df_test.ix[df_test.loc[df_test['GarageArea'].isnull()].index.tolist(),'GarageArea'] = 0

df_test.ix[df_test.loc[df_test['TotalBsmtSF'].isnull()].index.tolist(),'TotalBsmtSF'] = 0

df_test.ix[df_test.loc[df_test['GarageCars'].isnull()].index.tolist(),'GarageCars'] = 0

df_test.ix[df_test.loc[df_test['BsmtFinSF1'].isnull()].index.tolist(),'BsmtFinSF1'] = 0

df_test.ix[df_test.loc[df_test['BsmtUnfSF'].isnull()].index.tolist(),'BsmtUnfSF'] = 0

df_test.shape
total2 = df_test.isnull().sum().sort_values(ascending=False)

percent2 = (df_test.isnull().sum()/df_test.isnull().count()).sort_values(ascending=False)

missing_data1 = pd.concat([total2, percent2], axis=1, keys=['Total', 'Percent'])

missing_data1[missing_data1['Total'] > 0]
df_test.shape
#df_full['LotFrontage']


for colname, col in df_train.select_dtypes(include = ['int64','float64']).iteritems():

    plt.scatter(df_train[colname], df_train['SalePrice'])

    plt.ylabel('Sale Price')

    plt.xlabel(colname)

    plt.show()
df_lin_full = df_train[df_train.columns.difference(['SalePrice'])].append(df_test, ignore_index = False)

df_lin_full.shape
imp_feats = ['Id','GrLivArea','OverallQual','GarageCars','TotalBsmtSF','1stFlrSF','FullBath']

df_lin_full = df_lin_full[imp_feats]

df_lin_full.shape
df_lin_train = df_lin_full[:df_train.shape[0]]

df_lin_train.head(5)
df_lin_test = df_lin_full[df_train.shape[0]:]

df_lin_test.head(5)

df_lin_test.isnull().sum()
#Creating a scoring function to check mdoel performance using k-Fold Sampling of the training dataset

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error



n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train.values)

    rmse= np.sqrt(-cross_val_score(model, X_train.values, y_train, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)



print('Done')
from sklearn.linear_model import LinearRegression

lm = LinearRegression()

# Log transforming the SalePrice variable since it is left skewed

X = df_lin_train.drop('Id',1)

y = np.log(df_train['SalePrice'])



lm.fit(X, y)
benchmark_model = pd.DataFrame(zip(X.columns,lm.coef_), columns = ['features','Model Coefficients'])

benchmark_model
#sns.pairplot(X, x_vars=['GrLivArea'], y_vars=y, size=7, aspect=0.7, kind='reg')
benchmark_pred = lm.predict(df_lin_test.drop('Id',1))

len(benchmark_pred)

benchmark_sub = pd.DataFrame(np.exp(benchmark_pred), columns = ['SalePrice'])

benchmark_sub['Id'] = df_lin_test['Id']

#One of the Id values has become null. I will ahev to fill it back in

benchmark_sub.loc[benchmark_sub['Id'].isnull()].index.tolist()

benchmark_sub.ix[1379,'Id'] = 2840

benchmark_sub.ix[1375:1380,]



benchmark_sub['Id'] = benchmark_sub['Id'].astype('int64')

benchmark_sub.head(10)
#writing to a csv file

benchmark_sub.to_csv('benchmark_submission.csv')
from IPython.display import Image

Image(filename='Benchmark_Sub.jpg')
df_fl = df_train[df_train.columns.difference(['SalePrice'])].append(df_test, ignore_index = False)

df_fl = pd.get_dummies(df_fl)

df_fl.head(10)
df_train = df_fl[:df_train.shape[0]]

df_train['SalePrice'] = target_variable

df_test = df_fl[df_train.shape[0]:]
#pd.get_dummies(df_train).head(10)

df_train.head(10)

#df_train.shape

#df_test['LotFrontage']
pd.get_dummies(df_test).head(10)

df_test.head(10)

#df_test.shape
from sklearn.ensemble import RandomForestRegressor

#from sklearn.datasets import make_regression

from sklearn.grid_search import GridSearchCV

from sklearn.preprocessing import OneHotEncoder

from random import seed



seed(124578)

enc = OneHotEncoder()

rfg = RandomForestRegressor()

tuned_parameters = {'n_estimators': [100, 150, 200, 250, 300, 400, 500, 700, 800, 1000], 'max_depth': [1, 2, 3], 'min_samples_split': [1.0, 2, 3]}



clf = GridSearchCV(rfg, tuned_parameters, cv=10, n_jobs=-1, verbose=1)





X_train = df_train.drop('SalePrice',1)

y_train = np.log(df_train['SalePrice'])

clf.fit(X_train, y_train )
rf_best = clf.best_estimator_

#plt.hist(rf_best.feature_importances_, label = df_test.columns)

rf_best
#Writing model to pickle file

filename = 'randomforest_model.sav'

pickle.dump(rf_best, open(filename, 'wb'))
#Loading model from pickle file

rf_model_load = pickle.load(open(filename, 'rb'))

rf_model_load
rf_best.fit(X_train, y_train)

print('Model Fitting done....')
rf_rmse = rmsle_cv(rf_best)

rf_rmse.mean()
rf_opt_pred = rf_best.predict(df_test)

print('Prediction with optimium model on test set done...')
rf_sub = pd.DataFrame(np.exp(rf_opt_pred), columns = ['SalePrice'])

rf_sub['Id'] = df_test['Id']

rf_sub.head(10)
#writing this result to a csv and making another submission

rf_sub.to_csv('rf_submission.csv')
from sklearn.metrics import make_scorer, mean_squared_error

from sklearn.cross_validation import cross_val_score



rf_scorer = make_scorer(mean_squared_error, False)

n_est = [150,250,500,750,1000]



for i in n_est:

    rf_clf = RandomForestRegressor(n_estimators=i, n_jobs=-1)

    rf_cv_score = np.sqrt(-cross_val_score(estimator=rf_clf, X=X_train, y=y_train, cv=15, scoring = rf_scorer))



    plt.figure(figsize=(10,5))

    plt.bar(range(len(rf_cv_score)), rf_cv_score)

    plt.title('Cross Validation Score for '+ str(i) + " estimators")

    plt.ylabel('RMSE')

    plt.xlabel('Iteration')



    plt.plot(range(len(rf_cv_score) + 1), [rf_cv_score.mean()] * (len(rf_cv_score) + 1))

    plt.tight_layout()
#rf_best = clf.best_estimator_

rf_best = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,

           max_features='auto', max_leaf_nodes=None,

           min_impurity_decrease=0.0, min_impurity_split=None,

           min_samples_leaf=1, min_samples_split=2,

           min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=1,

           oob_score=False, random_state=None, verbose=0, warm_start=False)
#Writing model to pickle file

filename = 'randomforest_model.sav'

pickle.dump(rf_best, open(filename, 'wb'))
#Loading model from pickle file

rf_model_load = pickle.load(open(filename, 'rb'))

rf_model_load
rmsle_cv(rf_best).mean()
rf_best.fit(X_train, y_train)

print('Model Fitting done....')
rf_opt_pred = rf_best.predict(df_test)

print('Prediction with optimium model on test set done...')
rf_sub = pd.DataFrame(np.exp(rf_opt_pred), columns = ['SalePrice'])

rf_sub['Id'] = df_test['Id']

rf_sub.head(10)
#writing this result to a csv and making another submission

rf_sub.to_csv('rf_submission.csv')
from IPython.display import Image

Image(filename='RF_N500_Sub.jpg')
##########################################################################################################################
from sklearn.grid_search import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

from sklearn.model_selection import StratifiedKFold



seed(235689)

#seed(235690)

xg_model = XGBRegressor()



#GridSearchCV

n_estimators = [50, 100, 150, 200, 250, 300, 500, 750, 1000]

max_depth = [2, 3, 4, 5, 6]

learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]

param_grid = param_grid = dict(max_depth=max_depth, n_estimators=n_estimators, learning_rate = learning_rate)



k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=5)

grid_search = GridSearchCV(xg_model, param_grid, scoring="neg_mean_squared_log_error", n_jobs=-1, cv=20, verbose=1)

print("GridSearch done ...")
X_train = df_train.drop('SalePrice',1)

y_train = np.log(df_train['SalePrice'])

final_result = grid_search.fit(X_train, y_train)

print("Best: %f using %s" % (final_result.best_score_, final_result.best_params_))
final_result.grid_scores_
final_xgb_model = final_result.best_estimator_

final_xgb_model
print('Beginning Model Fitting....')

final_xgb_model.fit(X_train, y_train)

print('Model fitting completed...')
#Writing model to pickle file

filename = 'xgboost_model.sav'

pickle.dump(final_xgb_model, open(filename, 'wb'))
#Loading model from pickle file

xg_model_load = pickle.load(open('xgboost_model.sav', 'rb'))

xg_model_load
rmsle_cv(final_xgb_model).mean()
xgb_pred = final_xgb_model.predict(df_test)

print('Prediction with optimium model on test set done...')
xgb_sub = pd.DataFrame(np.exp(xgb_pred), columns = ['SalePrice'])

xgb_sub['Id'] = df_test['Id']

xgb_sub.head(10)
#writing this result to a csv and making another submission

xgb_sub.to_csv('xgb_submission.csv')
from IPython.display import Image

Image(filename='XGB_Sub.jpg')
datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
from sklearn.linear_model import Lasso

from sklearn.grid_search import GridSearchCV



seed(785623)



lasso_model = Lasso()



#GridSearchCV

alpha = [0.0001, 0.0002, 0.0003, 0.0004,0.0005,0.0006,0.007, 0.0008,0.0009,0.001, 0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09, 0.1, 0.2, 0.3,0.4]

param_grid_ls = param_grid = dict(alpha=alpha)



grid_search_ls = GridSearchCV(lasso_model, param_grid_ls, scoring="neg_mean_squared_error", n_jobs=-1, cv=15, verbose=1)

print("GridSearch done ...")



X_train = df_train.drop('SalePrice',1)

y_train = np.log(df_train['SalePrice'])

final_result_ls = grid_search_ls.fit(X_train, y_train)
final_lasso_model = final_result_ls.best_estimator_

final_lasso_model
final_result_ls.grid_scores_
#Writing model to pickle file

filename = 'lasso_model.sav'

pickle.dump(final_lasso_model, open(filename, 'wb'))
#Loading model from pickle file

ls_model_load = pickle.load(open('lasso_model.sav', 'rb'))

ls_model_load
rmsle_cv(final_lasso_model).mean()
print('Beginning Model Fitting....')

final_lasso_model.fit(X_train, y_train)

print('Model fitting completed...')
lasso_pred = final_lasso_model.predict(df_test)

print('Prediction with optimium model on test set done...')
lasso_sub = pd.DataFrame(np.exp(lasso_pred), columns = ['SalePrice'])

lasso_sub['Id'] = df_test['Id']

lasso_sub.head(10)
#writing this result to a csv and making another submission

lasso_sub.to_csv('lasso_submission.csv')
from IPython.display import Image

Image(filename='Lasso_Sub.jpg')
from IPython.display import Image

Image(filename='Final_Models.jpg')
#Loading saved models from pickle files

rf_model_load = pickle.load(open('randomforest_model.sav', 'rb'))

xg_model_load = pickle.load(open('xgboost_model.sav', 'rb'))

ls_model_load = pickle.load(open('lasso_model.sav', 'rb'))
ls_model_load
X_train = df_train.drop('SalePrice',1)

y_train = np.log(df_train['SalePrice'])

X_train.head(10)
raw_data = {'Models': ['Benchmark Linear regression', 'Random Forest Regressor', 'XGBoost Regressor', 'Lasso Regularized Linear Regression'],

        'train_rmsle_score': [0,rmsle_cv(rf_model_load).mean(),rmsle_cv(xg_model_load).mean(),rmsle_cv(ls_model_load).mean()],

        'test_rmsle_score': [0.17649,0.14544,0.14029,0.12671]}

df = pd.DataFrame(raw_data, columns = ['Models', 'train_rmsle_score', 'test_rmsle_score'])

df.shape
1-df['train_rmsle_score']
# Setting the positions and width for the bars

pos = list(range(len(df['train_rmsle_score']))) 

width = 0.4 



# Plotting the bars

fig, ax = plt.subplots(figsize=(10,5))



# Create a bar with train_rmsle_score data,

# in position pos,

plt.bar(pos, 

        #using df['pre_score'] data,

        df['train_rmsle_score'], 

        # of width

        width, 

        # with alpha 0.5

        alpha=0.5, 

        # with color

        color='#EE3224', 

        # with label the first value in first_name

        label=df['Models'][0]) 



# Create a bar with test_rmsle_score data,

# in position pos + some width buffer,

plt.bar([p + width for p in pos], 

        #using df['mid_score'] data,

        df['test_rmsle_score'],

        # of width

        width, 

        # with alpha 0.5

        alpha=0.5, 

        # with color

        color='#F78F1E', 

        # with label the second value in first_name

        label=df['Models'][1]) 



# Set the y axis label

ax.set_ylabel('Score')



# Set the chart's title

ax.set_title('Model RMSLE Scores for Training and Test sets.')



# Set the position of the x ticks

ax.set_xticks([p + 1.5 * width for p in pos])



# Set the labels for the x ticks

ax.set_xticklabels(df['Models'], rotation=45, ha='right')



# Setting the x-axis and y-axis limits

plt.xlim(min(pos)-width, max(pos)+width*4)

plt.ylim([0, max(df['train_rmsle_score'] + df['test_rmsle_score'])])



# Adding the legend and showing the plot

plt.legend(['Training Score', 'Testing Score'], loc='upper left')

plt.grid()

plt.show()
accuracies = pd.DataFrame(df['Models'])

accuracies['Train_accuracy'] = 1 - df['train_rmsle_score']

accuracies['Test_accuracy'] = 1 - df['test_rmsle_score']

accuracies.ix[0,'Train_accuracy'] = 0

accuracies
# Setting the positions and width for the bars

pos = list(range(len(accuracies['Train_accuracy']))) 

width = 0.4 



# Plotting the bars

fig, ax = plt.subplots(figsize=(10,5))



# Create a bar with train_rmsle_score data,

# in position pos,

plt.bar(pos, 

        #using accuracies['Train_accuracy'] data,

        accuracies['Train_accuracy'], 

        # of width

        width, 

        # with alpha 0.5

        alpha=0.5, 

        # with color

        color='#EE3224', 

        # with label the first value in first_name

        label=accuracies['Models'][0]) 



# Create a bar with test_rmsle_score data,

# in position pos + some width buffer,

plt.bar([p + width for p in pos], 

        #using accuracies['Test_accuracy'] data,

        accuracies['Test_accuracy'],

        # of width

        width, 

        # with alpha 0.5

        alpha=0.5, 

        # with color

        color='#F78F1E', 

        # with label the second value in first_name

        label=accuracies['Models'][1]) 



# Set the y axis label

ax.set_ylabel('Accuracy Scores')



# Set the chart's title

ax.set_title('Model Accuracy Scores for Training and Test sets.')



# Set the position of the x ticks

ax.set_xticks([p + 1.5 * width for p in pos])



# Set the labels for the x ticks

ax.set_xticklabels(accuracies['Models'], rotation=45, ha='right')



# Setting the x-axis and y-axis limits

plt.xlim(min(pos)-width, max(pos)+width*4)

plt.ylim([0, max(accuracies['Train_accuracy'] + accuracies['Test_accuracy'])])



# Adding the legend and showing the plot

plt.legend(['Training Accuracy', 'Testing Accuracy'], loc='upper left')

plt.grid()

plt.show()
#fitting the three models to the training data

rf_model_load.fit(X_train, y_train)

print('Random Forest model fitted')

xg_model_load.fit(X_train, y_train)

print('XGBoost model fitted')

ls_model_load.fit(X_train, y_train)

print('Lasso model fitted')
#Generating predictions on the test data for the three models loaded from pickle data files.

rf_pred_wa = rf_model_load.predict(df_test)

xg_pred_wa = xg_model_load.predict(df_test)

ls_pred_wa = ls_model_load.predict(df_test)

rf_pred_wa
#Generating predictions on the test data for the three models loaded from pickle data files.

#rf_pred_wat = np.exp(rf_model_load.predict(X_train))

#xg_pred_wat = np.exp(xg_model_load.predict(X_train))

#ls_pred_wat = np.exp(ls_model_load.predict(X_train))



rf_pred_wat = rf_model_load.predict(X_train)

xg_pred_wat = xg_model_load.predict(X_train)

ls_pred_wat = ls_model_load.predict(X_train)
#Defining a rmse calculator to check training accuracy

def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))
x1 = rf_pred_wa

x2 = xg_pred_wa

x3 = ls_pred_wa

#y_actual = np.exp(y_train)



#df_stack = pd.DataFrame(x1,x2,x3,y_actual, columns=['RandomForest','XGBoost','LassoRegression','ActualValues'])

df_stack_tst = pd.DataFrame(x1,columns=['RandomForest'])

df_stack_tst['XGBoost'] = x2

df_stack_tst['LassoRegression'] = x3

#df_stack_tst['ActualValues'] = y_train

df_stack_tst.head(5)
y1 = rf_pred_wat

y2 = xg_pred_wat

y3 = ls_pred_wat

#y_actual = np.exp(y_train)



#df_stack = pd.DataFrame(x1,x2,x3,y_actual, columns=['RandomForest','XGBoost','LassoRegression','ActualValues'])

df_stack_trn = pd.DataFrame(x1,columns=['RandomForest'])

df_stack_trn['XGBoost'] = y2

df_stack_trn['LassoRegression'] = y3

df_stack_trn['ActualValues'] = y_train

df_stack_trn.head(5)
results = pd.DataFrame(columns=['XGBoost Weight','Lasso Weight','Training RMSLE'])

results['XGBoost Weight'] = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

results['Lasso Weight'] = [1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.0]

results
wghts = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

index = [0,1,2,3,4,5,6,7,8,9]



dfr = []

x = 0

for i in wghts:

    stacked1 =  (df_stack_trn['XGBoost']*i + df_stack_trn['LassoRegression']*(1.0-i))

    r = rmsle(y_train,stacked1)

    dfr.append(r)

    #results.ix[index,'Lasso Weight'] = (1.0-i)

    #results.ix[index,'Training RMSLE'] = r

    

#stacked1.head(5)

dfr
results['Training RMSLE'] = dfr

results
i = 0.9

model_1 = df_stack_tst['XGBoost']*i + df_stack_tst['LassoRegression']*(1.0-i)

#odel_1 = np.exp(model_1)

stack_pred_1 = pd.DataFrame(np.exp(model_1), columns = ['SalePrice'])

stack_pred_1['Id'] = df_test['Id']

stack_pred_1.to_csv('Stacked_Model_1.csv')

stack_pred_1.head(5)
i = 0.7

model_2 = df_stack_tst['XGBoost']*i + df_stack_tst['LassoRegression']*(1.0-i)

#odel_1 = np.exp(model_1)

stack_pred_2 = pd.DataFrame(np.exp(model_2), columns = ['SalePrice'])

stack_pred_2['Id'] = df_test['Id']

stack_pred_2.to_csv('Stacked_Model_2.csv')

stack_pred_2.head(5)
i = 0.6

model_3 = df_stack_tst['XGBoost']*i + df_stack_tst['LassoRegression']*(1.0-i)

#odel_1 = np.exp(model_1)

stack_pred_3 = pd.DataFrame(np.exp(model_3), columns = ['SalePrice'])

stack_pred_3['Id'] = df_test['Id']

stack_pred_3.to_csv('Stacked_Model_3.csv')

stack_pred_3.head(5)
i = 0.2

model_4 = df_stack_tst['XGBoost']*i + df_stack_tst['LassoRegression']*(1.0-i)

#odel_1 = np.exp(model_1)

stack_pred_4 = pd.DataFrame(np.exp(model_4), columns = ['SalePrice'])

stack_pred_4['Id'] = df_test['Id']

stack_pred_4.to_csv('Stacked_Model_4.csv')

stack_pred_4.head(5)
from IPython.display import Image

Image(filename='All_Stacked.jpg')
from IPython.display import Image

Image(filename='Stackd_Model_4.jpg')
i = 0.1

model_5 = df_stack_tst['XGBoost']*i + df_stack_tst['LassoRegression']*(1.0-i)

#odel_1 = np.exp(model_1)

stack_pred_5 = pd.DataFrame(np.exp(model_5), columns = ['SalePrice'])

stack_pred_5['Id'] = df_test['Id']

stack_pred_5.to_csv('Stacked_Model_5.csv')

stack_pred_5.head(5)
from IPython.display import Image

Image(filename='Stackd_Model_5.jpg')
i = 0.5

model_6 = df_stack_tst['XGBoost']*i + df_stack_tst['LassoRegression']*(1.0-i)

#odel_1 = np.exp(model_1)

stack_pred_6 = pd.DataFrame(np.exp(model_6), columns = ['SalePrice'])

stack_pred_6['Id'] = df_test['Id']

stack_pred_6.to_csv('Stacked_Model_6.csv')

stack_pred_6.head(5)
i = 0.3

model_7 = df_stack_tst['XGBoost']*i + df_stack_tst['LassoRegression']*(1.0-i)

#odel_1 = np.exp(model_1)

stack_pred_7 = pd.DataFrame(np.exp(model_7), columns = ['SalePrice'])

stack_pred_7['Id'] = df_test['Id']

stack_pred_7.to_csv('Stacked_Model_7.csv')

stack_pred_7.head(5)
i = 0.4

model_8 = df_stack_tst['XGBoost']*i + df_stack_tst['LassoRegression']*(1.0-i)

#odel_1 = np.exp(model_1)

stack_pred_8 = pd.DataFrame(np.exp(model_8), columns = ['SalePrice'])

stack_pred_8['Id'] = df_test['Id']

stack_pred_8.to_csv('Stacked_Model_8.csv')

stack_pred_8.head(5)
#Model 7 :: XGBoost(0.3) + Lasso(0.7)

from IPython.display import Image

Image(filename='Stackd_Model_7.jpg')
#Model 8 :: XGBoost(0.4) + Lasso(0.6)

from IPython.display import Image

Image(filename='Stackd_Model_8.jpg')
i = 0.2

stacked_pred_trn = df_stack_trn['XGBoost']*i + df_stack_trn['LassoRegression']*(1.0-i)



pred_comp1 = pd.DataFrame(np.exp(stacked_pred_trn), columns = ['Stacked Predicted_SalePrice'])

pred_comp1['Actual_SalePrice'] = target_variable

pred_comp1['Id'] = df_test['Id']

pred_comp1.head(5)


stacked_pred_trn_ls = ls_model_load.predict(X_train)



pred_comp2 = pd.DataFrame(np.exp(stacked_pred_trn_ls), columns = ['Lasso Predicted_SalePrice'])

pred_comp2['Actual_SalePrice'] = target_variable

pred_comp2['Id'] = df_test['Id']

pred_comp2.head(5)


stacked_pred_trn_lm = lm.predict(df_lin_test.drop('Id',1))



pred_comp3 = pd.DataFrame(np.exp(stacked_pred_trn_lm), columns = ['Benchmark Predicted_SalePrice'])

pred_comp3['Actual_SalePrice'] = target_variable

pred_comp3['Id'] = df_test['Id']

pred_comp3.head(5)
#Plotting these results to check how the model fot's the training data

x = pred_comp1['Actual_SalePrice']

y = pred_comp1['Stacked Predicted_SalePrice']

z = pred_comp2['Lasso Predicted_SalePrice']

w = pred_comp3['Benchmark Predicted_SalePrice']

ids = pred_comp1.shape[0]/2

fig,ax1 = plt.subplots(figsize=(25,35))



ax1 = fig.add_subplot(311)

ax1.scatter(x[:ids], y[:ids], s=10, c='b', marker="s", label='Actual SalePrice')

ax1.scatter(x[ids:],y[ids:], s=10, c='r', marker="o", label='Stacked Predicted SalePrice')

plt.legend(loc='upper left');



ax1 = fig.add_subplot(312)

ax1.scatter(x[:ids], z[:ids], s=10, c='b', marker="s", label='Actual SalePrice')

ax1.scatter(x[ids:],z[ids:], s=10, c='r', marker="o", label='Lasso Predicted SalePrice')



ax1 = fig.add_subplot(313)

ax1.scatter(x[:ids], z[:ids], s=10, c='b', marker="s", label='Actual SalePrice')

ax1.scatter(x[ids:],w[ids:], s=10, c='r', marker="o", label='Benchmark Predicted SalePrice')



plt.legend(loc='upper left');



plt.show()
final_df_models = ['Benchmark Model','Random Forest Try 1','Random Forest Try 2','Gradient Boosting XGBoost','Lasso Regression','Stacked Model 1 :: XGBoost(0.9) + Lasso(0.1)','Stacked Model 2 :: XGBoost(0.7) + Lasso(0.3)','Stacked Model 3 :: XGBoost(0.6) + Lasso(0.4)','Stacked Model 4 :: XGBoost(0.2) + Lasso(0.8)','Stacked Model 5 :: XGBoost(0.1) + Lasso(0.9)','Stacked Model 6 :: XGBoost(0.5) + Lasso(0.5)','Stacked Model 7 :: XGBoost(0.3) + Lasso(0.7)','Stacked Model 8 :: XGBoost(0.4) + Lasso(0.6)']

final_df_scores = [0.17649,0.20190,0.14544,0.14029,0.12671,0.13677,0.13101,0.12883,0.12526,0.12571,0.12714,0.12535,0.12598]



final_sum_df = pd.DataFrame(np.column_stack([final_df_models, final_df_scores]),columns = ['Models','Test data RMSLE from Kaggle'])



final_sum_df['Test data RMSLE from Kaggle'] = final_sum_df['Test data RMSLE from Kaggle'].astype('float64')

final_sum_df
pos = list(range(len(final_sum_df['Models']))) 

width = 0.2 



# Plotting the bars

fig, ax = plt.subplots(figsize=(20,20))



# Create a bar with train_rmsle_score data,

# in position pos,

plt.bar(pos, 

        #using accuracies['Train_accuracy'] data,

        final_sum_df['Test data RMSLE from Kaggle'], 

        # of width

        width, 

        # with alpha 0.5

        alpha=0.5, 

        # with color

        color='#EE3224', 

        # with label the first value in first_name

        label=final_sum_df['Models'][0]) 







# Set the y axis label

ax.set_ylabel('Accuracy Scores')



# Set the chart's title

ax.set_title('Model Accuracy Scores for Test sets.')



# Set the position of the x ticks

ax.set_xticks([p + 1.5 * width for p in pos])



# Set the labels for the x ticks

ax.set_xticklabels(final_sum_df['Models'], rotation=45, ha='right')



# Setting the x-axis and y-axis limits

plt.xlim(min(pos)-width, max(pos)+width*4)

plt.ylim([0, 0.05+max(final_sum_df['Test data RMSLE from Kaggle'])])



# Adding the legend and showing the plot

plt.legend(['Training Accuracy'], loc='upper left')

plt.grid()

plt.show()
lasso_coef = ls_model_load.coef_
lasso_cols = X_train.columns

lasso_cols.shape
lasso_df = pd.DataFrame(np.column_stack([lasso_cols,lasso_coef]), columns = ['Feature','Lasso Coefficient'])

lasso_df.sort_values(by ='Lasso Coefficient', ascending = False).head(25)
final_df_models1 = ['Benchmark Model','Random Forest Try 1','Random Forest Try 2','Gradient Boosting XGBoost','Lasso Regression','Stacked Model 1 :: XGBoost(0.9) + Lasso(0.1)','Stacked Model 2 :: XGBoost(0.7) + Lasso(0.3)','Stacked Model 3 :: XGBoost(0.6) + Lasso(0.4)','Stacked Model 4 :: XGBoost(0.2) + Lasso(0.8)','Stacked Model 5 :: XGBoost(0.1) + Lasso(0.9)','Stacked Model 6 :: XGBoost(0.5) + Lasso(0.5)','Stacked Model 7 :: XGBoost(0.3) + Lasso(0.7)','Stacked Model 8 :: XGBoost(0.4) + Lasso(0.6)']

final_df_scores1 = [0.17649,0.20190,0.14544,0.14029,0.12671,0.13677,0.13101,0.12883,0.12526,0.12571,0.12714,0.12535,0.12598]

final_df_ranks1 = [1800,1800,1395,1318,902,902,902,902,839,839,839,839,839]



final_sum_df1 = pd.DataFrame(np.column_stack([final_df_models1, final_df_scores1, final_df_ranks1]),columns = ['Models','Test data RMSLE from Kaggle','Leadership Board Ranking'])



final_sum_df1['Test data RMSLE from Kaggle'] = final_sum_df1['Test data RMSLE from Kaggle'].astype('float64')

final_sum_df1['Leadership Board Ranking'] = final_sum_df1['Leadership Board Ranking'].astype('int64')



final_sum_df1.sort_values(by = 'Leadership Board Ranking', ascending = False)

final_sum_df1
fig, ax1= plt.subplots(figsize=(20,20))

ax2 = ax1.twinx()  # set up the 2nd axis



ax1.bar(pos, 

        #using accuracies['Train_accuracy'] data,

        final_sum_df1['Test data RMSLE from Kaggle'], 

        # of width

        width, 

        # with alpha 0.5

        alpha=0.4, 

        # with color

        color='#EE3224', 

        # with label the first value in first_name

        label=final_sum_df1['Models'][0]) 



ax2.plot(final_df_ranks1)

ax1.axes.set_xticklabels(final_sum_df1['Models'], rotation = 45, ha = 'right')

ax1.xaxis.set_visible(True)



ax1.set_ylabel('RMSLE Scores')

ax2.set_ylabel('Leaderboard Rankings Scores')

# Set the chart's title

ax1.set_title('Model Test RMSLE Scores and Kaggle Leaderboard ranking progression.')



plt.xlim(min(pos)-width, max(pos)+width*4)

ax1.set_xticks([p + 1.5 * width for p in pos])

plt.legend(['Training Accuracy', 'Leaderboard Ranking'], loc='upper left')



#ax2.set_xticklabels(final_sum_df1['Models'], rotation=45, ha='right')