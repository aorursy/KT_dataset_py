# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
plt.style.available
plt.style.use('seaborn-darkgrid')
import scipy.stats as stats
import pylab 
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings
warnings.filterwarnings('ignore')


# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
fig, ax = plt.subplots(1,3, figsize=(21,6))
sns.distplot(train.SalePrice, ax=ax[0])
sns.boxplot(y= train.SalePrice,ax=ax[1], color='green')
stats.probplot(train.SalePrice, dist="norm", plot=pylab)
pylab.show()

print('Skewness: ',train.SalePrice.skew())
print('Kurtosis: ',train.SalePrice.kurt())
model = LinearRegression().fit(train.GrLivArea.values.reshape(-1,1), train.SalePrice.values.reshape(-1,1))


fig, ax = plt.subplots(1,2, figsize=(16,6))
sns.scatterplot(train.GrLivArea, train.SalePrice, ax=ax[0], color='blue')
ax[0].set_title('Regression Plot')
for i in range(len(train.SalePrice)):
    ax[0].plot((train.GrLivArea.iloc[i], train.GrLivArea.iloc[i]), (train.SalePrice.values[i], model.predict(train.GrLivArea.values.reshape(-1,1))[i]), 'r-', color='red', alpha=0.4)
ax[0].plot(train.GrLivArea, model.predict(train.GrLivArea.values.reshape(-1,1)), color='red')
ax[1].set_title('Residual Plot')
sns.residplot(train.GrLivArea, train.SalePrice, ax=ax[1], color='green', scatter_kws={'alpha':0.3})
plt.show()
fig, ax = plt.subplots(1,3, figsize=(20,6))

sns.residplot(train.GrLivArea, np.log1p(train.SalePrice), lowess=True, color="red", ax=ax[0], scatter_kws={'alpha': 0.1, 'color':'blue'})
ax[0].set_title('Log Transformation')
sns.residplot(train.GrLivArea, np.sqrt(train.SalePrice), lowess=True, color="red", ax=ax[1], scatter_kws={'alpha': 0.1, 'color':'blue'})
ax[1].set_title('Square Root Transformation')
sns.residplot(train.GrLivArea, boxcox1p(train.SalePrice, boxcox_normmax(train.SalePrice + 1)), lowess=True, color="red", ax=ax[2], scatter_kws={'alpha': 0.1, 'color':'blue'})
ax[2].set_title('Box-Cox Transformation')
plt.show()
fig, ax = plt.subplots(2, 2, figsize=(20,10))

sns.distplot(train.SalePrice, color="red", ax=ax[0][0])
ax[0][0].set_title('Sale Prices')
sns.distplot(np.log1p(train.SalePrice), ax=ax[0][1])
ax[0][1].set_title('Log-transformed Sale Prices')
sns.boxplot(y= train.SalePrice, color="red", ax=ax[1][0])
sns.boxplot(y= np.log1p(train.SalePrice), ax=ax[1][1])
plt.show()
from collections import Counter
def detect_outliers(data, features):
    indices = {}
    fo = {}    
    for f in features:
        outliers = []
        median = np.median(data[f])
        
        q1 = np.nanpercentile(data[f], 25)
        q3 = np.nanpercentile(data[f], 75)
        iqr = q3 - q1
        
        low  = q1 - 1.5*iqr
        high = q3 + 1.5*iqr
        
        outlier_indices = data[(data[f] < low) | (data[f] > high)].index
        outliers.extend(outlier_indices)
        
        if outliers: 
            fo.update({f : len(outliers)})
            indices.update({f : outliers})
    return fo

#     indices = [i for i,j in Counter(outliers).items() if j > 0]
#     return indices

print('Number of Outliers for SalePrice: ',detect_outliers(train, ['SalePrice']))
train['LogSalePrice'] = np.log1p(train.SalePrice)
print('Number of Outliers for Log-transformed SalePrice: ',detect_outliers(train, ['LogSalePrice']))
train.drop(['LogSalePrice'], axis=1, inplace=True)
fig, ax = plt.subplots(1, 2, figsize=(20,6))
sns.scatterplot(train.MoSold, train.SalePrice, ax=ax[0])
sns.scatterplot(train.YrSold, train.SalePrice, ax=ax[1])
ax[0].set_xticks(train.MoSold.unique())
ax[1].set_xticks(train.YrSold.unique())
plt.show()
plt.figure(figsize=(17,6))
sns.swarmplot(train.YearBuilt, train.SalePrice)
plt.xticks(rotation=90)

train[['SalePrice', 'YearBuilt']].corr()
plt.figure(figsize=(8,7))
sns.boxplot(train.Street, train.SalePrice)

print('Street value counts\n')
print(train.Street.value_counts())
plt.figure(figsize=(8,7))

sns.boxplot(train.Utilities, train.SalePrice)
print(train.Utilities.value_counts())
plt.figure(figsize=(10,7))

sns.boxplot(train.PoolQC.fillna('No Pool'), train.SalePrice)
print(train.PoolQC.value_counts())
plt.figure(figsize=(10,7))

sns.boxplot(train.Fireplaces, train.SalePrice)
print(train.Fireplaces.value_counts())
num_features = [f for f in train.columns if train[f].dtype != 'O']
numerical = ['SalePrice','LotArea','OverallQual','MasVnrArea','TotalBsmtSF','1stFlrSF','GrLivArea','FullBath','BsmtFullBath','GarageArea','PoolArea']

sns.pairplot(train[numerical])
plt.show()
plt.subplots(figsize = (20,15))
mask = np.zeros_like(train[num_features[1:(len(num_features)-1)]].corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True


sns.heatmap(train[num_features[1:(len(num_features)-1)]].corr(), mask = mask, annot=True, center = 0, cmap='Reds', fmt='.2f')

d = train.corr()[(train.corr().index=='SalePrice')].T.reset_index()
d[d.SalePrice > 0.6].reset_index(drop=True).iloc[:6].sort_values(by='SalePrice', ascending=False).reset_index(drop=True)
print('High Correlations Between Features\n\n')

v=[]
for i in train[num_features[1:(len(num_features)-1)]]:
    for idx,j in enumerate(train.corr()[i][train.corr()[i].values > 0.7]):
        if j<1 and j not in v:
            print(i, '-', train.corr()[i][train.corr()[i].values > 0.7].index[idx],': ', j,'\n')
            v.append(j)
data = pd.concat([train, test], axis=0).reset_index(drop=True)
data.tail()
nulls = pd.Series(data.isna().sum()[(data.isna().sum()>0) & (data.isna().sum().index !='SalePrice')].sort_values(ascending=False).values, index=data.isna().sum()[(data.isna().sum()>0) & (data.isna().sum().index !='SalePrice')].sort_values(ascending=False).index)
dtypes = pd.Series([data[i].dtype for i in nulls.index], index=nulls.index)

nulld = pd.concat([nulls, dtypes], axis=1, keys=['# of Nulls', 'Dtype'])
nulld
idx= []
idx.extend(data[data.GarageType.isnull()].index)
idx2 = []
idx2.extend(data[data.GarageCond.isnull()].index)
print(set(idx2) - set(idx))

data[['GarageType','GarageFinish','GarageYrBlt','GarageCond','GarageYrBlt']].iloc[[2126, 2576]]
for i in nulld[(nulld.Dtype == 'object')].index:
    
    if  nulld.loc[i]['# of Nulls']>20:
        data[i] = data[i].fillna('None')
        
    else:
        data[i] = data[i].fillna(data[i].mode()[0])
for i in data[data.LotFrontage.isna()].index:
    data.LotFrontage.iloc[i] = data.LotFrontage[data.Neighborhood == data.Neighborhood.iloc[i]].median()
    
for i in nulld[(nulld.Dtype != 'object')].index:
    data[i] = data[i].fillna(data[i].mode()[0])
print('Number of Missing Values: ',data[~data.SalePrice.isna()].isna().sum().values.sum())
data['TotalSF'] = data['1stFlrSF'] + data['2ndFlrSF'] + data['TotalBsmtSF']
data['TotalBath'] = data.BsmtFullBath + data.BsmtHalfBath + data.FullBath + data.HalfBath
data['TotalPorchSF'] = data['OpenPorchSF'] + data['3SsnPorch'] + data['EnclosedPorch'] + data['ScreenPorch'] + data['WoodDeckSF']

data['hasPool'] = [1 if i >0 else 0 for i in data['PoolArea']]
data['has2ndfloor'] = [1 if i >0 else 0 for i in data['2ndFlrSF']]
data['hasGarage'] = [1 if i >0 else 0 for i in data['GarageArea']]
data['hasBsmt'] = [1 if i >0 else 0 for i in data['TotalBsmtSF']]
data['hasFireplace'] = [1 if i >0 else 0 for i in data['Fireplaces']]
for i in ['MSSubClass', 'YrSold', 'MoSold']:
    data[i] = data[i].apply(lambda x: str(x))
test_id = data[data.SalePrice.isna()]['Id']

data.drop(['Street', 'Utilities', 'Id'], axis=1, inplace=True)
data = pd.get_dummies(data).reset_index(drop=True)
# test
test = data[data.SalePrice.isna()].drop(['SalePrice'], axis=1)

# train
train = data[~data.SalePrice.isna()]
[{i:train[i].value_counts().iloc[0]} for i in train.columns if train[i].value_counts().iloc[0] == len(train)]
# drop -->  MSSubClass_150

train = train.drop(['MSSubClass_150'], axis=1)
test  = test.drop(['MSSubClass_150'], axis=1)
x = train[~train.SalePrice.isna()].drop(['SalePrice'], axis=1)
y = train[~train.SalePrice.isna()].SalePrice

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=.3, random_state=3)
def gbrf(learning_rate, n_estimators, max_depth):
    params = dict(learning_rate=learning_rate, n_estimators=int(n_estimators), max_depth=int(max_depth))
    
    gbr = GradientBoostingRegressor()
    return min(cross_val_score(gbr, x, y, cv=StratifiedKFold(5, shuffle=True, random_state=3), scoring='neg_root_mean_squared_error'))
bounds = {'n_estimators':(100,3000), 'max_depth':(2,5), 'learning_rate':(0,1)}

optimizer = BayesianOptimization(f= gbrf,
                                 pbounds= bounds,
                                 random_state=3)

optimizer.maximize(n_iter=10, init_points=8, acq='ei')
gbr_params = optimizer.max
print('Optimum parameters for Gradient Boost\n\n',gbr_params['params'])
# import xgboost as xgb

# def black_box_function(n_estimators, max_depth, learning_rate):
#     params = dict(n_estimators= int(n_estimators), max_depth= int(max_depth), learning_rate= learning_rate, eval_metrics='rmse')
#     model = xgb.XGBModel(objective='reg:squarederror', **params)

#     model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=0)
#     return -model.evals_result()['validation_0']['rmse'][-1]
def xgbf(n_estimators, gamma, max_depth, learning_rate):
    d_train = xgb.DMatrix(x, y)
    params = dict(n_estimators= int(n_estimators), gamma= gamma, max_depth= int(max_depth), learning_rate= learning_rate, eval_metrics='rmse')
    
    return -xgb.cv(dtrain = d_train, params= params, folds=StratifiedKFold(5, shuffle=True, random_state=3))['test-rmse-mean'].iloc[-1]
bounds = {'n_estimators':(100,3000), 'gamma':(0,1), 'max_depth':(2,5), 'learning_rate':(0,1)}

optimizer = BayesianOptimization(f= xgbf,
                                 pbounds= bounds,
                                 random_state=3)

optimizer.maximize(n_iter=8, init_points=5, acq='ei')
xgb_params = optimizer.max
print('Optimum parameters for XG Boost\n\n',xgb_params['params'])
def lgbf(learning_rate, max_depth, num_leaves, n_estimators, reg_lambda):
    
    train_data = lgb.Dataset(x, label=y)
    params = dict(learning_rate= learning_rate, max_depth=int(max_depth), num_leaves=int(num_leaves), n_estimators=int(n_estimators), reg_lambda=int(reg_lambda), metrics='rmse', objective='regression')
    
    return -lgb.cv(train_set=train_data, params=params, nfold=5)['rmse-mean'][-1]
bounds = dict(learning_rate= (0,1), max_depth=(-1,3), num_leaves=(20,50), n_estimators=(100,3000), reg_lambda=(0,3))

optimizer = BayesianOptimization(f= lgbf,
                                 pbounds= bounds,
                                 random_state=3)

optimizer.maximize(n_iter=8, init_points=5, acq='ei')
lgb_params = optimizer.max
print('Optimum parameters for LightGBM\n\n',lgb_params['params'])
from catboost import Pool, cv


def cbf(learning_rate, reg_lambda):
    
    cv_data = Pool(data=x, label=y)
    params = dict(learning_rate= learning_rate, reg_lambda=reg_lambda, loss_function ='RMSE', verbose=False, eval_metric='RMSE', iterations=400)
    
    return -cv(cv_data, params, fold_count=5, plot=False)['test-RMSE-mean'].iloc[-1]
bounds = dict(learning_rate= (0,1), reg_lambda=(2,5))

optimizer = BayesianOptimization(f= cbf,
                                 pbounds= bounds,
                                 random_state=3)

optimizer.maximize(n_iter=8, init_points=5, acq='ei')
cb_params = optimizer.max
print('Optimum parameters for CatBoost\n\n',cb_params['params'])
final_estimators = [Lasso(), Ridge(), ElasticNet(), LinearSVR(), RandomForestRegressor()]

estimators = [('xgb', XGBRegressor(n_estimators=2981, learning_rate=0.5002050589158064, gamma=0.20137108244848478)),
              ('gbc', GradientBoostingRegressor(n_estimators=554, max_depth=3, learning_rate=0.6931379183129963)),
              ('lgb', LGBMRegressor(n_estimators=1982, num_leaves=28, reg_lambda=2.028764705940394, learning_rate=0.029876210878566956)),
              ('catb', CatBoostRegressor(learning_rate=0.051867793738858414, reg_lambda=3.3120792921610147, verbose=False))]

for i in final_estimators:
    sc = StackingRegressor(estimators = estimators,
                           final_estimator= i,
                           cv = StratifiedKFold(5, shuffle=True, random_state=3)).fit(x_train, y_train)
    
    print('Meta Regressor: ',i, '\n\n')
    print('R2 Score = ', sc.score(x_val, y_val))
    print('RMSE = ', np.sqrt(mean_squared_error(y_val, sc.predict(x_val))), '\n\n')
def blend_models(x_train, y_train, x_val):
    sr_r = StackingRegressor(estimators = estimators,final_estimator= Ridge()).fit(x_train, y_train).predict(x_val) * 0.3
    se_r = StackingRegressor(estimators = estimators,final_estimator= ElasticNet()).fit(x_train, y_train).predict(x_val) * 0.3
    cb_r = CatBoostRegressor(learning_rate=0.051867793738858414, reg_lambda=3.3120792921610147, verbose=False).fit(x_train, y_train).predict(x_val) * 0.3
    lg_r = LGBMRegressor(n_estimators=1982, num_leaves=28, reg_lambda=2.028764705940394, learning_rate=0.029876210878566956).fit(x_train, y_train).predict(x_val) * 0.1
    
    return sr_r + se_r + lg_r + cb_r
    
    
print('RMSE:', np.sqrt(mean_squared_error(y_val, blend_models(x_train, y_train, x_val))))
print()
print('R2 Score: ', r2_score(y_val, blend_models(x_train, y_train, x_val)))
test_predictions = pd.Series(blend_models(x, y, test), name='SalePrice')

ids = test_id.reset_index(drop=True)
results = pd.concat([ids, test_predictions], axis=1)
results.to_csv("houseprices.csv", index = False)