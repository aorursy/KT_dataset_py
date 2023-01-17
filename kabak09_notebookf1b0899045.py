# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from os import listdir

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

files = listdir('../input/')

df = pd.read_csv('../input/train.csv')

pd.set_option('display.max_rows', 200)

def describe (frame):
    return pd.DataFrame({'dtype': frame.dtypes, 
              'isnull': pd.isnull(frame).any(), 
              'std': frame.select_dtypes(exclude=['object']).std(),
              'max': frame.select_dtypes(exclude=['object']).max(),
              'min': frame.select_dtypes(exclude=['object']).min(),
              'mean': frame.select_dtypes(exclude=['object']).mean(),
              'median': frame.select_dtypes(exclude=['object']).median(),
              'count': frame.apply(lambda col: len(col.unique()))}, 
             index=frame.columns, columns=['dtype', 'isnull', 'std', 'max', 'min',
                                        'mean', 'median', 'count']).sort_values('dtype', ascending=True)

describe(df)
## Visualization of Distributions ##

% matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(17, 2, figsize=(12,60))

axes0=[]
for axis_list in axes:
    for axis in axis_list:
        axes0.append(axis)
        
def plot_dist (data):
    if len(data.unique())>10:
        sns.distplot(data, color='orange')
    else:
        sns.countplot(data, color='orange')

        
i=0
for axis in axes0:
    plt.subplot(axis)
    i+=1
    try:
        plot_dist(df.select_dtypes(exclude=['object']).iloc[:,i])
    except:
        print('Null Value in', df.select_dtypes(exclude=['object']).iloc[:,i].name)
        i+=1
        plot_dist(df.select_dtypes(exclude=['object']).iloc[:,i])
    

        

## Fill Null Values ##

desc_df = describe(df)
print(desc_df[(desc_df['isnull']==True)&
         (desc_df['dtype']!='object')].index)

## Plotting Distributions Before Filling NaN Values ##

fig, axes = plt.subplots(1,3, figsize=(15,6))

for axis, col in zip(axes, ['GarageYrBlt', 'LotFrontage', 'MasVnrArea']):
    plt.subplot(axis)
    plot_dist(df.loc[:,col].dropna())

## Fill null values for numeric numeric features ##

df.loc[:,['GarageYrBlt', 'LotFrontage', 'MasVnrArea']] = \
                df.loc[:,['GarageYrBlt', 'LotFrontage', 'MasVnrArea']]\
                                            .apply(lambda col: col.fillna(col.mean()), axis=0)


## Fill null values for categorical features ##

desc_df = desc_df[(desc_df['isnull']==True)&
                  (desc_df['dtype']=='object')].index

desc_df = [col for col in desc_df]

print(desc_df)

df.loc[:,desc_df] = df.loc[:,desc_df].fillna('unknown')


## Check for null values ##

True in pd.isnull(df).any().values
## Feature Engineering ## 
"""
def new_features (row):
    row['AreaIndex'] = row['LotArea']*row['GrLivArea']*row['1stFlrSF']*row['BsmtUnfSF']*row['TotalBsmtSF']
    return row

df = df.apply(new_features, axis=1)
"""
## Check for feature importances using random forest ##

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import TransformerMixin

class VarianceCalculator(TransformerMixin):
    def fit (self, X):
        self.data = X
        self.scaler = MinMaxScaler().fit(X)
        X = self.scaler.transform(X)
        self.var = VarianceThreshold().fit(X)
        return self
    def transform (self, X):
        return self.var.transform(X)

y = df.loc[:, 'SalePrice']
X = df.drop(['SalePrice', 'Id'], axis=1)

X = pd.get_dummies(X)

scaler = MinMaxScaler()
reg = RandomForestClassifier(n_estimators=100)

X_ = scaler.fit_transform(X)
reg.fit(X_, y)
y_pred = reg.predict(X_)

features = pd.Series(reg.feature_importances_, index=X.columns).sort_values(ascending=False)
print(features)
print('train rmse : {}'.format(np.sqrt(mean_squared_error(y, y_pred))))
print('train r2 score : {}'.format(r2_score(y, y_pred)))

##################################################################
variance = VarianceCalculator()      
###################################################################    
features_to_use = list(features[features>0.006].index)
features_to_use = [name.split('_')[0] if '_' in name else name for name in features_to_use]
features_to_use = list(set(features_to_use))+['SalePrice']
print('Using {} Features'.format(len(features_to_use)))
features_to_use
## Clean DataSets for Machine Learning Algorithm ##

train = df.loc[:,features_to_use].sample(500)
test = pd.read_csv('../input/test.csv').loc[:, features_to_use]

desc_test = describe(test)
null_numeric = list(desc_test[(desc_test['isnull']==True)&
                  (desc_test['dtype']!='object')].index)

null_categorical = list(desc_test[(desc_test['isnull']==True)&
                  (desc_test['dtype']=='object')].index)

## fill null values for numeric features ##
test.loc[:,null_numeric] = test.loc[:,null_numeric].fillna(-100000) 

## fill null values for categorical features ##

test.loc[:, null_categorical] = test.loc[:, null_categorical].fillna('unknown')
## Encoding Categorical Features ## 

from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline

categorical_features = list(describe(train)[describe(train)['dtype']=='object'].index)

for col in categorical_features:
    unique_cat = set(train[col].unique())|{'unknown'}
    train['{}_'.format(col)] = pd.Categorical(train[col], categories=unique_cat, ordered=False).codes
    test['{}_'.format(col)] = pd.Categorical(test[col], categories=unique_cat, ordered=False)\
                                                                            .fillna('unknown').codes
        

X_train = train.drop(categorical_features+['SalePrice'], axis=1)
y_train = train['SalePrice']
X_test = test.drop(categorical_features+['SalePrice'], axis=1)
y_test = test['SalePrice']

feature_mask = [True if feature[:-1] in categorical_features else False for feature in X_train.columns]
ohe = OneHotEncoder(categorical_features=feature_mask, handle_unknown='ignore', 
                         sparse=False).fit(X_train)

poly = PolynomialFeatures()
scaler = MinMaxScaler()
## Cross Validation Pipeline ##

from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import validation_curve

def cross_val (func, param_name, param_range, polynomial=False, **kwargs):
    reg = func(**kwargs)
    
    if polynomial==True:
        reg_pipe = Pipeline(steps=[('ohe', ohe), ('norm', scaler), ('poly', poly), ('reg', reg)])
    else:
        reg_pipe = Pipeline(steps=[('ohe', ohe), ('norm', scaler), ('reg', reg)])
    
    print('Cross Validation of {}...'.format(func.__name__))
    train_score, test_score = validation_curve(reg_pipe, X_train, y_train, 
                                               cv=3, param_name='reg__{}'.format(param_name),
                                               scoring='r2',
                                               param_range=param_range)
    
    print('Validation Finished...')
    
    train_max = [np.max(score) for score in train_score]
    train_min = [np.min(score) for score in train_score]
    train_avg = [np.mean(score) for score in train_score]
    
    test_max = [np.max(score) for score in test_score]
    test_min = [np.min(score) for score in test_score]
    test_avg = [np.mean(score) for score in test_score]
    
    plt.plot(param_range, train_avg, color='blue', label='train_score')
    plt.plot(param_range, test_avg, color='orange', label='test_score')
    plt.fill_between(param_range, train_max, train_min, alpha=0.2, color='blue')
    plt.fill_between(param_range, test_max, test_min, alpha=0.2, color='orange')
    plt.legend()
    plt.title(func.__name__)
    plt.xlabel(param_name)
    plt.ylabel('r2 score')
    

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3,2, figsize=(15,20))

plt.subplot(ax1)
cross_val(Lasso, 'alpha', np.logspace(0,2.3,100), tol=0.001, max_iter=1000000)
plt.subplot(ax2)
cross_val(Ridge, 'alpha', np.logspace(-1,2,100))
plt.subplot(ax3)
cross_val(RandomForestRegressor, 'max_depth', list(range(3,30)), n_estimators=200)
plt.subplot(ax4)
cross_val(GradientBoostingRegressor, 'learning_rate', np.logspace(-3,0,100), n_estimators=200)
plt.subplot(ax5)
cross_val(ElasticNet, 'alpha', np.logspace(0.2,2,100), l1_ratio=0.9, tol=0.001, max_iter=1000000)
plt.subplot(ax6)
cross_val(ElasticNet, 'l1_ratio', np.linspace(0,1,20), alpha=1, tol=0.001, max_iter=1000000)
    
## Optimizer Function ##

from sklearn.model_selection import RandomizedSearchCV

def optimizer(func, param_dist, n_iter=40, cv=3, polynomial=False, **kwargs):
    if polynomial == True:
        reg_pipe = Pipeline(steps=[('ohe', ohe), ('norm', scaler), ('poly', poly), ('reg', func(**kwargs))])
    else:
        reg_pipe = Pipeline(steps=[('ohe', ohe), ('norm', scaler), ('reg', func(**kwargs))])
    
    optimizer = RandomizedSearchCV(reg_pipe, param_distributions=param_dist, n_iter=n_iter,
                                   return_train_score=True, scoring='r2', cv=cv)
    optimizer.fit(X_train, y_train)
    results = pd.DataFrame(optimizer.cv_results_)

    return results.loc[:,['mean_train_score', 'mean_test_score']\
                                        +['param_{}'.format(name) for name in param_dist.keys()]]
## Optimizing ElasticNet ##

param_dist = {'reg__l1_ratio':np.linspace(0,1,50), 'reg__alpha':np.logspace(0,2.3,100)}

results = optimizer(func=ElasticNet, param_dist=param_dist,polynomial=True, max_iter=10000000, tol=0.001)\
                                                    .sort_values('mean_test_score', ascending=False)
    
results
## Optimizing GradientBoostingTrees ##

param_dist = {'reg__learning_rate': np.linspace(0.001, 1, 100),
              'reg__n_estimators': list(range(10, 400, 50)),
              'reg__max_depth': [2,3,4,5,6,7,8]}
              #'reg__loss': ['ls', 'lad', 'huber', 'quantile']}

results = optimizer(func=GradientBoostingRegressor, param_dist=param_dist, loss='huber')\
                                .sort_values(['mean_test_score', 
                                              'param_reg__n_estimators'], ascending=False)
    
results
## Scoring Using Test Dataset ##
from sklearn.metrics import r2_score, mean_squared_error

reg = GradientBoostingRegressor(n_estimators=110, max_depth=4, loss='ls')
reg_pipe = Pipeline(steps=[('ohe', ohe), ('norm', scaler), ('reg', reg)])
reg_pipe.fit(X_train, y_train)

submission = pd.DataFrame({'SalePrice': reg_pipe.predict(X_test)}, index=X_test.index)
import os

submission['Id'] = pd.read_csv('../input/test.csv').loc[:, ['Id']]
submission = submission[['Id', 'SalePrice']]
#print(submission.set_index('Id').to_csv())
