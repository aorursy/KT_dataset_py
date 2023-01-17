import pandas as pd

import numpy as np

import pickle

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')



pd.set_option('max_columns', None) 
pwd
df = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
df.head(30)
test.head()
df.shape, test.shape # target : price
df.info() # null 0, date - object
test.info() # null 0, date - object
df['buy_ym'] = df['date'].apply(lambda x : x[0:6])
test['buy_ym'] = test['date'].apply(lambda x : x[0:6])
df['buy_ym'] = df['buy_ym'].astype(int)
test['buy_ym'] = test['buy_ym'].astype(int)
df.buy_ym.max(), df.buy_ym.min()
df.buy_ym.describe()
df.price.describe()
sns.boxplot(data = df['price'] )

fig = plt.gcf()

fig.set_size_inches(10,5)
sns.distplot(df['price'])
print('Skewness : %f' % df.price.skew())

print('Kurtosis : %f' % df.price.kurt())  
cor_mat = df.corr()

mask = np.array(cor_mat)

mask[np.tril_indices_from(mask)] = False



flg = plt.gcf()

plt.figure(figsize = (15,15))

sns.heatmap(data = cor_mat, mask = mask, square = True, annot = True, cbar = True)
data = pd.concat( [df.price, df.grade], axis = 1)

f, ax = plt.subplots(figsize = (8,6))

fig = sns.boxplot(x = 'grade', y = 'price', data = data)
data = pd.concat( [df.price, df.long], axis = 1)

f, ax = plt.subplots(figsize = (8,6))

fig = sns.boxplot(x = 'long', y = 'price', data = data)
data = pd.concat( [df.price, df.buy_ym], axis = 1)

f, ax = plt.subplots(figsize = (8,6))

fig = sns.boxplot(x = 'buy_ym', y = 'price', data = data)
data = pd.concat( [df.price, df.buy_ym], axis = 1)

f, ax = plt.subplots(figsize = (8,6))

fig = sns.boxplot(x = 'buy_ym', y = 'price', data = df)
data = pd.concat([df.price, df.sqft_living], axis = 1)

f, ax = plt.subplots(figsize = (4,3))

fig = sns.regplot(x = 'sqft_living', y = 'price', data = df)
data = pd.concat([df.price, df.sqft_living15], axis = 1)

f, ax = plt.subplots(figsize = (4,3))

fig = sns.regplot(x = 'sqft_living15', y = 'price', data = data)  # sqft_living에 비해 가격 분산이 크다 즉 less specific하다
data = pd.concat([df.price, df.sqft_above], axis = 1)

f, ax = plt.subplots(figsize = (4,3))

fig = sns.regplot(x = 'sqft_above', y = 'price', data = data) 
data = pd.concat([df.price, df.yr_built], axis = 1)

f, ax = plt.subplots(figsize = (4,3))

fig = sns.regplot(x = 'yr_built', y = 'price', data = data) 
data = pd.concat([df.price, df.bathrooms], axis = 1)

f, ax = plt.subplots(figsize = (4,3))

fig = sns.regplot(x = 'bathrooms', y = 'price', data = data) 
data = pd.concat([df.price, df.bedrooms], axis = 1)

f, ax = plt.subplots(figsize = (4,3))

fig = sns.regplot(x = 'bedrooms', y = 'price', data = data) 
data = pd.concat([df.price, df.bathrooms], axis = 1)

f, ax = plt.subplots(figsize = (12,3))

fig = sns.boxplot(x = 'bathrooms', y = 'price', data = data) 
data = pd.concat([df.price, df.bedrooms], axis = 1)

f, ax = plt.subplots(figsize = (12,3))

fig = sns.boxplot(x = 'bedrooms', y = 'price', data = data) 
df.nunique()
df.isnull().sum()
test.isnull().sum()
df.loc[df.sqft_living > 12500, :]
df.sqft_living.describe()
df = df.loc[df.id != 8912] 
df.shape
df.shape
df.isnull().sum()
df.info()
df.date = pd.to_datetime(df.date)

test.date = pd.to_datetime(test.date)
df['buy_year'] = df['date'].dt.year

test['buy_year'] = test['date'].dt.year
df['sqft_total_size'] = df['sqft_above'] + df['sqft_basement']

test['sqft_total_size'] = test['sqft_above'] + test['sqft_basement']

df['total_rooms'] = df['bedrooms'] * df['bathrooms']

test['total_rooms'] = test['bedrooms'] * test['bathrooms']
df.loc[df.yr_renovated == 2014].count()
df.yr_renovated.max(), 
df.loc[df.yr_renovated != 0]
df.query('yr_renovated == 2015').buy_ym.count()
df.query('yr_renovated == 2014').buy_ym.count()
df['yr_built_renovated'] = np.where(df.yr_renovated != 0, df.yr_renovated, df.yr_built)
test['yr_built_renovated'] = np.where(test.yr_renovated != 0, test.yr_renovated, test.yr_built)
df.head(30)
df['yr_built_renovated'].min()
df['after_renovation'] = abs(df.buy_year - df.yr_built_renovated)
test['after_renovation'] = abs(test.buy_year - test.yr_built_renovated)
test.head()
df.after_renovation.describe()
sns.barplot(x = 'after_renovation', y = 'price', data = df)
df['sqft_ratio'] = df.sqft_living / df.sqft_lot

test['sqft_ratio'] = test.sqft_living / test.sqft_lot
df['perprice'] = df.price / df.sqft_living
zipcode_price = df[['zipcode', 'perprice']].groupby(['zipcode']).agg({'mean','var'})
zipcode_price.head()
level0 = zipcode_price.columns.get_level_values(0)

level1 = zipcode_price.columns.get_level_values(1)

zipcode_price.columns = level0 + '_' + level1
# zipcode_price.columns = ["_".join(x) for x in zipcode_price.columns.ravel()]
zipcode_price = zipcode_price.reset_index()
zipcode_price.head()
df = pd.merge(df, zipcode_price, how = 'left', on = 'zipcode')
df.head()
test = pd.merge(test, zipcode_price, how = 'left', on = 'zipcode')
test.head()
a = set(df.columns)

b = set(test.columns)
print(a-b)

print(b-a)
X_train = df.drop(['date', 'price', 'id', 'perprice', 'buy_year', 'yr_built_renovated'], axis = 1)

X_test = test.drop(['date', 'id', 'buy_year', 'yr_built_renovated'], axis = 1)
X_train.head()
X_test.head()
target = df.perprice
X_train_columns = X_train.columns
import lightgbm as lgb

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import KFold, cross_val_score

from sklearn.model_selection import cross_val_predict

from sklearn.preprocessing import RobustScaler

from sklearn.pipeline import make_pipeline



param = {'num_leaves': 8, 

        'min_data_in_leaf': 5, 

         'min_sum_hessian_in_leaf' : 0.001,

         'min_gain_to_split' : 1,

        'objective' : 'regression', 

        'max_depth': -1, 

        'learning_rate': 0.01, 

        'boosting' : 'gbdt', 

        'feature_fraction' : 0.5, 

        'bagging_freq': 5, 

        'bagging_fraction': 0.5, 

        'bagging_seed': 11, 

        'metric': 'rmse', 

        'lambda_l1': 0.1, 

        'lambda_l2': 0.1, 

        'verbosity': -1,

        'random_state': 42}

y_reg = target # target = df.perprice 
# prepare fit model with cross_validation

folds = KFold(n_splits= 10, shuffle= True, random_state = 42)

oof = np.zeros(len(X_train))

pred = np.zeros(len(X_test))

feature_importance_df = pd.DataFrame()
# run model 

for fold_, (trn_idx, val_idx) in enumerate(folds. split(X_train)):

    trn_data = lgb.Dataset(X_train.iloc[trn_idx][X_train_columns],

                           label = y_reg.iloc[trn_idx])

    val_data = lgb.Dataset(X_train.iloc[val_idx][X_train_columns], 

                          label = y_reg.iloc[val_idx])

    num_round = 100000

    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval = 1000, early_stopping_rounds = 3000)

    oof[val_idx] = clf.predict(X_train.iloc[val_idx][X_train_columns], num_iteration = clf.best_iteration)

    

    # feature importance

    fold_importance_df = pd.DataFrame()

    fold_importance_df['Feature'] = X_train.columns

    fold_importance_df['importance'] = clf.feature_importance()

    fold_importance_df['fold'] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis = 0)

    

    # predictions

    pred += clf.predict(X_test[X_train_columns], num_iteration = clf.best_iteration) / folds.n_splits

    

cv = np.sqrt(mean_squared_error(oof, y_reg))

print(cv) 
pred.shape
submission = pd.read_csv('../input/sample_submission.csv')
submission['price'] = pred * X_test.sqft_living
submission.head()
submission.to_csv('submission.csv', index = False)