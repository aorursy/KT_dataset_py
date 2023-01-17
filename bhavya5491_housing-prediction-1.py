# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

from scipy import stats

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory





train = pd.read_csv('../input/train.csv')

train.info()



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt



sns.heatmap(data=train.corr())

plt.gcf().clear

plt.show()

train.skew()
train_d = train.copy()

train_d = pd.get_dummies(train_d)

train_d.info()
keep_cols = train_d.select_dtypes(include=['number']).columns



train_d = train_d[keep_cols]

train_d.describe()
train_d = train_d.fillna(train_d.mean())

train_d.head()
test = pd.read_csv("../input/test.csv")

test.head()
test_d = test.copy()

test_d = pd.get_dummies(test_d)

test_d.info()
test_d = test_d.fillna(test_d.mean())
for col in keep_cols:

    if col not in test_d:

        test_d[col] = 0

test_d.info()
test_d = test_d[keep_cols]

test_d.info()
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV
rf_test = RandomForestRegressor(max_depth=30, n_estimators=500, max_features = 100, oob_score=True, random_state=1234)

cv_score = cross_val_score(rf_test, train_d.drop('SalePrice', axis = 1), train_d['SalePrice'], cv = 10, n_jobs = -1)

print('CV Score is: '+ str(np.mean(cv_score)))
train_0 = train.copy()

null_index = train_0.LotFrontage.isnull()
train_0.loc[null_index,'LotFrontage'] = 0
train_0 = pd.get_dummies(train_0)

keep_cols = train_0.select_dtypes(include=['number']).columns

train_0 = train_0[keep_cols]

train_0 = train_0.fillna(train_0.mean())
rf_test = RandomForestRegressor(max_depth=30, n_estimators=500, max_features = 100, oob_score=True, random_state=1234)

cv_score = cross_val_score(rf_test, train_0.drop('SalePrice', axis = 1), train_0['SalePrice'], cv = 10, n_jobs = -1)

print('CV Score is: '+ str(np.mean(cv_score)))
sns.barplot(data=train,x='Neighborhood',y='LotFrontage', estimator=np.median)

plt.xticks(rotation=90)

plt.show()

gb_neigh = train['LotFrontage'].groupby(train['Neighborhood'])

for i in gb_neigh:

    print (i)
train_LFm = train.copy()
for key,group in gb_neigh:

    # find where we are both simultaneously missing values and where the key exists

    lot_f_nulls_nei = train['LotFrontage'].isnull() & (train['Neighborhood'] == key)

    # fill in those blanks with the median of the key's group object

    train_LFm.loc[lot_f_nulls_nei,'LotFrontage'] = group.median()
train_LFm = pd.get_dummies(train_LFm)

keep_cols = train_LFm.select_dtypes(include=['number']).columns

train_LFm = train_LFm[keep_cols]
train_LFm = train_LFm.fillna(train_LFm.mean())

test_LFm = test.copy()

for key,group in gb_neigh:

    # find where we are both simultaneously missing values and where the key exists

    lot_f_nulls_nei = test['LotFrontage'].isnull() & (test['Neighborhood'] == key)

    # fill in those blanks with the median of the key's group object

    test_LFm.loc[lot_f_nulls_nei,'LotFrontage'] = group.median()




test_LFm = test_LFm.fillna(test_LFm.mean())



for col in keep_cols:

    if col not in test_LFm:

        test_LFm[col] = 0

test_LFm.info()



test_LFm = test_LFm[keep_cols]
from xgboost.sklearn import XGBRegressor
xgb_test = XGBRegressor(learning_rate=0.05,n_estimators=1000,max_depth=3,colsample_bytree=0.4)

cv_score = cross_val_score(xgb_test, train_LFm.drop(['SalePrice','Id'], axis = 1), train_LFm['SalePrice'], cv = 15, n_jobs = -1)
print('CV Score is: '+ str(np.mean(cv_score)))
has_rank = [col for col in train if 'TA' in list(train[col])]

print (has_rank)
#preds = xgb_test.predict(test_LFm.drop('SalePrice', axis = 1))

#out_preds = pd.DataFrame()

#out_preds['Id'] = test['Id']

#out_preds['SalePrice'] = preds

#out_preds.to_csv('output.csv', index=False)
dic_num = {'None':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}
train_c2n = train.copy()
train_c2n['MSSubClass'] = train_c2n['MSSubClass'].astype('category')
for key,group in gb_neigh:

    # find where we are both simultaneously missing values and where the key exists

    lot_f_nulls_nei = train['LotFrontage'].isnull() & (train['Neighborhood'] == key)

    # fill in those blanks with the median of the key's group object

    train_c2n.loc[lot_f_nulls_nei,'LotFrontage'] = group.median()
for col in has_rank:

    train_c2n[col+'_2num'] = train_c2n[col].map(dic_num)

    

train_c2n = pd.get_dummies(train_c2n)
train_cols = train_c2n.select_dtypes(include=['number']).columns

train_c2n = train_c2n[train_cols]
train_c2n = train_c2n.fillna(train_c2n.median())
xgb_test = XGBRegressor(learning_rate=0.05,n_estimators=500,max_depth=3,colsample_bytree=0.4)

cv_score = cross_val_score(xgb_test, train_c2n.drop(['SalePrice','Id'], axis = 1), train_c2n['SalePrice'], cv = 10, n_jobs=-1)
print('CV Score is: '+ str(np.mean(cv_score)))
from statistics import mode
low_var_cat = [col for col in train.select_dtypes(exclude=['number']) if 1 - sum(train[col] == mode(train[col]))/len(train) < 0.03]

low_var_cat
train_col = train.copy()
train_col = train_col.drop(['Street', 'Utilities', 'Condition2', 'RoofMatl', 'Heating'], axis = 1)
train_col['MSSubClass'] = train_col['MSSubClass'].astype('category')
for key,group in gb_neigh:

    # find where we are both simultaneously missing values and where the key exists

    lot_f_nulls_nei = train['LotFrontage'].isnull() & (train['Neighborhood'] == key)

    # fill in those blanks with the median of the key's group object

    train_col.loc[lot_f_nulls_nei,'LotFrontage'] = group.median()
for col in has_rank:

    train_col[col+'_2num'] = train_col[col].map(dic_num)
train_col = pd.get_dummies(train_col)
train_cols = train_col.select_dtypes(include=['number']).columns

train_col = train_col[train_cols]
train_col = train_col.fillna(train_col.median())
xgb_test = XGBRegressor(learning_rate=0.05,n_estimators=500,max_depth=4,colsample_bytree=0.4)

cv_score = cross_val_score(xgb_test, train_col.drop(['SalePrice','Id'], axis = 1), train_col['SalePrice'], cv = 15, n_jobs=-1)
print('CV Score is: '+ str(np.mean(cv_score)))
cat_hasnull = [col for col in train.select_dtypes(['object']) if train[col].isnull().any()]

cat_hasnull
cat_hasnull.remove('Electrical')
mode_elec = mode(train['Electrical'])

mode_elec
cat_hasnull = [col for col in train.select_dtypes(['object']) if train[col].isnull().any()]
cat_hasnull.remove('Electrical')
train_none = train.copy()
train_none = train_none.drop(['Street', 'Utilities', 'Condition2', 'RoofMatl', 'Heating'], axis = 1)
train_none['MSSubClass'] = train_none['MSSubClass'].astype('category')
for col in cat_hasnull:

    null_idx = train_none[col].isnull()

    train_none.loc[null_idx, col] = 'None'
null_idx_el = train_none['Electrical'].isnull()

train_none.loc[null_idx_el, 'Electrical'] = 'SBrkr'
for key,group in gb_neigh:

    # find where we are both simultaneously missing values and where the key exists

    lot_f_nulls_nei = train['LotFrontage'].isnull() & (train['Neighborhood'] == key)

    # fill in those blanks with the median of the key's group object

    train_none.loc[lot_f_nulls_nei,'LotFrontage'] = group.median()
for col in has_rank:

    train_none[col+'_2num'] = train_none[col].map(dic_num)
train_none = pd.get_dummies(train_none)
train_cols = train_none.select_dtypes(include=['number']).columns

train_none = train_none[train_cols]
train_none = train_none.fillna(train_none.median())
xgb_test = XGBRegressor(learning_rate=0.05,n_estimators=500,max_depth=3,colsample_bytree=0.4)

cv_score = cross_val_score(xgb_test, train_none.drop(['SalePrice','Id'], axis = 1), train_none['SalePrice'], cv = 15, n_jobs=-1)
print('CV Score is: '+ str(np.mean(cv_score)))
cols_skew = [col for col in train_none if '_2num' in col or '_' not in col]

train_none[cols_skew].skew()
cols_skew2 = [col for col in train_col if '_2num' in col or '_' not in col]

train_col[cols_skew].skew()
cols_unskew2 = train_col[cols_skew].columns[abs(train_col[cols_skew].skew()) > 1]

cols_unskew = train_none[cols_skew].columns[abs(train_none[cols_skew].skew()) > 1]
train_unskew = train_none.copy()

train_unskew2 = train_col.copy()
for col in cols_unskew:

    train_unskew[col] = np.log1p(train_none[col])

    

for col in cols_unskew:

    train_unskew2[col] = np.log1p(train_col[col])

    

keep_cols = train_unskew.select_dtypes(include=['number']).columns

train_unskew = train_unskew[keep_cols]



test_d = test.copy()

test_d = pd.get_dummies(test_d)

test_d = test_d.fillna(test_d.mean())



for col in keep_cols:

    if col not in test_d:

        test_d[col] = 0

        

test_d = test_d[keep_cols]
xgb_test = XGBRegressor(learning_rate=0.05,n_estimators=500,max_depth=3,colsample_bytree=0.4)

cv_score = cross_val_score(xgb_test, train_unskew.drop(['SalePrice','Id'], axis = 1), train_unskew['SalePrice'], cv = 5, n_jobs=-1)
print('CV Score is: '+ str(np.mean(cv_score)))
xgb_test.fit(train_unskew.drop('SalePrice', axis = 1), train_unskew['SalePrice'])

preds = xgb_test.predict(test_d.drop('SalePrice', axis = 1))

out_preds = pd.DataFrame()

out_preds['Id'] = test['Id']

out_preds['SalePrice'] = preds

out_preds.to_csv('output-unskew.csv', index=False)