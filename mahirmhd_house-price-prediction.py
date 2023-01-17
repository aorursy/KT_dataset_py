import pandas as pd

import numpy as np

import xgboost as xgb

import matplotlib.pyplot as plt

import math



from sklearn.model_selection import train_test_split

from pandas.api.types import is_string_dtype, is_numeric_dtype



%matplotlib inline
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
def display_all(df):

    with pd.option_context('display.max_rows', 1000, 'display.max_columns', 1000):

        display(df)
display_all(train.head().T)
display_all(train.describe(include='all').T)
train = train.drop('Id', axis=1)

Id = test.Id

test = test.drop('Id', axis=1)
train.shape, test.shape
train.info()
test.info()
display_all(test.isna().sum().count)
def train_cats(df):

    for n, c in df.items():

        if is_string_dtype(c): df[n] = c.astype('category').cat.as_ordered()



def apply_cats(train, df):

    for n, c in df.items():

        if (n in train.columns) and (train[n].dtype.name == 'category'):

            df[n] = c.astype('category').cat.as_ordered()

            df[n].cat.set_categories(train[n].cat.categories, ordered=True, inplace=True)
train_cats(train)

apply_cats(train, test)
def save(train, test):

    train.to_feather('train-raw')

    test.to_feather('test-raw')



def load():

    train = pd.read_feather('train-raw')

    test = pd.read_feather('test-raw')

    return train, test
save(train, test)
train, test = load()
def numericalize(df, col, name, max_n_cats):

    if not is_numeric_dtype(col) and (max_n_cats is None or len(col.cat.categories) > max_n_cats):

        df[name] = col.cat.codes + 1

    

def fix_missing(df, col, name, na_dict):

    if is_numeric_dtype(col):

        if pd.isnull(col).sum():

            df[name + '_na'] = pd.isnull(col)

            filler = na_dict[name] if name in na_dict else col.median()

            df[name] = col.fillna(filler)

            na_dict[name] = filler

    return na_dict

            

def proc_df(df, y_fld=None, na_dict=None, max_n_cats=None):

    

    df = df.copy()

    

    if na_dict is None: na_dict = {}

    na_dict_initial = na_dict.copy()

    

    if y_fld is None: y = None

    else : 

        y = df[y_fld].values

        df.drop(y_fld, axis=1, inplace=True)

        

    for n, c in df.items(): na_dict = fix_missing(df, c, n, na_dict)

    if len(na_dict_initial.keys()) > 0:

        df.drop([a+'_na' for a in list(set(na_dict.keys()) - set(na_dict_initial.keys()))], axis=1, inplace=True)

        

    for n, c in df.items(): numericalize(df, c, n, max_n_cats)

        

    df = pd.get_dummies(df, dummy_na=True)

    

    return [df, y, na_dict]
train_df, y, nas = proc_df(train, 'SalePrice', max_n_cats=7)

test_df, _ , _ = proc_df(test, na_dict=nas, max_n_cats=7)
#x_train, x_valid, y_train, y_valid = train_test_split(train, y, test_size=0.2)



def split(df,n): return df[:n].copy(),df[n:].copy()



n_valid = 400

n_train = len(train_df) - n_valid

x_train, x_valid = split(train_df, n_train)

y_train, y_valid = split(y, n_train)

raw_train, raw_valid = split(train, n_train)
x_train.shape, x_valid.shape, y_train.shape, y_valid.shape, raw_train.shape
display_all(x_train.head().T)
def rmse(x, y): return math.sqrt(((np.log(x)-np.log(y))**2).mean())



def print_score(m, name='RandomForest', has_valid=True):

    res = [rmse(m.predict(x_train), y_train), m.score(x_train, y_train)]

    

    if has_valid:

        res.append(rmse(m.predict(x_valid), y_valid))

        res.append(m.score(x_valid, y_valid))

    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)

    

    print (f'{name} : {res}')
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor



def train(x, y):

    rf = RandomForestRegressor(n_estimators=100, bootstrap=True, oob_score=True, min_samples_leaf=1,max_features=0.5, max_depth=35, min_samples_split=3 , random_state=42)

#     br = BaggingRegressor(n_estimators=100, bootstrap=True, oob_score=True, max_features=0.5, random_state=42)

#     xg = xgb.XGBRegressor(n_estimators=300,max_depth=3, min_child_weight=5,random_state=42)



    rf.fit(x, y)

#     br.fit(x_train, y_train)

#     xg.fit(x_train, y_train)



    print_score(rf ,'rf')

#     print_score('br' ,br)

#     print_score('xg' ,xg)

    

    return rf
m = train(x_train, y_train)

print_score(m)
def feat_imp(m, df): return pd.DataFrame({'cols':df.columns,'imp':m.feature_importances_}

                                        ).sort_values('imp', ascending=False)
fi = feat_imp(m[0], x_train)

fi[:10]
fi.plot('cols','imp', figsize=(14,7), legend=False)
fi = feat_imp(m, x_train)

fi[:30].plot('cols', 'imp', 'barh', figsize=(10, 6))
to_keep = fi[fi.imp > 0.003].cols; 

print(len(to_keep))
df_keep = train_df[to_keep].copy()

test_keep = test_df[to_keep].copy()

x_train, x_valid = split(df_keep, n_train)
m = train(x_train, y_train)
fi = feat_imp(m, x_train)

fi[:30].plot('cols', 'imp', 'barh', figsize=(10, 6))
raw_train[['OverallQual', 'SalePrice']].groupby('OverallQual').mean().plot(figsize=(10, 5))
raw_train.plot('GrLivArea','SalePrice',kind='scatter', figsize=(11, 7))
# raw_train.plot('GarageArea','GarageCars',kind='scatter', figsize=(11, 7))

raw_train[['GarageCars', 'GarageArea']].groupby('GarageCars').mean().plot(figsize=(10, 5))
raw_train.plot('GarageArea','SalePrice',kind='scatter', figsize=(11, 7))
raw_train[['GarageCars', 'SalePrice']].groupby('GarageCars').mean().plot(figsize=(10, 5))
to_drop = ['GarageArea', 'GarageCars']

for i in to_drop:

    print ('Dropping',i)

    m = train(x_train.drop(i, axis=1), y_train)

    print ('Score',rmse(m.predict(x_valid.drop(i, axis=1)), y_valid))
x_train.drop('GarageArea', axis=1, inplace=True)

x_valid.drop('GarageArea', axis=1, inplace=True)

test_keep.drop('GarageArea', axis=1, inplace=True)

m = train(x_train, y_train)
raw_train.plot('TotalBsmtSF','SalePrice',kind='scatter', figsize=(11, 7))
x_train.TotalBsmtSF.max()
ma = x_train.TotalBsmtSF.max()

mi = x_train.TotalBsmtSF.min()

a = (x_train.TotalBsmtSF - mi)/(ma-mi)

a[:5]
maxi = train_df.TotalBsmtSF.max()

mini = train_df.TotalBsmtSF.min()

x_train.TotalBsmtSF = (x_train.TotalBsmtSF - mini) / (maxi - mini)

x_valid.TotalBsmtSF = (x_valid.TotalBsmtSF - mini) / (maxi - mini)

test_keep.TotalBsmtSF = (test_keep.TotalBsmtSF - mini) / (maxi - mini)
m = train(x_train, y_train)
raw_train.plot('YearBuilt','SalePrice',kind='scatter', figsize=(11, 7))
raw_train[['YearBuilt', 'SalePrice']].groupby('YearBuilt').mean().plot(figsize=(10, 5))
rf = m[0]

pred = np.stack([t.predict(x_valid) for t in rf.estimators_])
x = raw_valid.copy()

x['pred_std'] = np.std(pred, axis=0)

x['pred'] = np.mean(pred, axis=0)
def analyze(name, df):

    flds = [name, 'SalePrice', 'pred', 'pred_std']

    ana = df[flds].groupby(name, as_index=False).mean()

    return ana
m.predict(test_keep)
Submission = pd.DataFrame({'Id':Id, 'SalePrice':m.predict(test_keep)})

Submission.to_csv('RandomForest.csv', index=False)