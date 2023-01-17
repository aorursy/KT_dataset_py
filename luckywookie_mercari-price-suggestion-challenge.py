import numpy as np

import pandas as pd



import time

import re

from __future__ import print_function

from collections import defaultdict



import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import matplotlib.cm as cm

import seaborn as sns



from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import make_union, make_pipeline

from sklearn.preprocessing import FunctionTransformer, StandardScaler, LabelEncoder, MinMaxScaler,  Imputer, LabelBinarizer, OneHotEncoder

from sklearn.feature_extraction import DictVectorizer

from sklearn.linear_model import LogisticRegressionCV

from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

from sklearn.model_selection import RandomizedSearchCV, train_test_split



import xgboost as xgb

import lightgbm as lgb



%matplotlib inline

plt.rcParams["figure.figsize"] = (15, 8)

pd.options.display.float_format = '{:.2f}'.format



import os

# print(os.listdir("../input"))



def check_category(row):

    """

    Function for fill empty category to None/None/None

    """

    if isinstance(row.category_name, str) and '/' in row.category_name:

        return row.category_name

    else:

        return 'None/None/None'



def check_brand(row):

    """

    Function for fill empty brand to No brand

    """

    if isinstance(row.brand_name, str) and row.brand_name:

        return row.brand_name

    else:

        return 'No brand'



def change_tables(data, type_sample='train'):

    """

    Function for split category_name to three columns. And drop empty values.

    """

    data['category_name_1'] = data.apply(check_category, axis=1)

    data['brand_name_1'] = data.apply(check_brand, axis=1)

    data['category_1'] = data.apply(lambda row: row.category_name_1.split('/')[0], axis=1)

    data['category_2'] = data.apply(lambda row: row.category_name_1.split('/')[1], axis=1)

    data['category_3'] = data.apply(lambda row: row.category_name_1.split('/')[2], axis=1)

    if 'price' not in data.columns:

        data['price'] = 0

    price = data['price']

    data['brand_name'] = data['brand_name_1']

    data_name_id = 'train_id'

    if type_sample == 'test':

        data_name_id = 'test_id'

    data.drop(labels=[data_name_id, 'category_name', 'price', 'item_condition_id', 'shipping', 'category_name_1', 'brand_name_1'], axis=1, inplace=True)

    data['price'] = price



df_train = pd.read_csv('../input/train.tsv', sep='\t')  # test_stg2.tsv

change_tables(df_train)

print('df_train len:', df_train.shape)

y_train = df_train['price']



df_test = pd.read_csv('../input/test.tsv', sep='\t')

change_tables(df_test, type_sample='test')

print('df_test len:', df_test.shape)
from sklearn.pipeline import make_union, make_pipeline

from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, StandardScaler, LabelEncoder, MinMaxScaler, LabelBinarizer, OneHotEncoder

from sklearn.feature_extraction import DictVectorizer

from sklearn.impute import SimpleImputer, MissingIndicator

    

class LenDescription(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        return self

        

    def transform(self, X, y=None):

        X['descr_length'] = X['item_description'].str.len()

        return X['descr_length'].as_matrix().reshape(-1, 1)



    def fit_transform(self, X, y=None):

        return self.fit(X).transform(X)





class DictDescription(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        return self

        

    def transform(self, X, y=None):

        return (row for _, row in X.iterrows())



    def fit_transform(self, X, y=None):

        return self.fit(X).transform(X)





class CountWords(BaseEstimator, TransformerMixin):

    

    def fit(self, X, y=None):

        return self

        

    def transform(self, X, y=None):

        X['splited_row'] = X.apply(lambda row: len(row.item_description.str.split(' ')), axis=1)

        return X['splited_row'].values



    def fit_transform(self, X, y=None):

        return self.fit(X).transform(X)





def get_brand_cols(df):

    return df[['brand_name']]



def get_category_cols(df):

    return df[['category_name']]



def get_name_cols(df):

    return df[['name']]



def get_descr_cols(df):

    return df[['item_description']]



def get_all_cols(df):

    return df[['name', 'brand_name']]



def get_cat_cols(df):

    return df[['category_1', 'category_2']]



def get_last_cat_cols(df):

    return df[['category_3']]



vec = make_union(*[

    make_pipeline(FunctionTransformer(get_cat_cols, validate=False), OneHotEncoder(sparse=False)),

    make_pipeline(FunctionTransformer(get_brand_cols, validate=False), OrdinalEncoder()),

    make_pipeline(FunctionTransformer(get_last_cat_cols, validate=False), OrdinalEncoder()),

    make_pipeline(FunctionTransformer(get_descr_cols, validate=False), LenDescription()),

#     make_pipeline(FunctionTransformer(get_descr_cols, validate=False), CountWords()),

    make_pipeline(FunctionTransformer(get_descr_cols, validate=False), DictDescription(), DictVectorizer()),

    make_pipeline(FunctionTransformer(get_name_cols, validate=False), OrdinalEncoder()),

])
len_train = df_train.shape[0]

len_test = df_test.shape[0]

X = df_train.append(df_test, ignore_index=True)

x_transform = vec.fit_transform(X)
Y = y_train.append(df_test['price'], ignore_index=True)

indices = np.arange(len_train)

X_train, X_test, Y_train, _ = train_test_split(x_transform, Y, train_size=len_train, shuffle=False)
import scipy.stats as st



param_grid = {

    "nthread": [-1],

    'objective':['reg:linear'],

    "n_estimators": [300, 500],

    "max_depth": st.randint(3, 8),

    "learning_rate": st.uniform(0.05, 0.5),

    "colsample_bytree": st.beta(10, 1),

    "subsample": st.beta(10, 1),

    "gamma": st.uniform(0, 10),

    'reg_alpha': st.expon(0, 50),

    'min_child_weight': [4, 8, 16],

}



xg_reg = RandomizedSearchCV(xgb.XGBRegressor(), param_grid, n_jobs=5, cv=2, verbose=True)

xg_reg.fit(X_train, Y_train)
print(xg_reg.best_params_)

print(xg_reg.best_estimator_)

print(xg_reg.best_score_)
score = xg_reg.score(X_train, Y_train)

print('score:', score)
preds = xg_reg.predict(X_test)



df_res = pd.read_csv('../input/sample_submission.csv')



g = df_res['price']

# print(g.values, type(g.values))

print('mean_squared_error: ', mean_squared_error(g, preds))

print('r2_score: ', r2_score(g, preds))
# X_test = pd.concat(X_test, pd.DataFrame({'price': preds}))

# X_test['test_id'] = X_test['test_id'].astype(np.int)

# X_test[['test_id', 'price']].to_csv('submission.csv')
import lightgbm as lgb



df_res = pd.read_csv('../input/sample_submission.csv')

g = df_res['price']



params = {

    'boosting_type': 'gbdt',

    'objective': 'regression',

    'metric': {'l2', 'l1'},

    'max_depth': 10, 

    'learning_rate': 0.01,

    'verbose': 1, 

    'early_stopping_round': 20}

n_estimators = 500



d_train = lgb.Dataset(X_train, label=Y_train)

d_valid = lgb.Dataset(X_test, label=g)

watchlist = [d_valid]



model = lgb.train(params, d_train, n_estimators, watchlist, verbose_eval=1)

preds = model.predict(X_test, num_iteration=model.best_iteration)

print('mean_squared_error: ', mean_squared_error(g, preds))
# data = X_test.tocoo(copy=False)

# data_x = pd.DataFrame({'index': data.row, 'price': data.price}

#                  )[['index', 'price']].reset_index(drop=True)

# X = pd.concat([data_x, pd.DataFrame({'price': preds})])

# X['test_id'] = X['test_id'].astype(np.int)

# X[['test_id', 'price']].to_csv('submission.csv')