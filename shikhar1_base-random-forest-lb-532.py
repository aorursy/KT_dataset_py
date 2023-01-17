%load_ext autoreload

%autoreload 2

%matplotlib inline



#from fastai libraries

# from fastai.imports import *

# from fastai.structured import *



import pandas as pd

# from pandas_summary import DataFrameSummary

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from IPython.display import display

from sklearn import metrics

from sklearn.model_selection import train_test_split

# from treeinterpreter import treeinterpreter as ti

pd.set_option('display.float_format', lambda x: '%.5f' % x)

from scipy.cluster import hierarchy as hc

import os

import numpy as np
types_dict_train = {'train_id': 'int64',

             'item_condition_id': 'int8',

             'price': 'float64',

             'shipping': 'int8'}
train = pd.read_csv('../input/train.tsv',delimiter='\t',low_memory=True,dtype=types_dict_train)
types_dict_test = {'test_id': 'int64',

             'item_condition_id': 'int8',

             'shipping': 'int8'}
test = pd.read_csv('../input/test.tsv',delimiter='\t',low_memory= True,dtype=types_dict_test)
train.head()
test.head()
train.shape,test.shape #
def display_all(df):

    with pd.option_context("display.max_rows", 1000): 

        with pd.option_context("display.max_columns", 1000): 

            display(df)
display_all(train.describe(include='all').transpose())
# train_cats(train)
# train_cats(test)
train.category_name = train.category_name.astype('category')

train.item_description = train.item_description.astype('category')



train.name = train.name.astype('category')

train.brand_name = train.brand_name.astype('category')
test.category_name = test.category_name.astype('category')

test.item_description = test.item_description.astype('category')



test.name = test.name.astype('category')

test.brand_name = test.brand_name.astype('category')
train.dtypes
test.dtypes
train.apply(lambda x: x.nunique())
test.apply(lambda x: x.nunique())
train.isnull().sum(),train.isnull().sum()/train.shape[0]
test.isnull().sum(),test.isnull().sum()/test.shape[0]
os.makedirs('data/tmp',exist_ok=True)
# train.to_feather('data/tmp/train_raw')



# test.to_feather('data/tmp/test_raw')
# train = pd.read_feather('data/tmp/train_raw')

# test = pd.read_feather('data/tmp/test_raw')
train = train.rename(columns = {'train_id':'id'})
train.head()
test = test.rename(columns = {'test_id':'id'})
test.head()
train['is_train'] = 1

test['is_train'] = 0
train_test_combine = pd.concat([train.drop(['price'],axis =1),test],axis = 0)
train_test_combine.category_name = train_test_combine.category_name.astype('category')

train_test_combine.item_description = train_test_combine.item_description.astype('category')



train_test_combine.name = train_test_combine.name.astype('category')

train_test_combine.brand_name = train_test_combine.brand_name.astype('category')
train_test_combine = train_test_combine.drop(['item_description'],axis = 1)
train_test_combine.name = train_test_combine.name.cat.codes
train_test_combine.category_name = train_test_combine.category_name.cat.codes
train_test_combine.brand_name = train_test_combine.brand_name.cat.codes
# train_test_combine.item_description = train_test_combine.item_description.cat.codes
train_test_combine.head()
train_test_combine.dtypes
df_test = train_test_combine.loc[train_test_combine['is_train']==0]

df_train = train_test_combine.loc[train_test_combine['is_train']==1]
df_test = df_test.drop(['is_train'],axis=1)
df_train = df_train.drop(['is_train'],axis=1)
df_test.shape
df_train.shape
df_train['price'] = train.price
df_train['price'] = df_train['price'].apply(lambda x: np.log(x) if x>0 else x)
df_train.head()
# df_train.to_feather('data/tmp/train_raw_pro')
# df_test.to_feather('data/tmp/test_raw_pro')
# df_train = pd.read_feather('data/tmp/train_raw_pro')

# df_test = pd.read_feather('data/tmp/test_raw_pro')
x_train,y_train = df_train.drop(['price'],axis =1),df_train.price
# reset_rf_samples()
m = RandomForestRegressor(n_jobs=-1,min_samples_leaf=3,n_estimators=200)

m.fit(x_train, y_train)

m.score(x_train,y_train)
preds = m.predict(df_test)
preds = pd.Series(np.exp(preds))
type(preds)
submit = pd.concat([df_test.id,preds],axis=1)
submit.columns = ['test_id','price']
submit.to_csv("./rf_v3.csv", index=False)