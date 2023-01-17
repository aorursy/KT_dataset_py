# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
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
types_dict_test = {'test_id': 'int64',
             'item_condition_id': 'int8',
             'shipping': 'int8'}
train = pd.read_csv('../input/train.tsv',delimiter='\t',low_memory=True,dtype=types_dict_train)
test = pd.read_csv('../input/test.tsv',delimiter='\t',low_memory= True,dtype=types_dict_test)
train.head()
test.head()
train.head()
train['is_train'] = 1
test['is_train'] = 0
train = train.rename(columns = {'train_id':'id'})
test = test.rename(columns = {'test_id':'id'})
train_test_combine = pd.concat([train.drop(['price'],axis =1),test],axis = 0)
train_test_combine.category_name = train_test_combine.category_name.astype('category')
train_test_combine.name = train_test_combine.name.astype('category')
train_test_combine.brand_name = train_test_combine.brand_name.astype('category')
train_test_combine.name = train_test_combine.name.cat.codes
train_test_combine.category_name = train_test_combine.category_name.cat.codes
train_test_combine.brand_name = train_test_combine.brand_name.cat.codes
train_test_combine['len_desc'] = train_test_combine['item_description'].str.len()
train_test_combine = train_test_combine.drop(['item_description'],axis = 1)
train_test_combine.head()
train_test_combine['len_desc'] = train_test_combine['len_desc'].fillna(0)
train_test_combine.len_desc = train_test_combine.len_desc.astype('category')
train_test_combine.len_desc = train_test_combine.len_desc.cat.codes
df_test = train_test_combine.loc[train_test_combine['is_train']==0]
df_train = train_test_combine.loc[train_test_combine['is_train']==1]
df_test = df_test.drop(['is_train'],axis=1)
df_train = df_train.drop(['is_train'],axis=1)
df_train['price'] = train.price
df_train['price'] = df_train['price'].apply(lambda x: np.exp(x))
x_train,y_train = df_train.drop(['price'],axis =1),df_train.price
rf2 = RandomForestRegressor(n_jobs=-1, n_estimators=100,  random_state=42, max_features=0.5, min_samples_leaf=3)
rf2.fit(x_train,y_train)
rf2.score(x_train,y_train)
preds = rf2.predict(df_test)
preds = pd.Series(np.log(preds))
submit = pd.concat([test.id,preds],axis=1)
submit.columns = ['test_id','price']
submit.to_csv("./rf_v4.csv", index=False)
