# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import csv

import matplotlib.pyplot as plt

import re

# Evalaluation

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.model_selection import cross_val_score

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.preprocessing import LabelBinarizer

import scipy

import lightgbm as lgb

from sklearn.linear_model import Ridge

import time

import gc

from scipy.sparse import csr_matrix, hstack

get_ipython().magic('pylab inline')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

df = pd.read_csv("../input/train.tsv", delimiter='\t')

test = pd.read_csv("../input/test.tsv", delimiter='\t')

Y = df['price']

train_test_split = df.shape[0]

brand = dict()



for i in range(len(df['brand_name'])):

    if(not pd.isnull(df['brand_name'][i])):

        if (df['brand_name'][i] not in brand):

            brand[df['brand_name'][i]] = []

        brand[df['brand_name'][i]].append(np.log1p(df['price'][i]))

        

del df['price']

del df['train_id']

del test['test_id']

df = pd.concat([df,test])

df = df.reset_index(drop=True)

split_str = df["category_name"].str.split("/", expand=True, n=2)

split_str.columns = ["cat1", "cat2", "cat3"]

split_str["cat2"][split_str["cat2"].isnull()] = -1

split_str["cat3"][split_str["cat3"].isnull()] = -1

df["category1"] = df["category_name"]

df["category2"] = split_str["cat2"]

df["category3"] = split_str["cat3"]

df["category1"][df["category1"].isnull()] = -1

df.drop(['category_name'], axis=1, inplace=True)

df



avg_col = []

var_col = []

avg_dict = {}

var_dict = {}

for i in range(len(df['brand_name'])):

    if pd.isnull(df['brand_name'][i]):

        avg_col.append(0)

        var_col.append(0)

    else:

        name = df['brand_name'][i]

        if name not in avg_dict:

            if name not in brand:

                avg_col.append(0)

                var_col.append(0)

                continue

            priceList = brand[name]

            avg_dict[name] = np.mean(priceList)

            var_dict[name] = np.var(priceList)

        avg_col.append(avg_dict[name])

        var_col.append(var_dict[name])

print(len(avg_col), len(var_col))



df["brand_avg"] = avg_col

df["brand_var"] = var_col

df["brand_name"] = pd.Categorical(df["brand_name"])

df["brand_code"] = df.brand_name.cat.codes



def change_to_code(colName, df):

    df[colName] = pd.Categorical(df[colName])

    df[colName+"_code"] = df[colName].cat.codes

for category in ["category1", "category2", "category3"]:

    change_to_code(category, df)

    

df['name'] = df['name'].fillna('missing')

df['item_description'] = df['item_description'].fillna('missing')



from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

count = CountVectorizer(min_df=10)

X_name = count.fit_transform(df["name"])



count_descp = TfidfVectorizer(max_features = 20000, 

                              ngram_range = (1,3),

                              stop_words = "english")

X_descp = count_descp.fit_transform(df["item_description"])

del df['name']

del df['item_description']

del df['brand_name']

del df['category1']

del df['category2']

del df['category3']

import scipy

item_condition_dummy = pd.get_dummies(df['item_condition_id'])

item_condition_dummy.columns = ["c1", "c2", "c3", "c4", "c5"]

df = pd.concat([df, item_condition_dummy], axis=1)

del df['item_condition_id']

Y = np.log1p(Y)

print(df)

sparse_matrix = scipy.sparse.csr_matrix(df.values)

X = scipy.sparse.hstack((sparse_matrix, 

                         X_descp,

                         X_name)).tocsr()



print(X.shape)

train_test_split = 1482535

train_X = X[:train_test_split]

test_X = X[train_test_split:]



test = pd.read_csv("test.tsv", delimiter='\t')

submission: pd.DataFrame = test[['test_id']]

def trainAndTest(X_train, Y_train, X_test):

    

    model = Ridge(solver="sag", fit_intercept=True, random_state=205, alpha=3)

    model.fit(X_train, Y_train)

    predsR = model.predict(X=X_test)

    print('Predict ridge completed')

    

    model = Ridge(solver="auto", fit_intercept=True, random_state=144, alpha=3)

    model.fit(X_train, Y_train)

    predsR2 = model.predict(X=X_test)

    print('Predict ridge 2 completed.')

    

    X_train_train, X_train_valid, Y_train_train, Y_train_valid = train_test_split(X_train, Y_train, test_size = 0.1, random_state = 144) 

    d_train = lgb.Dataset(X_train_train, label=Y_train_train, max_bin=8192)

    d_valid = lgb.Dataset(X_train_valid, label=Y_train_valid, max_bin=8192)

    watchlist = [d_train, d_valid]

    

    params = {

        'learning_rate': 0.65,

        'application': 'regression',

        'max_depth': 3,

        'num_leaves': 60,

        'verbosity': -1,

        'metric': 'RMSE',

        'data_random_seed': 1,

        'bagging_fraction': 0.5,

        'nthread': 4

    }



    params2 = {

        'learning_rate': 0.85,

        'application': 'regression',

        'max_depth': 3,

        'num_leaves': 140,

        'verbosity': -1,

        'metric': 'RMSE',

        'data_random_seed': 2,

        'bagging_fraction': 1,

        'nthread': 4

    }

    model = lgb.train(params, train_set=d_train, num_boost_round=8000, valid_sets=watchlist, \

    early_stopping_rounds=500, verbose_eval=500) 

    predsL = model.predict(X_test)

    print('Finished to predict lgb 1')

    X_train_train, X_train_valid, Y_train_train, Y_train_valid = train_test_split(X_train, Y_train, test_size = 0.1, random_state = 101) 

    d_train2 = lgb.Dataset(X_train_train, label=Y_train_train, max_bin=8192)

    d_valid2 = lgb.Dataset(X_train_valid, label=Y_train_valid, max_bin=8192)

    watchlist2 = [d_train2, d_valid2]

    

    model = lgb.train(params2, train_set=d_train2, num_boost_round=4000, valid_sets=watchlist2, \

    early_stopping_rounds=500, verbose_eval=500) 

    predsL2 = model.predict(X_test)

    print('Finished to predict lgb 2')

    

#     preds = predsR2*0.20 + predsR*0.20 + predsL*0.40 + predsL2*0.20

#     print(np.sqrt(mean_squared_log_error(np.expm1(preds),np.expm1(Y_test))))

    

#     preds = predsR2*0.19 + predsR*0.19 + predsL*0.44 + predsL2*0.18

#     print(np.sqrt(mean_squared_log_error(np.expm1(preds),np.expm1(Y_test))))

    

#     preds = predsR2*0.18 + predsR*0.18 + predsL*0.45 + predsL2*0.19

#     print(np.sqrt(mean_squared_log_error(np.expm1(preds),np.expm1(Y_test))))

    

#     preds = predsR2*0.17 + predsR*0.17 + predsL*0.46 + predsL2*0.20

#     print(np.sqrt(mean_squared_log_error(np.expm1(preds),np.expm1(Y_test))))

    

    preds = predsR2*0.16 + predsR*0.16 + predsL*0.48 + predsL2*0.20

    submission['price'] = np.expm1(preds)

    submission.to_csv("submission.csv", index=False)

#     print(np.sqrt(mean_squared_log_error(np.expm1(preds),np.expm1(Y_test))))



from sklearn.model_selection import train_test_split

trainAndTest(train_X, Y, test_X)

# Any results you write to the current directory are saved as output.