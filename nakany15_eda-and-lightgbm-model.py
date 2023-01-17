# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output
import sys

import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import datetime

%matplotlib inline

import lightgbm as lgb

from catboost import CatBoostRegressor

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import KFold

pd.options.display.max_columns = None
train_df = pd.read_csv("../input/train.csv", low_memory=False)

test_df = pd.read_csv("../input/test.csv", low_memory=False)
train_df.head()
train_df.info()
NA_col = train_df.isnull().sum()

#NA_col = NA_col[NA_col.values >(0.1*len(train_df))]

plt.figure(figsize=(20,4))

NA_col.plot(kind='bar')

plt.title('List of Columns & NA counts where NA values are more than 30%')

plt.show()
NA_col = test_df.isnull().sum()

#NA_col = NA_col[NA_col.values >(0.1*len(train_df))]

plt.figure(figsize=(20,4))

NA_col.plot(kind='bar')

plt.title('List of Columns & NA counts where NA values are more than 30%')

plt.show()
for col in [c for c in train_df.columns if c in train_df.describe().columns]:

    print("unique values for ", col, train_df[col].drop_duplicates().shape)
for col in [c for c in train_df.columns if c not in train_df.describe().columns]:

    print("unique values for ", col, train_df[col].drop_duplicates().shape)
train_df.describe()
test_df.describe()
fig, axes = plt.subplots( int(len([c for c in train_df.columns if c in train_df.describe().columns])/3), 3, figsize=(24, 48))

for col, axis in zip([c for c in train_df.columns if c in train_df.describe().columns], axes.flatten()):

    train_df.hist(column = col, bins=20, ax=axis)
fig, axes = plt.subplots( int(len([c for c in test_df.columns if c in test_df.describe().columns])/3), 3, figsize=(24, 48))

for col, axis in zip([c for c in train_df.columns if c in test_df.describe().columns], axes.flatten()):

    train_df.hist(column = col, bins=10, ax=axis, color="#5F9BFF", alpha=.5, label = 'train')

    test_df.hist(column = col, bins=10, ax=axis, color="#F8766D", alpha=.5, label = 'test')    
fig, axes = plt.subplots( int(len([c for c in train_df.columns if c in train_df.describe().columns])/3), 3, figsize=(24, 48))

for col, axis in zip([c for c in train_df.columns if c in train_df.describe().columns], axes.flatten()):

    train_df.plot(kind = 'scatter',x = col,y = 'PRICE', ax=axis)
train_df['log_price'] = np.log(train_df['PRICE'])

train_df.plot(kind = 'scatter', x = 'X', y = 'Y', c = 'log_price', cmap ='GnBu' , 

              title = 'geographic price distribution')
train_df['sale_year'] = train_df['SALEDATE'].str.slice(0,4).astype("float16")

train_df.groupby('sale_year')['log_price'].mean().plot(title = 'price trends based on sale year')

train_df.drop(['sale_year'], axis = 1, inplace = True)
train_df.groupby('AYB')['log_price'].count().plot(title = 'resale count trends based on AYB')
train_df.groupby('AYB')['log_price'].mean().plot(title = 'price trends based on AYB')
train_df.groupby('EYB')['log_price'].mean().plot(title = 'price trends based on EYB')

train_df.drop(['log_price'], axis = 1, inplace = True)
train_df.query('PRICE > 10000000')['PRICE'].value_counts()
#unique value counts for numerics

for col in [c for c in train_df.columns if c in train_df.describe().columns]:

    print("unique values for ", col, train_df.query('PRICE == 137427545.0')[col].drop_duplicates().shape)
#unique value counts for categoricals

for col in [c for c in train_df.columns if c not in train_df.describe().columns]:

    print("unique values for ", col, train_df.query('PRICE == 137427545.0')[col].drop_duplicates().shape)
train_df.query("CENSUS_TRACT == 1002.0 and YR_RMDL == 2005 and SALEDATE == '2007-04-10 00:00:00'").shape
test_df.query("CENSUS_TRACT == 1002.0 and YR_RMDL == 2005 and SALEDATE == '2007-04-10 00:00:00'").shape
# query for price == 137427545

test_df.loc[test_df.query("CENSUS_TRACT == 1002.0 and YR_RMDL == 2005 and SALEDATE == '2007-04-10 00:00:00'").index]
# unique value counts for numerics

for col in [c for c in train_df.columns if c in train_df.describe().columns]:

    print("unique values for ", col, train_df.query('PRICE == 53969391.0')[col].drop_duplicates().shape)
# unique value counts for categoricals

for col in [c for c in train_df.columns if c  not in train_df.describe().columns]:

    print("unique values for ", col, train_df.query('PRICE == 53969391.0')[col].drop_duplicates().shape)
X_grid = train_df.query('PRICE == 53969391.0')['X'].drop_duplicates().values[0]

Y_grid = train_df.query('PRICE == 53969391.0')['Y'].drop_duplicates().values[0]

SaleDate = train_df.query('PRICE == 53969391.0')['SALEDATE'].drop_duplicates().values[0]
train_df.loc[(train_df['X'] == X_grid) & (train_df['Y'] == Y_grid) & (train_df['SALEDATE'] == SaleDate)].shape
# query for price == 53969391

test_df.loc[(train_df['X'] == X_grid) & (train_df['Y'] == Y_grid) & (train_df['SALEDATE'] == SaleDate)].index
X_knn = train_df[['X', 'Y']].copy()

y_knn = np.log(train_df['PRICE'])

X_knn['X'].fillna(X_knn['X'].mean(), inplace = True)

X_knn['Y'].fillna(X_knn['Y'].mean(), inplace = True)
X_knn_test = test_df[['X', 'Y']].copy()

X_knn_test['X'].fillna(X_knn_test['X'].mean(), inplace = True)

X_knn_test['Y'].fillna(X_knn_test['Y'].mean(), inplace = True)
folds = KFold(n_splits=5, shuffle=True, random_state=42)

predictions1 = np.zeros(len(test_df))

predictions2 = np.zeros(len(test_df))

predictions3 = np.zeros(len(test_df))

oof1 = np.zeros(len(train_df))

oof2 = np.zeros(len(train_df))

oof3 = np.zeros(len(train_df))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_knn.values, y_knn.values)):

    print("Fold {}".format(fold_))

    trn_idx_cond = np.array(train_df.iloc[trn_idx].query("SOURCE == 'Condominium'").index)

    trn_idx_resid = np.array(train_df.iloc[trn_idx].query("SOURCE == 'Residential'").index)

    clf1 = KNeighborsRegressor(n_neighbors=10, n_jobs = -1)    

    clf1.fit(X_knn.iloc[trn_idx], y_knn.iloc[trn_idx])

    oof1[val_idx] = np.exp(clf1.predict(X_knn.iloc[val_idx])) 

    predictions1 += np.exp(clf1.predict(X_knn_test.values)) / folds.n_splits

train_df['Neighbor_price'] = oof1

test_df['Neighbor_price'] = predictions1
X_train_test = pd.concat([train_df.drop(["PRICE", "ID"], axis=1), test_df.drop(["ID"], axis=1)], axis=0)

X_train_test["year"] = X_train_test['SALEDATE'].str.slice(0,4).astype("float16")

X_train_test["month"] = X_train_test['SALEDATE'].str.slice(5,7).astype("float16")

X_train_test["SALEDATE"] = pd.to_datetime(X_train_test["SALEDATE"]).astype("int")

X_train_test["GIS_LAST_MOD_DTTM"] = pd.to_datetime(X_train_test["GIS_LAST_MOD_DTTM"]).astype("int")

X_train_test["Area_pre_room"] = X_train_test["LANDAREA"]/X_train_test["ROOMS"]

X_train_test["Unit_per_Room"] = X_train_test["NUM_UNITS"]/X_train_test["ROOMS"]

X_train_test["Total_Bath"] = X_train_test["HF_BATHRM"] + X_train_test["BATHRM"]

X_train_test['yr_diff'] = X_train_test['year'] - X_train_test['AYB']

X_train_test['Room_per_Story'] = X_train_test["ROOMS"]/X_train_test["STORIES"]

X_train_test['FP_per_Room'] = X_train_test["FIREPLACES"]/X_train_test["ROOMS"]

#X_train_test['kitchen_per_floor'] = X_train_test["KITCHENS"]/X_train_test["STORIES"]

X_train_test['yr_diff2'] = X_train_test['year'] - X_train_test['YR_RMDL']
X_train_test["X_diff"] = abs(X_train_test["X"] - 77.0753707272469)

X_train_test['Y_diff'] = abs(X_train_test["Y"] - 38.9377808506798)

X_train_test["distance"] = np.sqrt(X_train_test["X_diff"]**2 + X_train_test['Y_diff']**2)

X_train_test.drop(["X_diff", "Y_diff"], axis = 1, inplace = True)





X_train_test["X_diff2"] = abs(X_train_test["X"] - 76.93)

X_train_test['Y_diff2'] = abs(X_train_test["Y"] - 38.88)

X_train_test["distance2"] = np.sqrt(X_train_test["X_diff2"]**2 + X_train_test['Y_diff2']**2)

X_train_test.drop(["X_diff2", "Y_diff2"], axis = 1, inplace = True)
categorical_features = [c for c in X_train_test.columns if c not in X_train_test.describe().columns] + ["USECODE"]

#catcols_idx  = [i for i in range(len(X_train_test.columns)) if X_train_test.columns[i] in categorical_features]
categorical_features += ["ZIPCODE"]
for col in categorical_features:

    X_train_test[col] = X_train_test[col].astype('category')

    X_train_test[col] = X_train_test[col].cat.add_categories(["other"])

    X_train_test[col].fillna("other", inplace = True)
X = X_train_test.iloc[:train_df.shape[0], :]

X_test = X_train_test.iloc[train_df.shape[0]:, :]

y = np.log(train_df["PRICE"].values)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=71)
params = {

    'boosting_type': 'gbdt',

    'objective': 'regression',

    'metric': 'rmse',

    'num_leaves': 530,

    'learning_rate': 0.008,

    'feature_fraction': 0.35,

    'verbose': 0

}



lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_features)

lgb_valid = lgb.Dataset(X_valid, y_valid, categorical_feature=categorical_features)





print('Starting training...')



model = lgb.train(

    params, 

    lgb_train, 

    valid_sets=(lgb_train, lgb_valid), 

    num_boost_round=10000, 

    verbose_eval = 100,

    early_stopping_rounds=100

)
X2 = X.copy()

X2['pred'] = np.exp(model.predict(X2))

X2['PRICE'] = train_df.PRICE
X2['residual'] = X2['PRICE'] - X2['pred']

X2['APE'] = X2['residual']/X2['PRICE']
X2['APE'].describe()
X2.sort_values(by = 'APE', ascending = True).head(20)
X2.sort_values(by = 'APE', ascending = False).head(20)
#test_pred = np.exp(model.predict(X_test))

#sub_df = pd.DataFrame({"ID":test_df["ID"].values,"PRICE":test_pred})

#sub_df.to_csv("sub2.csv", index=False)
lgb.plot_importance(model, max_num_features = 30)
X_train_test[["GBA", "LIVING_GBA"]].info()
X_train_test2 = X_train_test[X_train_test["LIVING_GBA"].notnull()]

y2 = X_train_test2["LIVING_GBA"]

X_train_test2 = X_train_test2.drop(["LIVING_GBA"], axis = 1)

y2 = np.log(y2 + 1)
X_train2, X_valid2, y_train2, y_valid2 = train_test_split(X_train_test2, y2, test_size=0.2, random_state=71)
params = {

    'boosting_type': 'gbdt',

    'objective': 'regression',

    'metric': 'rmse',

    'num_leaves': 530,

    'learning_rate': 0.008,

    'feature_fraction': 0.35,

    'verbose': 0

}



lgb_train = lgb.Dataset(X_train2, y_train2, categorical_feature=categorical_features)

lgb_valid = lgb.Dataset(X_valid2, y_valid2, categorical_feature=categorical_features)





print('Starting training...')

model = lgb.train(

    params, 

    lgb_train, 

    valid_sets=lgb_valid, 

    num_boost_round=10000, 

    verbose_eval = 100,

    early_stopping_rounds=100

)
X_train_test["LIVING_GBA_pred"] = np.exp(model.predict(X_train_test.drop(["LIVING_GBA"], axis = 1))) - 1
X_train_test["LIVING_GBA"].hist()
X_train_test[X_train_test["LIVING_GBA"].isnull()].LIVING_GBA_pred.hist()
X_train_test["LIVING_GBA"].fillna(X_train_test["LIVING_GBA_pred"], inplace = True)

X_train_test.drop(["LIVING_GBA_pred"], axis = 1, inplace = True)
#X_train = X_train_test.iloc[:train_df.shape[0], :]

#X_test = X_train_test.iloc[train_df.shape[0]:, :]

#y_train = np.log(train_df["PRICE"].values)

#X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=71)
# specify your configurations as a dict

"""

params = {

    'boosting_type': 'gbdt',

    'objective': 'regression',

    'metric': 'rmse',

    'num_leaves': 511,

    'learning_rate': 0.01,

    'feature_fraction': 0.8,

    'verbose': 0

}

# create dataset for lightgbm

lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_features)

lgb_valid = lgb.Dataset(X_valid, y_valid, categorical_feature=categorical_features)





print('Starting training...')

# train

model = lgb.train(

    params, 

    lgb_train, 

    valid_sets=lgb_valid, 

    num_boost_round=10000, 

    early_stopping_rounds=200,

    verbose_eval = 100,

)

"""
X_train_test2 = X_train_test[X_train_test["GBA"].notnull()]

y2 = X_train_test2["GBA"]

X_train_test2 = X_train_test2.drop(["GBA"], axis = 1)

y2 = np.log(y2 + 1)
X_train2, X_valid2, y_train2, y_valid2 = train_test_split(X_train_test2, y2, test_size=0.2, random_state=71)


params = {

    'boosting_type': 'gbdt',

    'objective': 'regression',

    'metric': 'rmse',

    'num_leaves': 530,

    'learning_rate': 0.008,

    'feature_fraction': 0.35,

    'verbose': 0

}



lgb_train = lgb.Dataset(X_train2, y_train2, categorical_feature=categorical_features)

lgb_valid = lgb.Dataset(X_valid2, y_valid2, categorical_feature=categorical_features)





print('Starting training...')

model = lgb.train(

    params, 

    lgb_train, 

    valid_sets=lgb_valid, 

    num_boost_round=10000, 

    verbose_eval = 100,

    early_stopping_rounds=100

)
lgb.plot_importance(model, max_num_features = 30)
X_train_test["GBA_pred"] = np.exp(model.predict(X_train_test.drop(["GBA"], axis = 1))) - 1
X_train_test[["GBA","GBA_pred"]].head()
X_train_test[["GBA","GBA_pred"]].describe()
X_train_test["GBA"].fillna(X_train_test["GBA_pred"], inplace = True)

X_train_test.drop(["GBA_pred"], axis = 1 , inplace = True)
X_train = X_train_test.iloc[:train_df.shape[0], :]

X_test = X_train_test.iloc[train_df.shape[0]:, :]

y_train = np.log(train_df["PRICE"].values)

#X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=71)
params = {

    'boosting_type': 'gbdt',

    'objective': 'regression',

    'metric': 'rmse',

    'num_leaves': 530,

    'learning_rate': 0.008,

    'feature_fraction': 0.35,

    'verbose': 0

}
folds = KFold(n_splits=5, shuffle=True, random_state=42)

predictions = np.zeros(len(test_df))

oof = np.zeros(len(train_df))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):

    print("Fold {}".format(fold_))

    lgb_train = lgb.Dataset(X_train.iloc[trn_idx], y_train[trn_idx], categorical_feature=categorical_features)

    lgb_valid = lgb.Dataset(X_train.iloc[val_idx], y_train[val_idx], categorical_feature=categorical_features)

    # train

    model = lgb.train(

        params, 

        lgb_train, 

        valid_sets=lgb_valid, 

        num_boost_round=10000, 

        early_stopping_rounds=200,

        verbose_eval = 100,

    )

    oof[val_idx] = np.exp(model.predict(X_train.iloc[val_idx]))

    predictions += np.exp(model.predict(X_test)) / folds.n_splits
lgb.plot_importance(model, max_num_features = 30)
#test_pred = np.exp(model.predict(X_test))

sub_df = pd.DataFrame({"ID":test_df["ID"].values,"PRICE":predictions})

#sub_df.loc[test_df.query("CENSUS_TRACT == 1002.0 and YR_RMDL == 2005 and SALEDATE == '2007-04-10 00:00:00'").index, 'PRICE'] = 137427545.0

# query for price == 53969391

#sub_df.loc[test_df.loc[(train_df['X'] == X_grid) & (train_df['Y'] == Y_grid) & (train_df['SALEDATE'] == SaleDate)].index, 'PRICE'] = 53969391.0

sub_df.to_csv("sub2.csv", index=False)
# specify your configurations as a dict

"""

params = {

    'boosting_type': 'gbdt',

    'objective': 'regression',

    'metric': 'rmse',

    'num_leaves': 530,

    'learning_rate': 0.008,

    'feature_fraction': 0.45,

    'verbose': 0

}

# create dataset for lightgbm

lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_features)

lgb_valid = lgb.Dataset(X_valid, y_valid, categorical_feature=categorical_features)





print('Starting training...')

# train

model = lgb.train(

    params, 

    lgb_train, 

    valid_sets=lgb_valid, 

    num_boost_round=10000, 

    early_stopping_rounds=200,

    verbose_eval = 100,

)

"""