# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_csv = pd.read_csv('/kaggle/input/santander-value-prediction-challenge/train.csv')

test_csv = pd.read_csv('/kaggle/input/santander-value-prediction-challenge/test.csv')
print(train_csv.shape, test_csv.shape)
train_csv.head()
import matplotlib.pyplot as plt

import seaborn as sns
plt.figure(figsize=(12, 8))

plt.scatter(range(train_csv.shape[0]), np.sort(train_csv.target.values))

plt.grid()

plt.xlabel('index', fontsize=12)

plt.ylabel('Target', fontsize=12)

plt.title("Target Distribution", fontsize=14)
plt.figure(figsize=(12,8))

plt.hist(train_csv.target.values, bins=50)

plt.xlabel('Target', fontsize=12)

plt.title("Target Histogram", fontsize=14)
plt.figure(figsize=(12,8))

plt.hist(np.log1p(train_csv.target.values), bins=50)

plt.xlabel('Target', fontsize=12)

plt.title("Target Histogram", fontsize=14)
num_nunique = train_csv.nunique().reset_index()

num_nunique.columns = ['Col_name', 'Value_columns']

num_nunique
only1_nuique = num_nunique[num_nunique.Value_columns == 1]

only1_nuique.shape
from scipy.stats import spearmanr
from tqdm import tqdm, tqdm_notebook

from scipy.stats import spearmanr, pearsonr

import warnings

warnings.filterwarnings('ignore')

col_names = []

cor_value = []

for col in tqdm(train_csv.columns, ncols=100 , leave= True, desc="Spearman r : "):

    if col not in ['ID','target']:

        col_names.append(col)

        cor_value.append(spearmanr(train_csv[col].values, train_csv.target.values)[0])

corrs = pd.DataFrame({'Feature_Name':col_names,'Corr_value':cor_value})

corrs = corrs.sort_values(by = 'Corr_value')
corr_df = corrs[(corrs.Corr_value > 0.1) | (corrs.Corr_value < -0.1)].reset_index()

corr_df.drop('index', axis=1, inplace=True)
corr_df = corr_df.set_index('Feature_Name')

corr_df.plot(kind='barh', figsize = (12,15), title='Correlation of variables')
corr_df = corrs[(corrs.Corr_value > 0.11) | (corrs.Corr_value < -0.11)].reset_index()

corr_df.drop('index', axis = 1, inplace = True)

corr_df = corr_df.set_index('Feature_Name')

corr_df.plot(kind='barh', figsize = (12,15), title='Correlation of variables')
labels = []

values = []

for col in tqdm(train_csv.columns, ncols = 100, desc = 'Pearson r :'):

    if col not in ['ID', 'target']:

        corr_value = pearsonr(train_csv[col].values, train_csv['target'].values)

        values.append(corr_value)

        labels.append(col)

corr_df = pd.DataFrame({'Feature_Name' : labels, 'Feature_Value' : values})

corr_df = corr_df.sort_values(by = 'Feature_Value')
target = train_csv.target

train_csv.drop(list(only1_nuique.Col_name) + ['ID', 'target'], axis = 1, inplace=True)

test_csv.drop(list(only1_nuique.Col_name), axis = 1, inplace=True)
test_id = test_csv.ID

test_csv.drop('ID', axis = 1, inplace=True)
target = np.log1p(target)
from sklearn import ensemble

model = ensemble.ExtraTreesRegressor(n_estimators=200, max_depth=20, max_features=0.5,

                                     n_jobs=-1, random_state=50)

model.fit(train_csv, target)
feature_importances = pd.DataFrame({'Feature_Name' : train_csv.columns, 'feature_importance' : model.feature_importances_})

feature_importances = feature_importances.set_index('Feature_Name')



# sort by importances

feature_importances = feature_importances.sort_values(by = 'feature_importance')

std_feat_importances = np.std([tree.feature_importances_ for tree in model.estimators_],axis=0)

feature_importances.iloc[-50:].plot(kind='barh',figsize = (10,20),color = 'g',

                         xerr=std_feat_importances[-50:], align='center',

                                   title='top 50 Feature Importances')

import lightgbm as lgb

from sklearn import  model_selection, metrics

params = {

    "objective" : "regression",

    "metric" : "rmse",

    "num_leaves" : 30,

    "learning_rate" : 0.01,

    "bagging_seed" : 1884,

    "device" : "gpu",

    "gpu_platform_id" : 0,

    "gpu_device_id" : 0,

    "num_thread" : 8

}

val_data = train_csv[0:500]

val_label = target[0:500]

val_set = lgb.Dataset(val_data, val_label)

train_set = lgb.Dataset(train_csv, label=target)

model = lgb.train(params, train_set ,5000, valid_sets = [val_set],  early_stopping_rounds = 100, verbose_eval = 200)
lgb.plot_importance(model, max_num_features=50, height=0.8, figsize=(12, 20))

plt.title("LightGBM - Feature Importance top 50", fontsize=15)
result = model.predict(test_csv, num_iteration=model.best_iteration)

result = np.expm1(result)
submit = pd.DataFrame({'ID':test_id, 'target':result})



submit.to_csv('submit3_3.csv',index = False)
importance = model.feature_importance(importance_type='split')

feature_name = model.feature_name()
features = pd.DataFrame({'Feature_Name' : feature_name, 'Feature_Important' : importance}).sort_values(by=['Feature_Important'], ascending=[False])[:1000]

train = train_csv[list(features.Feature_Name)]

test = test_csv[list(features.Feature_Name)]
ntrain = train.shape[0]

ntest = test.shape[0]
weight = ((train != 0).sum()/len(train)).values

tmp_train = train[train!=0]

tmp_test = test[test!=0]

train["weight_count"] = (tmp_train*weight).sum(axis=1)

test["weight_count"] = (tmp_test*weight).sum(axis=1)

train["count_not0"] = (train != 0).sum(axis=1)

test["count_not0"] = (test != 0).sum(axis=1)

train["sum"] = train.sum(axis=1)

test["sum"] = test.sum(axis=1)

train["var"] = tmp_train.var(axis=1)

test["var"] = tmp_test.var(axis=1)

train["median"] = tmp_train.median(axis=1)

test["median"] = tmp_test.median(axis=1)

train["mean"] = tmp_train.mean(axis=1)

test["mean"] = tmp_test.mean(axis=1)

train["std"] = tmp_train.std(axis=1)

test["std"] = tmp_test.std(axis=1)

train["max"] = tmp_train.max(axis=1)

test["max"] = tmp_test.max(axis=1)

train["min"] = tmp_train.min(axis=1)

test["min"] = tmp_test.min(axis=1)

train["skew"] = tmp_train.skew(axis=1)

test["skew"] = tmp_test.skew(axis=1)

train["kurtosis"] = tmp_train.kurtosis(axis=1)

test["kurtosis"] = tmp_test.kurtosis(axis=1)
train["weight_count"] = train["weight_count"].fillna(0)

train["count_not0"] = train["count_not0"].fillna(0)

train["sum"] = train["sum"].fillna(0)

train["var"] = train["var"].fillna(0)

train["median"] = train["median"].fillna(0)

train["mean"] = train["mean"].fillna(0)

train["std"] = train["std"].fillna(0)

train["max"] = train["max"].fillna(0)

train["min"] = train["min"].fillna(0)

train["skew"] = train["skew"].fillna(0)

train["kurtosis"] = train["kurtosis"].fillna(0)



test["weight_count"] = test["weight_count"].fillna(0)

test["count_not0"] = test["count_not0"].fillna(0)

test["sum"] = test["sum"].fillna(0)

test["var"] = test["var"].fillna(0)

test["median"] = test["median"].fillna(0)

test["mean"] = test["mean"].fillna(0)

test["std"] = test["std"].fillna(0)

test["max"] = test["max"].fillna(0)

test["min"] = test["min"].fillna(0)

test["skew"] = test["skew"].fillna(0)

test["kurtosis"] = test["kurtosis"].fillna(0)
train.shape
from sklearn import random_projection

NUM_OF_COM = 100

tmp = pd.concat([train,test])

transformer = random_projection.SparseRandomProjection(n_components = NUM_OF_COM)

RP = transformer.fit_transform(tmp)

rp = pd.DataFrame(RP)

columns = ["RandomProjection{}".format(i) for i in range(NUM_OF_COM)]

rp.columns = columns

rp_train = rp[:ntrain]

rp_test = rp[ntrain:]

rp_test.index = test.index



#concat RandomProjection and raw data

train = pd.concat([train,rp_train],axis=1)

test = pd.concat([test,rp_test],axis=1)

train.shape
from sklearn import ensemble

model = ensemble.ExtraTreesRegressor(n_estimators=200, max_depth=20, max_features=0.5,

                                     n_jobs=-1, random_state=50)

model.fit(train, target)
result = model.predict(test)

result = np.expm1(result)



submit = pd.DataFrame({'ID':test_id, 'target':result})



submit.to_csv('submit3_4_extra.csv',index = False)
import xgboost as xgb

import lightgbm as lgb



model_xgb = xgb.XGBRegressor(colsample_bytree=0.055, colsample_bylevel =0.5, 

                             gamma=1.5, learning_rate=0.02, max_depth=32, 

                             objective='reg:linear',booster='gbtree',

                             min_child_weight=57, n_estimators=1000, reg_alpha=0, 

                             reg_lambda = 0,eval_metric = 'rmse', subsample=0.7, 

                             silent=1, n_jobs = -1, early_stopping_rounds = 14,

                             random_state =7, nthread = -1)

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=144,

                              learning_rate=0.005, n_estimators=720, max_depth=13,

                              metric='rmse',is_training_metric=True,

                              max_bin = 55, bagging_fraction = 0.8,verbose=-1,

                              bagging_freq = 5, feature_fraction = 0.9) 



model_xgb.fit(train, target)

model_lgb.fit(train, target)
result = model_xgb.predict(test)

result = np.expm1(result)



submit = pd.DataFrame({'ID':test_id, 'target':result})



submit.to_csv('submit3_4_xgb.csv',index = False)
result = model_lgb.predict(test)

result = np.expm1(result)



submit = pd.DataFrame({'ID':test_id, 'target':result})



submit.to_csv('submit3_4_lgb.csv',index = False)