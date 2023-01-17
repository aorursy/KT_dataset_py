# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# importes

import time

import warnings

import gc

import os

from six.moves import urllib

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

import datetime

warnings.filterwarnings('ignore')

%matplotlib inline

plt.style.use('seaborn')

from scipy.stats import norm, skew

from sklearn.preprocessing import StandardScaler

from sklearn.utils import shuffle

from sklearn.pipeline import Pipeline

from sklearn.pipeline import FeatureUnion

from sklearn.linear_model import Lasso

from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error

from sklearn.model_selection import train_test_split #training and testing data split

from sklearn import metrics #accuracy measure

from sklearn.metrics import confusion_matrix #for confusion matrix

from scipy.stats import reciprocal, uniform

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import KFold #for K-fold cross validation

from sklearn.model_selection import cross_val_score #score evaluation

from sklearn.model_selection import cross_val_predict #prediction

from sklearn.model_selection import cross_validate

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.utils import check_array

from scipy import sparse
np.random.seed(123)

gc.collect()

%matplotlib inline

plt.rcParams['axes.labelsize'] = 14

plt.rcParams['xtick.labelsize'] = 12

plt.rcParams['ytick.labelsize'] = 12
#redução de memória

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
train = reduce_mem_usage(pd.read_csv('../input/competicao-dsa-machine-learning-jun-2019/dataset_treino.csv',parse_dates=["first_active_month"]))

test = reduce_mem_usage(pd.read_csv('../input/competicao-dsa-machine-learning-jun-2019/dataset_teste.csv', parse_dates=["first_active_month"]))
train.shape
train.info()
train["month"] = train["first_active_month"].dt.month

train["year"] = train["first_active_month"].dt.year

train['week'] = train["first_active_month"].dt.weekofyear

train['dayofweek'] = train['first_active_month'].dt.dayofweek

train['days'] = (datetime.date(2018, 2, 1) - train['first_active_month'].dt.date).dt.days



test["month"] = test["first_active_month"].dt.month

test["year"] = test["first_active_month"].dt.year

test['week'] = test["first_active_month"].dt.weekofyear

test['dayofweek'] = test['first_active_month'].dt.dayofweek

test['days'] = (datetime.date(2018, 2, 1) - test['first_active_month'].dt.date).dt.days
def aggregate_transaction_hist(trans, prefix):  

        

    agg_func = {

        'purchase_date' : ['max','min'],

        'month_diff' : ['mean'],

        'weekend' : ['sum', 'mean'],

        'authorized_flag': ['sum', 'mean'],

        'category_1': ['sum','mean'],

        'purchase_amount': ['sum', 'mean', 'max', 'min', 'std'],

        'installments': ['sum', 'mean', 'max', 'min', 'std'],  

        #'merchant_id': ['nunique'],

        'month_lag': ['max','min','mean','var'],

        'month_diff' : ['mean'],

        'card_id' : ['size'],

        'month': ['nunique'],

        'hour': ['nunique'],

        'weekofyear': ['nunique'],

        'dayofweek': ['nunique'],

        'year': ['nunique'],

        'subsector_id': ['nunique'],

        'merchant_category_id' : ['nunique'],

        'Christmas_Day_2017':['mean'],

        #'Mothers_Day_2017':['mean'],

        'fathers_day_2017':['mean'],

        'Children_day_2017':['mean'],

        'Black_Friday_2017':['mean'],

        'Valentine_day_2017':['mean'],

        'Mothers_Day_2018':['mean']

    }

    

    agg_trans = trans.groupby(['card_id']).agg(agg_func)

    agg_trans.columns = [prefix + '_'.join(col).strip() 

                           for col in agg_trans.columns.values]

    agg_trans.reset_index(inplace=True)

    

    df = (trans.groupby('card_id')

          .size()

          .reset_index(name='{}transactions_count'.format(prefix)))

    

    agg_trans = pd.merge(df, agg_trans, on='card_id', how='left')

    

    return agg_trans
transactions = reduce_mem_usage(pd.read_csv('../input/transacoes-historicas-1000000/transacoes_historicas_1000000.csv'))

transactions['authorized_flag'] = transactions['authorized_flag'].map({'Y': 1, 'N': 0})

transactions['category_1'] = transactions['category_1'].map({'Y': 1, 'N': 0})
transactions['purchase_date'] = pd.to_datetime(transactions['purchase_date'])

transactions['year'] = transactions['purchase_date'].dt.year

transactions['weekofyear'] = transactions['purchase_date'].dt.weekofyear

transactions['month'] = transactions['purchase_date'].dt.month

transactions['dayofweek'] = transactions['purchase_date'].dt.dayofweek

transactions['weekend'] = (transactions.purchase_date.dt.weekday >=5).astype(int)

transactions['hour'] = transactions['purchase_date'].dt.hour 

transactions['month_diff'] = ((datetime.datetime.today() - transactions['purchase_date']).dt.days)//30

transactions['month_diff'] += transactions['month_lag']

transactions['category_2'] = transactions['category_2'].fillna(1.0,inplace=True)

transactions['category_3'] = transactions['category_3'].fillna('A',inplace=True)

transactions['merchant_id'] = transactions['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)

gc.collect()
agg_func = {

        'mean': ['mean'],

    }

for col in ['category_2','category_3']:

    transactions[col+'_mean'] = transactions['purchase_amount'].groupby(transactions[col]).agg(agg_func)
#feriados e possveis influencias

transactions['Christmas_Day_2017'] = (pd.to_datetime('2017-12-25') - transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

transactions['fathers_day_2017'] = (pd.to_datetime('2017-08-13') - transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

transactions['Children_day_2017'] = (pd.to_datetime('2017-10-12') - transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

transactions['Black_Friday_2017'] = (pd.to_datetime('2017-11-24') - transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

transactions['Valentine_day_2017'] = (pd.to_datetime('2017-06-12') - transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

transactions['Mothers_Day_2018'] = (pd.to_datetime('2018-05-13') - transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

gc.collect()
merge_trans = aggregate_transaction_hist(transactions, prefix='hist_')

del transactions

gc.collect()

train = pd.merge(train, merge_trans, on='card_id',how='left')

test = pd.merge(test, merge_trans, on='card_id',how='left')

del merge_trans

gc.collect()
train.head(10)
train['hist_purchase_date_max'] = pd.to_datetime(train['hist_purchase_date_max'])

train['hist_purchase_date_min'] = pd.to_datetime(train['hist_purchase_date_min'])

train['hist_purchase_date_diff'] = (train['hist_purchase_date_max'] - train['hist_purchase_date_min']).dt.days

train['hist_purchase_date_average'] = train['hist_purchase_date_diff']/train['hist_card_id_size']

train['hist_purchase_date_uptonow'] = (datetime.datetime.today() - train['hist_purchase_date_max']).dt.days

train['hist_first_buy'] = (train['hist_purchase_date_min'] - train['first_active_month']).dt.days

for feature in ['hist_purchase_date_max','hist_purchase_date_min']:

    train[feature] = train[feature].astype(np.int64) * 1e-9
test['hist_purchase_date_max'] = pd.to_datetime(test['hist_purchase_date_max'])

test['hist_purchase_date_min'] = pd.to_datetime(test['hist_purchase_date_min'])

test['hist_purchase_date_diff'] = (test['hist_purchase_date_max'] - test['hist_purchase_date_min']).dt.days

test['hist_purchase_date_average'] = test['hist_purchase_date_diff']/test['hist_card_id_size']

test['hist_purchase_date_uptonow'] = (datetime.datetime.today() - test['hist_purchase_date_max']).dt.days

test['hist_first_buy'] = (test['hist_purchase_date_min'] - test['first_active_month']).dt.days

for feature in ['hist_purchase_date_max','hist_purchase_date_min']:

    test[feature] = test[feature].astype(np.int64) * 1e-9
def aggregate_transaction_new(trans, prefix):  

        

    agg_func = {

        'purchase_date' : ['max','min'],

        'month_diff' : ['mean'],

        'weekend' : ['sum', 'mean'],

        'authorized_flag': ['sum'],

        'category_1': ['sum','mean'],

        'purchase_amount': ['sum', 'mean', 'max', 'min', 'std'],

        'installments': ['sum', 'mean', 'max', 'min', 'std'],  

        'month_lag': ['max','min','mean','var'],

        'month_diff' : ['mean'],

        'card_id' : ['size'],

        'month': ['nunique'],

        'hour': ['nunique'],

        'weekofyear': ['nunique'],

        'dayofweek': ['nunique'],

        'year': ['nunique'],

        'subsector_id': ['nunique'],

        'merchant_category_id' : ['nunique'],

        'Christmas_Day_2017':['mean'],

        'fathers_day_2017':['mean'],

        'Children_day_2017':['mean'],

        'Black_Friday_2017':['mean'],

        'Valentine_Day_2017' : ['mean'],

        'Mothers_Day_2018':['mean']

    }

    agg_trans = trans.groupby(['card_id']).agg(agg_func)

    agg_trans.columns = [prefix + '_'.join(col).strip() 

                           for col in agg_trans.columns.values]

    agg_trans.reset_index(inplace=True)

    df = (trans.groupby('card_id')

          .size()

          .reset_index(name='{}transactions_count'.format(prefix)))

    

    agg_trans = pd.merge(df, agg_trans, on='card_id', how='left')

    return agg_trans
new_transactions = reduce_mem_usage(pd.read_csv('../input/dsa-upload-novas-transacoes-comerciantes/novas_transacoes_comerciantes.csv'))

new_transactions['authorized_flag'] = new_transactions['authorized_flag'].map({'Y': 1, 'N': 0})

new_transactions['category_1'] = new_transactions['category_1'].map({'Y': 1, 'N': 0})
new_transactions['purchase_date'] = pd.to_datetime(new_transactions['purchase_date'])

new_transactions['year'] = new_transactions['purchase_date'].dt.year

new_transactions['weekofyear'] = new_transactions['purchase_date'].dt.weekofyear

new_transactions['month'] = new_transactions['purchase_date'].dt.month

new_transactions['dayofweek'] = new_transactions['purchase_date'].dt.dayofweek

new_transactions['weekend'] = (new_transactions.purchase_date.dt.weekday >=5).astype(int)

new_transactions['hour'] = new_transactions['purchase_date'].dt.hour 

new_transactions['month_diff'] = ((datetime.datetime.today() - new_transactions['purchase_date']).dt.days)//30

new_transactions['month_diff'] += new_transactions['month_lag']

new_transactions['category_2'] = new_transactions['category_2'].fillna(1.0,inplace=True)

new_transactions['category_3'] = new_transactions['category_3'].fillna('A',inplace=True)

new_transactions['merchant_id'] = new_transactions['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)

new_transactions['Christmas_Day_2017'] = (pd.to_datetime('2017-12-25') - new_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

new_transactions['fathers_day_2017'] = (pd.to_datetime('2017-08-13') - new_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

new_transactions['Children_day_2017'] = (pd.to_datetime('2017-10-12') - new_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

new_transactions['Valentine_Day_2017'] = (pd.to_datetime('2017-06-12') - new_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

new_transactions['Black_Friday_2017'] = (pd.to_datetime('2017-11-24') - new_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

new_transactions['Mothers_Day_2018'] = (pd.to_datetime('2018-05-13') - new_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

agg_func = {

        'mean': ['mean'],

    }

for col in ['category_2','category_3']:

    new_transactions[col+'_mean'] = new_transactions['purchase_amount'].groupby(new_transactions[col]).agg(agg_func)



gc.collect()
merge_new = aggregate_transaction_new(new_transactions, prefix='new_')

del new_transactions

gc.collect()



train = pd.merge(train, merge_new, on='card_id',how='left')

test = pd.merge(test, merge_new, on='card_id',how='left')

del merge_new



gc.collect()
train['new_purchase_date_max'] = pd.to_datetime(train['new_purchase_date_max'])

train['new_purchase_date_min'] = pd.to_datetime(train['new_purchase_date_min'])

train['new_purchase_date_diff'] = (train['new_purchase_date_max'] - train['new_purchase_date_min']).dt.days

train['new_purchase_date_average'] = train['new_purchase_date_diff']/train['new_card_id_size']

train['new_purchase_date_uptonow'] = (datetime.datetime.today() - train['new_purchase_date_max']).dt.days

train['new_first_buy'] = (train['new_purchase_date_min'] - train['first_active_month']).dt.days

for feature in ['new_purchase_date_max','new_purchase_date_min']:

    train[feature] = train[feature].astype(np.int64) * 1e-9



test['new_purchase_date_max'] = pd.to_datetime(test['new_purchase_date_max'])

test['new_purchase_date_min'] = pd.to_datetime(test['new_purchase_date_min'])

test['new_purchase_date_diff'] = (test['new_purchase_date_max'] - test['new_purchase_date_min']).dt.days

test['new_purchase_date_average'] = test['new_purchase_date_diff']/test['new_card_id_size']

test['new_purchase_date_uptonow'] = (datetime.datetime.today() - test['new_purchase_date_max']).dt.days

test['new_first_buy'] = (test['new_purchase_date_min'] - test['first_active_month']).dt.days

for feature in ['new_purchase_date_max','new_purchase_date_min']:

    test[feature] = test[feature].astype(np.int64) * 1e-9

    

train['card_id_total'] = train['new_card_id_size'] + train['hist_card_id_size']

train['purchase_amount_total'] = train['new_purchase_amount_sum'] + train['hist_purchase_amount_sum']

test['card_id_total'] = test['new_card_id_size'] + test['hist_card_id_size']

test['purchase_amount_total'] = test['new_purchase_amount_sum'] + test['hist_purchase_amount_sum']

gc.collect()
train.shape
test.shape
train.head(10)
nulls = np.sum(train.isnull())

nullcols = nulls.loc[(nulls != 0)]

dtypes = train.dtypes

dtypes2 = dtypes.loc[(nulls != 0)]

info = pd.concat([nullcols, dtypes2], axis=1).sort_values(by=0, ascending=False)

print(info)

print("There are", len(nullcols), "columns with missing values in data set")
#Check for missing values in training set

nulls = np.sum(test.isnull())

nullcols = nulls.loc[(nulls != 0)]

dtypes = test.dtypes

dtypes2 = dtypes.loc[(nulls != 0)]

info = pd.concat([nullcols, dtypes2], axis=1).sort_values(by=0, ascending=False)

print(info)

print("There are", len(nullcols), "columns with missing values in test set")
numeric_dtypes = ['float64']

numerics = []

for i in train.columns:

    if train[i].dtype in numeric_dtypes: 

        numerics.append(i)

        

train.update(train[numerics].fillna(0))
numeric_dtypes = ['float64']

numerics = []

for i in test.columns:

    if test[i].dtype in numeric_dtypes: 

        numerics.append(i)

        

test.update(test[numerics].fillna(0))
#remoção de outliers

#-10-20-30

train['outliers'] = 0

train.loc[train['target'] < -30, 'outliers'] = 1

train['outliers'].value_counts()
for features in ['feature_1','feature_2','feature_3']:

    order_label = train.groupby([features])['outliers'].mean()

    train[features] = train[features].map(order_label)

    test[features] =  test[features].map(order_label)
# X e Y

df_train_columns = [c for c in train.columns if c not in ['card_id', 'first_active_month','target','outliers']]

target = train['target']
import lightgbm as lgb

#7500 - 10000 - 15000 - 20000

num_round_qt=10000

#100 - 150 - 200 - 250

early_stopping_qtd=100

treads=16

#0.1 - 0.05 - 0.005

learning_rate=0.005



param = {'num_leaves': 31,

         'min_data_in_leaf': 32, 

         'objective':'regression',

         'max_depth': -1,

         'learning_rate': learning_rate,

         "boosting": "gbdt",

         "feature_fraction": 0.9,

         "bagging_freq": 1,

         "bagging_fraction": 0.9,

         "bagging_seed": 11,

         "metric": 'rmse',

         "lambda_l1": 0.1,

         "verbosity": -1,

         "nthread": treads}



folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=4190)

oof = np.zeros(len(train))

predictions = np.zeros(len(test))

feature_importance_df = pd.DataFrame()



for fold_, (trn_idx, val_idx) in enumerate(folds.split(train,train['outliers'].values)):

    print("fold {}".format(fold_))

    trn_data = lgb.Dataset(train.iloc[trn_idx][df_train_columns], label=target.iloc[trn_idx])

    val_data = lgb.Dataset(train.iloc[val_idx][df_train_columns], label=target.iloc[val_idx])



    num_round = num_round_qt

    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=-1, early_stopping_rounds = early_stopping_qtd)

    oof[val_idx] = clf.predict(train.iloc[val_idx][df_train_columns], num_iteration=clf.best_iteration)

    

    fold_importance_df = pd.DataFrame()

    fold_importance_df["Feature"] = df_train_columns

    fold_importance_df["importance"] = clf.feature_importance()

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    

    predictions += clf.predict(test[df_train_columns], num_iteration=clf.best_iteration) / folds.n_splits



np.sqrt(mean_squared_error(oof, target))
cols = (feature_importance_df[["Feature", "importance"]]

        .groupby("Feature")

        .mean()

        .sort_values(by="importance", ascending=False)[:1000].index)



best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]



plt.figure(figsize=(14,25))

sns.barplot(x="importance",

            y="Feature",

            data=best_features.sort_values(by="importance",

                                           ascending=False))

plt.title('LightGBM Features (avg over folds)')

plt.tight_layout()

plt.savefig('lgbm_importances_v5.png')


# usando 2 casas decimais e setando os valores negativos com o máximo de -10

sample_submission = pd.read_csv('../input/competicao-dsa-machine-learning-jun-2019/sample_submission.csv')

sample_submission['target'] =[float(np.round(x,2)) for x in predictions]

sample_submission['target'] = np.where(sample_submission.target < -10, -10, sample_submission.target)
sample_submission.to_csv('submit.csv', index=False)
sample_submission.head(10)