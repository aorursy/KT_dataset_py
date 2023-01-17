import numpy as np

import pandas as pd

from datetime import datetime

import gc

import warnings

from bayes_opt import BayesianOptimization



import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb

from sklearn.model_selection import KFold

import warnings

import time

import sys

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import mean_squared_error

from datetime import datetime

warnings.simplefilter(action='ignore', category=FutureWarning)
def timer(start_time=None):

    if not start_time:

        start_time = datetime.now()

        return start_time

    elif start_time:

        thour, temp_sec = divmod(

            (datetime.now() - start_time).total_seconds(), 3600)

        tmin, tsec = divmod(temp_sec, 60)

        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))
def reduce_mem_usage(df):

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type != object:

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

    print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
train = pd.read_csv("../input/elo-world/train.csv", index_col=0)

train = reduce_mem_usage(train)



target = train['target']

del train['target']
unimportant_features = [

    'auth_category_2_1.0_mean',

    'auth_category_2_2.0_mean',

    'auth_category_2_3.0_mean',

    'auth_category_2_5.0_mean',

    'hist_category_2_3.0_mean',

    'hist_category_2_4.0_mean',

    'hist_category_2_5.0_mean',

    'hist_category_3_A_mean',

    'hist_installments_min',

    'hist_installments_std',

    'hist_month_lag_std',

    'hist_purchase_amount_max',

    'hist_purchase_month_max',

    'hist_purchase_month_min',

    'hist_purchase_month_std',

    'installments_min_mean',

    'new_category_2_1.0_mean',

    'new_category_2_2.0_mean',

    'new_category_2_3.0_mean',

    'new_category_2_5.0_mean',

    'new_city_id_nunique',

    'new_installments_std',

    'new_state_id_nunique',

    'purchase_amount_mean_mean'

]

features = [c for c in train.columns if c not in ['card_id', 'first_active_month']]

#features = [f for f in features if f not in unimportant_features]

categorical_feats = [c for c in features if 'feature_' in c]
def LGB_CV(

          max_depth,

          num_leaves,

          min_data_in_leaf,

          feature_fraction,

          bagging_fraction,

          lambda_l1

         ):

    

    folds = KFold(n_splits=5, shuffle=True, random_state=15)

    oof = np.zeros(train.shape[0])



    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):

        print("fold n°{}".format(fold_))

        trn_data = lgb.Dataset(train.iloc[trn_idx][features],

                               label=target.iloc[trn_idx],

                               categorical_feature=categorical_feats)

        val_data = lgb.Dataset(train.iloc[val_idx][features],

                               label=target.iloc[val_idx],

                               categorical_feature=categorical_feats)

    

        param = {

            'num_leaves': int(num_leaves),

            'min_data_in_leaf': int(min_data_in_leaf), 

            'objective':'regression',

            'max_depth': int(max_depth),

            'learning_rate': 0.01,

            "boosting": "gbdt",

            "feature_fraction": feature_fraction,

            "bagging_freq": 1,

            "bagging_fraction": bagging_fraction ,

            "bagging_seed": 11,

            "metric": 'rmse',

            "lambda_l1": lambda_l1,

            "verbosity": -1

        }

    

        clf = lgb.train(param,

                        trn_data,

                        10000,

                        valid_sets = [trn_data, val_data],

                        verbose_eval=500,

                        early_stopping_rounds = 200)

        

        oof[val_idx] = clf.predict(train.iloc[val_idx][features],

                                   num_iteration=clf.best_iteration)

        

        del clf, trn_idx, val_idx

        gc.collect()

        

    return -mean_squared_error(oof, target)**0.5
LGB_BO = BayesianOptimization(LGB_CV, {

    'max_depth': (4, 10),

    'num_leaves': (5, 130),

    'min_data_in_leaf': (10, 150),

    'feature_fraction': (0.7, 1.0),

    'bagging_fraction': (0.7, 1.0),

    'lambda_l1': (0, 6)

    })
print('-'*126)



start_time = timer(None)

with warnings.catch_warnings():

    warnings.filterwarnings('ignore')

    LGB_BO.maximize(init_points=2, n_iter=20, acq='ei', xi=0.0)

timer(start_time)