# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.metrics import roc_auc_score

from sklearn.model_selection import KFold

from sklearn.preprocessing import OneHotEncoder

from sklearn import metrics



#Light GBM

import lightgbm as lgb





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
# Replace -1 for missing



for col in train.columns: 

    train.loc[train[col] == -1, col] = np.nan 



for col in test.columns: 

    test.loc[test[col] == -1, col] = np.nan     
train.describe()
test.describe()
seed = 42



param = {'num_leaves': 31,

        'boosting_type': 'gbdt',

        'learning_rate': 0.01,

        'max_bin': 255,

        'min_data_in_leaf': 1500,

        'feature_fraction': 0.7,

        'bagging_freq': 1,

        'bagging_fraction': 0.7,

        'lambda_l1': 1, 

        'lambda_l2': 1, 

        'save_binary': True,

        'seed': seed,

        'feature_fraction_seed': seed,

        'bagging_seed': seed,

        'drop_seed': seed,

        'data_random_seed': seed,

        'objective': 'binary',  

        'verbose': 1,

        'metric': 'binary_logloss',

        'is_unbalance': False, 

        'boost_from_average': True

    }
#one-hot encoding of categorical variables



cat_names = ['ps_ind_02_cat', 'ps_ind_04_cat', 'ps_ind_05_cat', 'ps_car_01_cat',

             'ps_car_02_cat', 'ps_car_03_cat', 'ps_car_04_cat', 'ps_car_05_cat',

             'ps_car_06_cat', 'ps_car_07_cat', 'ps_car_08_cat', 'ps_car_09_cat', 

             'ps_car_10_cat', 'ps_car_11_cat']



train = pd.get_dummies(train, columns = cat_names)

test = pd.get_dummies(test, columns = cat_names)
#Remove calc featutes

calc_names = ['ps_calc_01',    'ps_calc_02',    'ps_calc_03',    'ps_calc_04',

              'ps_calc_05',    'ps_calc_06',    'ps_calc_07',   'ps_calc_08',

              'ps_calc_09',    'ps_calc_10',    'ps_calc_11',    'ps_calc_12',

              'ps_calc_13',    'ps_calc_14',    'ps_calc_15_bin','ps_calc_16_bin',

              'ps_calc_17_bin','ps_calc_18_bin','ps_calc_19_bin',

              'ps_calc_20_bin']



train.drop(columns = calc_names, inplace=True)

predictors = list(train.columns)

predictors.remove('id')

predictors.remove('target')
nfold = 5

target = 'target'

skf = KFold(n_splits=nfold, shuffle=True, random_state=2019)



oof = np.zeros(len(train))

predictions = np.zeros(len(test))
i = 1

for train_index, valid_index in skf.split(train, train.target.values):

    print("\nfold {}".format(i))

    

    #Train data

    t=train.iloc[train_index]

        

    xg_train = lgb.Dataset(t[predictors].values,

                           label=t[target].values,

                           feature_name=predictors,

                           free_raw_data = False

                           )

    

    xg_valid = lgb.Dataset(train.iloc[valid_index][predictors].values,

                           label=train.iloc[valid_index][target].values,

                           feature_name=predictors,

                           free_raw_data = False

                           )   



    num_rounds = 1400

    clf = lgb.train(param, xg_train, num_rounds, valid_sets = [xg_train, xg_valid], 

                    verbose_eval=2000, early_stopping_rounds = 1000) 

    oof[valid_index] = clf.predict(train.iloc[valid_index][predictors].values, num_iteration=clf.best_iteration) 

    

    predictions += clf.predict(test[predictors], num_iteration=clf.best_iteration) / nfold

    i = i + 1


print("\n\nCV AUC: {:<0.5f}".format(metrics.roc_auc_score(train.target.values, oof)))

print("\n\nCV log loss: {:<0.5f}".format(metrics.log_loss(train.target.values, oof)))

print("\n\nCV Gini: {:<0.5f}".format(2 * metrics.roc_auc_score(train.target.values, oof) -1))





sub_df = pd.read_csv('../input/sample_submission.csv')

sub_df["target"] = predictions

sub_df[:10]

sub_df.to_csv("lightgbm.csv", index=False)