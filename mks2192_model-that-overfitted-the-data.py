import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb

from sklearn.metrics import mean_squared_error

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

import warnings

warnings.filterwarnings('ignore')

import random

from tqdm import tqdm

from catboost import CatBoostRegressor

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from tqdm import tqdm

from skhep.modeling.bayesian_blocks import bayesian_blocks

import pickle

from glob import glob



df_train = pd.read_csv("../input/santander-customer-transaction-prediction/train.csv")

df_test = pd.read_csv('../input/santander-customer-transaction-prediction/test.csv')



ID_code = df_test.ID_code.values



target = df_train.target



import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.



pickle_list_p_01 = glob("../input/baysial-blcok-p-01/*.pkl")


for k in tqdm(pickle_list_p_01):

    pickle_file = open(k, 'rb')

    bins1 = pickle.load(pickle_file)

    Var = int(k[28:-4])+1

    #print(Var)



    for count, p in enumerate(range(Var-20, Var)):

        var = "var_" + str(p)

        mag_var = "mag_var_" + str(p)



        df_train[mag_var] = 0

        df_test[mag_var] = 0



        binn = bins1[count]



        for i in range(len(binn)):

            try:

                Idx_train = df_train[(df_train[var] >= binn[i]) & (df_train[var] < binn[i+1])].index#[mag_var] = bins[i]

                df_train[mag_var].loc[Idx_train] = binn[i]



                Idx_test = df_test[(df_test[var] >= binn[i]) & (df_test[var] < binn[i+1])].index#[mag_var] = bins[i]

                df_test[mag_var].loc[Idx_test] = binn[i]



            except:

                pass





for i in tqdm(range(200)):



    train_one = df_train[df_train.target ==1]

    train_zero = df_train[df_train.target ==0]



    s_1 = pd.Series(train_one['mag_var_'+ str(i)].value_counts())

    s_0 = pd.Series(train_zero['mag_var_'+ str(i)].value_counts())



    map_ratio = s_1/s_0



    dict1 = map_ratio.to_dict()

    df_train['mag_var_'+ str(i)] = pd.CategoricalIndex(df_train['mag_var_'+ str(i)]).map(dict1)



    df_test['mag_var_'+ str(i)] = pd.CategoricalIndex(df_test['mag_var_'+ str(i)]).map(dict1)





for i in df_test.columns[1:]:

    df_test[i] = pd.to_numeric(df_test[i]).fillna(0)

    df_train[i] = pd.to_numeric(df_train[i]).fillna(0)



features = [c for c in df_train.columns if c not in ['ID_code', 'target']]



param = {

    'bagging_freq': 5,          'bagging_fraction': 0.335,   'boost_from_average':'false',   'boost': 'gbdt',

    'feature_fraction': 0.041,   'learning_rate':.01 ,     'max_depth': 8,                'metric':'auc',

    'min_data_in_leaf': 20,     'min_sum_hessian_in_leaf': 10.0,'num_leaves': 30,           #'num_threads': 8,

      'objective': 'binary',      'verbosity': 1, 'max_bin' : 256, 

}





train_df = df_train[features]

test_df = df_test[features]





##-----------------------------------------------------------------------------------------##

# random_state= 44000

num_folds = 5

folds = StratifiedKFold(n_splits=num_folds, shuffle=False, random_state=2319)

oof = np.zeros(len(train_df))

predictions = np.zeros(len(test_df))

##-----------------------------------------------------------------------------------------##



for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):

    print("Fold idx:{}".format(fold_ + 1))

    trn_data = lgb.Dataset(train_df.iloc[trn_idx][features], label=target.iloc[trn_idx])

    val_data = lgb.Dataset(train_df.iloc[val_idx][features], label=target.iloc[val_idx])

    clf = lgb.train(param, trn_data, 1000000, valid_sets = [trn_data, val_data], verbose_eval=200, early_stopping_rounds = 400)

    oof[val_idx] = clf.predict(train_df.iloc[val_idx][features], num_iteration=clf.best_iteration)

    predictions += clf.predict(test_df[features], num_iteration=clf.best_iteration) / folds.n_splits



print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))

##-----------------------------------------------------------------------------------------##

plt.figure(figsize=(14,50))

Se = pd.Series(clf.feature_importance(), index = features).sort_values().plot('barh')

plt.show()


prediction = clf.predict(df_test[features])



sub_df = pd.DataFrame({"ID_code":ID_code})

sub_df["target"] = prediction

sub_df.to_csv("submission_mg.csv", index=False)
