def add_columns(df):

    for col in df.columns:

        # Normalize the data, so that it can be used in norm.cdf(), as though it is a standard normal variable

        df[col] = ((df[col] - df[col].mean()) / df[col].std()).astype('float32')



        # Square

        df[col+'_s'] = df[col] * df[col]



        # Cube

        df[col+'_c'] = df[col] * df[col] * df[col]



        # 4th power

        df[col+'_q'] = df[col] * df[col] * df[col] * df[col]



        # Cumulative percentile (not normalized)

        df[col+'_r'] = rankdata(df[col]).astype('float32')



        # Cumulative normal percentile

        df[col+'_n'] = norm.cdf(df[col]).astype('float32')

    for col in df.columns:

        df[col] = ((df[col] - df[col].mean()) / df[col].std()).astype('float32')

import pandas as pd

import numpy as np

from sklearn.model_selection import StratifiedKFold

from scipy.stats import rankdata

import lightgbm as lgb

from sklearn import metrics

import gc

import warnings

from scipy.stats import norm, rankdata

from scipy.stats import norm, rankdata

from sklearn.decomposition import PCA



pd.set_option('display.max_columns', 200)
train_df = pd.read_csv('../input/train.csv')



test_df = pd.read_csv('../input/test.csv')
test_df.head()
target = train_df.target

ID_code = test_df.ID_code



train_df = train_df.drop('target',axis = 1)

train_df = train_df.drop('ID_code',axis=1)

test_df = test_df.drop('ID_code',axis=1)
pca = PCA(n_components=1, copy=True,  random_state=4)

pca.fit(pandas.concat([train_df,test_df]))



P1 = pca.transform(train_df)

P2 = pca.transform(test_df)

len_train = len(train_df)

df = pd.concat([train_df,test_df])

add_columns(df)



train_df = df[:len_train]

test_df = df[len_train:]

del df
train_df['P1'] = P1[:,0]

test_df['P1'] = P2[:,0]

predictors = train_df.columns.values.tolist()
bayesian_tr_index, bayesian_val_index  = list(StratifiedKFold(n_splits=2, shuffle=True, random_state=1).split(train_df, target))[0]
# bayes_params ={'feature_fraction': 0.058874549652091476,

#  'lambda_l1': 3.953340038056426,

#  'lambda_l2': 3.1010968272896537,

#  'learning_rate': 0.014946929978539511,

#  'max_depth': 19.47829334300404,

#  'min_data_in_leaf': 19.88287907569752,

#  'min_gain_to_split': 0.9527282592922265,

#  'min_sum_hessian_in_leaf': 0.0027063753098792356,

#  'num_leaves': 34.454229480072925}




# param_lgb = {

#         'num_leaves': int(bayes_params['num_leaves']), # remember to int here

#         'max_bin': 63,

#         'min_data_in_leaf': int(bayes_params['min_data_in_leaf']), # remember to int here

#         'learning_rate': bayes_params['learning_rate'],

#         'min_sum_hessian_in_leaf': bayes_params['min_sum_hessian_in_leaf'],

#         'bagging_fraction': 1.0, 

#         'bagging_freq': 5, 

#         'feature_fraction': bayes_params['feature_fraction'],

#         'lambda_l1': bayes_params['lambda_l1'],

#         'lambda_l2': bayes_params['lambda_l2'],

#         'min_gain_to_split': bayes_params['min_gain_to_split'],

#         'max_depth': int(bayes_params['max_depth']), # remember to int here

#         'save_binary': True,

#         'seed': 1337,

#         'feature_fraction_seed': 1337,

#         'bagging_seed': 1337,

#         'drop_seed': 1337,

#         'data_random_seed': 1337,

#         'objective': 'binary',

#         'boosting_type': 'gbdt',

#         'verbose': 1,

#         'metric': 'auc',

#         'is_unbalance': True,

#         'boost_from_average': False,

#     }
# param_lgb = {

#          'num_leaves': 6, # remember to int here

#          'max_bin': 63,

#          'min_data_in_leaf': 20, # remember to int here

#          'learning_rate': 0.018,

#          'min_sum_hessian_in_leaf': 0.009,

#          'bagging_fraction': 1.0, 

#          'bagging_freq': 5, 

#          'feature_fraction': 0.075,

#          'lambda_l1': 1.569,

#          'lambda_l2': 3.9436,

#          'min_gain_to_split': 0.0006,

#          'max_depth': 20, # remember to int here

#          'save_binary': True,

#          'seed': 1337,

#          'feature_fraction_seed': 1337,

#          'bagging_seed': 1337,

#          'drop_seed': 1337,

#          'data_random_seed': 1337,

#          'objective': 'binary',

#          'boosting_type': 'gbdt',

#          'verbose': 1,

#          'metric': 'auc',

#          'is_unbalance': True,

#          'boost_from_average': False,

#      }
#v8 bayes optimization https://www.kaggle.com/poppins/lgb-bayesian-parameters-finding-rank-average

params={'feature_fraction': 0.05303293150022274,

 'lambda_l1': 4.742447900648306,

 'lambda_l2': 3.797046693025151,

 'learning_rate': 0.012061342816038765,

 'max_depth': 14.202377588960745,

 'min_data_in_leaf': 9.784931229481742,

 'min_gain_to_split': 0.06340137767250764,

 'min_sum_hessian_in_leaf': 0.00672958286082188,

 'num_leaves': 14.036858809140814}
param_lgb = {

        'num_leaves': int(params['num_leaves']), # remember to int here

        'max_bin': 63,

        'min_data_in_leaf': int(params['min_data_in_leaf']), # remember to int here

        'learning_rate': params['learning_rate'],

        'min_sum_hessian_in_leaf': params['min_sum_hessian_in_leaf'],

        'bagging_fraction': 1.0, 

        'bagging_freq': 5, 

        'feature_fraction': params['feature_fraction'],

        'lambda_l1': params['lambda_l1'],

        'lambda_l2': params['lambda_l2'],

        'min_gain_to_split': params['min_gain_to_split'],

        'max_depth': int(params['max_depth']), # remember to int here

        'save_binary': True,

        'seed': 1337,

        'feature_fraction_seed': 1337,

        'bagging_seed': 1337,

        'drop_seed': 1337,

        'data_random_seed': 1337,

        'objective': 'binary',

        'boosting_type': 'gbdt',

        'verbose': 1,

        'metric': 'auc',

        'is_unbalance': True,

        'boost_from_average': False,

    }

nfold = 5
gc.collect()
skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=2019)
oof = np.zeros(len(train_df))

predictions = np.zeros((len(test_df),nfold))



i = 1

for train_index, valid_index in skf.split(train_df, target.values):

    print("\nfold {}".format(i))

    xg_train = lgb.Dataset(train_df.iloc[train_index][predictors].values,

                           label=target.iloc[train_index].values,

                           feature_name=predictors,

                           free_raw_data = False

                           )

    xg_valid = lgb.Dataset(train_df.iloc[valid_index][predictors].values,

                           label=target.iloc[valid_index].values,

                           feature_name=predictors,

                           free_raw_data = False

                           )   



    

    clf = lgb.train(param_lgb, xg_train, 10000, valid_sets = [xg_valid], verbose_eval=1000)

    oof[valid_index] = clf.predict(train_df.iloc[valid_index][predictors].values, num_iteration=clf.best_iteration) 

    

    predictions[:,i-1] += clf.predict(test_df[predictors], num_iteration=clf.best_iteration)

    i = i + 1



print("\n\nCV AUC: {:<0.6f}".format(metrics.roc_auc_score(target.values, oof)))
predictions
print("Rank averaging on", nfold, "fold predictions") 

rank_predictions = np.zeros((predictions.shape[0],1))

for i in range(nfold):

    rank_predictions[:, 0] = np.add(rank_predictions[:, 0], rankdata(predictions[:, i].reshape(-1,1))/rank_predictions.shape[0]) 



rank_predictions /= nfold
sub_df = pd.DataFrame({"ID_code": ID_code.values})

sub_df["target"] = rank_predictions

sub_df[:10]
sub_df.to_csv("Customer_Transaction_rank_predictions.csv", index=False)