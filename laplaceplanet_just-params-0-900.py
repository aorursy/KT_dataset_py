from sklearn.model_selection import KFold,StratifiedKFold,KFold

from scipy.stats import norm, rankdata

import lightgbm as lgb

from sklearn.metrics import roc_auc_score

import numpy as np

import pandas as pd

import time

import gc

random_state = 13

np.random.seed(random_state)
print('read data')

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

len(df_train[df_train.target == 1]) / len(df_train)

df_train = df_train.round(3)

df_test = df_test.round(3)
df_train_one = df_train[df_train.target == 1]

df_train_zero = df_train[df_train.target == 0]

one_mean = df_train_one.describe().loc['mean'] 

zero_mean = df_train_zero.describe().loc['mean'] 

diff = one_mean - zero_mean

top_features = [c for c in (diff.sort_values().head(5).index)] 

tail_features = [c for c in diff.sort_values().tail(5).index] 

imp_features = top_features + tail_features

top_mean = [c for c in (diff.sort_values().head(5).values)] 

tail_mean = [c for c in diff.sort_values().tail(5).values] 

imp_mean = top_mean + tail_mean

print(imp_mean)

print(imp_features)
test_ID = df_test['ID_code'].values

Y = df_train.target.values.astype(np.float32)

target = df_train.target

df_train = df_train.drop(['ID_code','target'], axis=1)

df_test = df_test.drop(['ID_code'], axis=1)

original_features = df_train.columns
# len_train = len(df_train)

# merged = pd.concat([df_train, df_test])

# del df_test, df_train

# gc.collect()

# for col in imp_features:

#     # Normalize the data, so that it can be used in norm.cdf(), 

#     # as though it is a standard normal variable

#     merged[col] = ((merged[col] - merged[col].mean()) 

#     / merged[col].std()).astype('float32')

#     # Square

#     merged[col+'^2'] = merged[col] * merged[col]

#     # Cube

#     merged[col+'^3'] = merged[col] * merged[col] * merged[col]

#     # 4th power

#     merged[col+'^4'] = merged[col] * merged[col] * merged[col] * merged[col]

#     # Cumulative percentile (not normalized)

#     merged[col+'_cp'] = rankdata(merged[col]).astype('float32')

#     # Cumulative normal percentile

#     merged[col+'_cnp'] = norm.cdf(merged[col]).astype('float32')

    

# important features statics information

# merged['imp_features_mean'] = merged[imp_features].mean(axis = 1)

# merged['imp_features_sum'] = merged[imp_features].sum(axis = 1)

# merged['imp_features_max'] = merged[imp_features].max(axis = 1)

# merged['imp_features_min'] = merged[imp_features].min(axis = 1)

# merged['imp_features_var'] = merged[imp_features].var(axis = 1)

# merged['imp_features_median'] = merged[imp_features].median(axis = 1)



# # all features statics information

# merged['all_features_mean'] = merged.mean(axis = 1)

# merged['all_features_sum'] = merged.sum(axis = 1)

# merged['all_features_max'] = merged.max(axis = 1)

# merged['all_features_min'] = merged.min(axis = 1)

# merged['all_features_var'] = merged.var(axis = 1)

# merged['all_features_median'] = merged.median(axis = 1)



# + - * /

# for i in range(len(imp_features)):

#     for j in range(i+1,len(imp_features)):

#         merged[imp_features[i] + 'minus' + imp_features[j]] = merged[imp_features[i]] - merged[imp_features[j]]  

#         merged[imp_features[i] + 'plus' + imp_features[j]] = merged[imp_features[i]] + merged[imp_features[j]]  

#         merged[imp_features[i] + 'multi' + imp_features[j]] = merged[imp_features[i]] * merged[imp_features[j]]  

#         merged[imp_features[i] + 'divide' + imp_features[j]] = merged[imp_features[i]] / merged[imp_features[j]]  

        

# new_features = set(merged.columns) - set(original_features)

# for col in imp_features:

#     merged[col] = ((merged[col] - merged[col].mean()) 

#     / merged[col].std()).astype('float32')

# df_train = merged.iloc[:len_train]

# df_test = merged.iloc[len_train:]

# df_train.shape
# for df in [df_train,df_test]:

#     for col in imp_features:

#         df[col + '_category'] = df[col].round(0).astype('category')

# category_features = [col + '_category' for col in imp_features]

# df_train[category_features].head()
# # count features

# for df in[df_train,df_test]:

#     df['positive'] = df.apply(lambda x: sum(x > 0),axis = 1)

#     df['negative'] = df.apply(lambda x: sum(x < 0),axis = 1)
# param = {

#     "objective" : "binary",

#     "metric" : 'auc',

#     "max_depth" : 2,

#     "num_leaves" : 2,

#     "learning_rate" : 0.055,

#     "bagging_fraction" : 0.3,

#     "feature_fraction" : 0.15,

#     "lambda_l1" : 5,

#     "lambda_l2" : 5,

#     "bagging_seed" : 42,

#     "verbosity" : 1,

#     "random_state": 4950 

# }

param = {

    "objective" : "binary",

    "metric" : 'auc',

#      'boost' : "gbdt",

    'boost_from_average' : "false",

#     'tree_learner': "serial",

    "max_depth" : -1,

    "num_leaves" : 13,

    "learning_rate" : 0.01,

    "bagging_fraction" : 0.4,

     'bagging_freq' : 5,

     'min_data_in_leaf' : 80,

     'min_sum_hessian_in_leaf' : 10.0,

    "feature_fraction" : 0.05,

#     "lambda_l1" : 5,

#     "lambda_l2" : 5,

    "bagging_seed" : 42,

    "verbosity" : 1,

    "random_state": 4950,

     'num_threads': 8,

}
# from imblearn.over_sampling import SMOTE

# sm = SMOTE(random_state=2,sampling_strategy = 0.115 ,k_neighbors = 8,n_jobs = 4)

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=31415)

oof = np.zeros(len(df_train))

predictions = np.zeros(len(df_test))

# feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train.values, target.values)):

    print("Fold {}".format(fold_))

#     X_train_res, y_train_res = sm.fit_sample(df_train.iloc[trn_idx],target.iloc[trn_idx])

#     print(sum(target.iloc[trn_idx]) / len(df_train.iloc[trn_idx]))

#     print(sum(y_train_res) / len(df_train.iloc[trn_idx]))

    trn_data = lgb.Dataset(df_train.iloc[trn_idx],target.iloc[trn_idx])

    val_data = lgb.Dataset(df_train.iloc[val_idx], label=target.iloc[val_idx])

    # watchlist is xgb version

    # watchlist = [(trn_data, 'train'), (val_data, 'valid')]

    lgb_model = lgb.train(param, trn_data, 40000, 

                          valid_sets = [trn_data, val_data], 

                          early_stopping_rounds=2000, 

                          verbose_eval=1000)

#                          categorical_feature= category_features )          

    oof[val_idx] = lgb_model.predict(df_train.iloc[val_idx], num_iteration = lgb_model.best_iteration)

    predictions += lgb_model.predict(df_test, num_iteration = lgb_model.best_iteration) / folds.n_splits

#     fold_importance_df = pd.DataFrame()

#     fold_importance_df["Feature"] = features

#     fold_importance_df["importance"] = clf.feature_importance()

#     fold_importance_df["fold"] = fold_ + 1

#     feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))
# clf = lgb.train(param, trn_data, len(lgb_cv["auc-mean"]), valid_sets=(trn_data), verbose_eval=1000)

# y_pred = clf.predict(df_test, num_iteration=clf.best_iteration)
print('save result.')

pd.DataFrame({'ID_code':test_ID,'target':predictions}).to_csv('submission.csv',index=False)

print('done.')