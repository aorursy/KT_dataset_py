import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", 25)

import warnings
warnings.filterwarnings("ignore")

np.set_printoptions(suppress=True)
os.listdir("../input/open-shopee-code-league-marketing-analytics")
train = pd.read_csv("../input/open-shopee-code-league-marketing-analytics/train.csv")
test = pd.read_csv("../input/open-shopee-code-league-marketing-analytics/test.csv")
users = pd.read_csv("../input/open-shopee-code-league-marketing-analytics/users.csv")
traintest = train.append(test)
traintest = traintest.merge(users, on = 'user_id', how= 'left')
traintest.columns
used_cols= [
            'subject_line_length', 
            'open_count_last_10_days', 
            'open_count_last_30_days', 
            'open_count_last_60_days',
           ]
traintest = traintest[['row_id', 'open_flag'] + used_cols ]
traintest[used_cols].describe()
traintest[used_cols].quantile([.75,.95, .999, 1.0])
traintest.shape[0] * (1 - .999)
cuurent_cols = [ 'open_count_last_10_days', 'open_count_last_30_days', 'open_count_last_60_days']
current_threshold = [12, 30, 56]

for col, val in zip(cuurent_cols, current_threshold):
    print("Change", (traintest[col] > val).sum(), "Rows from", col)
    traintest.loc[traintest[col] > val, col] = val
X = traintest[used_cols]
X['20_interval'] = X['open_count_last_30_days'] - X['open_count_last_10_days'] 
X['30_interval'] = X['open_count_last_60_days'] - X['open_count_last_30_days'] 
X['50_interval'] = X['open_count_last_60_days'] - X['open_count_last_10_days'] 

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X)
from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=3, include_bias= False, interaction_only= True)
X = pf.fit_transform(X)
X = pd.DataFrame(X)
X.shape
from itertools import combinations 
import random
random.seed(2020)
total_col = X.shape[1]

new_feature_name = total_col

comb = combinations(range(total_col), 2) 
for a,b in comb:
    if random.random() < 0.20: # only take 20% of combination randomly
        X[new_feature_name] = X[a] - X[b]
        new_feature_name += 1

print(X.shape)
traintest = traintest[['row_id', 'open_flag']]
X = pd.DataFrame(X)
traintest = pd.concat([traintest.reset_index(drop = True), X], axis = 1)
# columns for model
cols = list(X.columns )
len(cols)
# source :https://www.kaggle.com/fabiendaniel/detecting-malwares-with-lgbm
import lightgbm as lgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import matthews_corrcoef

import time

train = traintest.loc[~traintest.open_flag.isnull()]
test = traintest.loc[traintest.open_flag.isnull()]
print(train.shape)
print(test.shape)
X_full = train.drop("open_flag", axis = 1)
y_full = train.open_flag

X_train, X_valid, y_train, y_valid = train_test_split(X_full, y_full , random_state = 123, 
                                stratify = y_full, 
                                test_size = 0.2)

X_test = test.drop("open_flag", axis = 1)
print(X_train.shape) 
print(X_valid.shape)
print(X_test.shape)
X_train[:4]
param = {'num_leaves': 10,
         'min_data_in_leaf': 5, 
         'objective':'binary',
         'max_depth': -1,
         'learning_rate': 0.005,
         "boosting": "gbdt",
         "feature_fraction": 0.8,
         "bagging_freq": 1,
         "bagging_fraction": 0.8 ,
         "bagging_seed": 2020,
         "metric": 'auc',
         "tree_learner": "voting",
         "lambda_l1": 0.3,
         "random_state": 2020,
         "verbosity": -1}

# manual folds
max_iter = 5
folds = KFold(n_splits=max_iter, shuffle=True, random_state=2020)
oof = np.zeros(len(X_train))
predictions = np.zeros(len(X_valid))
predictions_test = np.zeros(len(X_test))

start = time.time()
feature_importance_df = pd.DataFrame()
start_time= time.time()
score = [0 for _ in range(folds.n_splits)]

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train.values, y_train.values)):
    print("fold n°{}".format(fold_))
    trn_data = lgb.Dataset(X_train.iloc[trn_idx][cols],
                           label=y_train.iloc[trn_idx],
                          )
    val_data = lgb.Dataset(X_train.iloc[val_idx][cols],
                           label=y_train.iloc[val_idx],
                          )

    num_round = 10000
    clf = lgb.train(param,
                    trn_data,
                    num_round,
                    valid_sets = [trn_data, val_data],
                    verbose_eval=250,
                    early_stopping_rounds = 200)
    
    oof[val_idx] = (clf.predict(X_train.iloc[val_idx][cols], num_iteration=clf.best_iteration) > 0.5).astype(int)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = cols
    fold_importance_df["importance"] = clf.feature_importance(importance_type='gain')
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
   
    # print("time elapsed: {:<5.2}s".format((time.time() - start_time) / 3600))
    score[fold_] = matthews_corrcoef(y_train.iloc[val_idx], oof[val_idx])
    
    predictions += clf.predict(X_valid[cols], num_iteration=clf.best_iteration)
    predictions_test += clf.predict(X_test[cols], num_iteration=clf.best_iteration)
    if fold_ == max_iter - 1: break
        
if (folds.n_splits == max_iter):
    print("CV score: {:<8.5f}".format(matthews_corrcoef(y_train, oof)))
else:
    print("CV score: {:<8.5f}".format(sum(score) / max_iter))
        
prediction_end = ((predictions / ((fold_ + 1) / 2) ) > 0.5).astype(int) 
print("VALID score: {:<8.5f} ".format(matthews_corrcoef(y_valid, prediction_end)))

predictions_test_end = ((predictions_test / ((fold_ + 1) / 2) ) > 0.5).astype(int) 

# pred_test = model.predict(test.drop("open_flag", axis = 1))
test['open_flag'] = predictions_test_end
test['open_flag'] = test['open_flag'].astype(int)
test['open_flag'].value_counts()
test[['row_id', 'open_flag']].to_csv("baseline_lgbm_4_features.csv", index = False)