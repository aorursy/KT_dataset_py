import gc

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.metrics import roc_auc_score

from scipy.signal import savgol_filter

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

warnings.filterwarnings('ignore')
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_df.head()
test_df.head()
features = [col for col in train_df.columns if col not in ['target', 'ID_code']]
for c in features:

    print(c)

    train_df.sort_values(by=c,inplace=True)

    train_df[c] = savgol_filter(train_df.target, window_length=5001,polyorder=1)

    test_df.sort_values(by=c,inplace=True)

    test_df[c] = savgol_filter(train_df.target, window_length=5001,polyorder=1)#Yes I am using the train set values!!!!!!
import lightgbm as lgb

random_state = 42

params = {

    "objective" : "binary", "metric" : "auc", "boosting": 'gbdt', "max_depth" : -1, "num_leaves" : 13,

    "learning_rate" : 0.01, "bagging_freq": 5, "bagging_fraction" : 0.4, "feature_fraction" : 0.05,

    "min_data_in_leaf": 80, "min_sum_hessian_in_leaf": 10, "tree_learner": "serial", "boost_from_average": "false",

    "bagging_seed" : random_state, "verbosity" : 1, "seed": random_state, "n_jobs":4

}

noOfFolds = 5

skf = StratifiedKFold(n_splits=noOfFolds, shuffle=True, random_state=random_state)

oof = np.zeros(train_df.shape[0])

predictions = np.zeros(test_df.shape[0])

val_aucs = []





X_test = test_df[features].values



for fold, (trn_idx, val_idx) in enumerate(skf.split(train_df, train_df.target)):

    X_train, y_train = train_df.iloc[trn_idx][features], train_df.iloc[trn_idx]['target']

    X_valid, y_valid = train_df.iloc[val_idx][features], train_df.iloc[val_idx]['target']

    

      

    trn_data = lgb.Dataset(X_train, label=y_train)

    val_data = lgb.Dataset(X_valid, label=y_valid)

    evals_result = {}

    lgb_clf = lgb.train(params,trn_data,100000,valid_sets = [trn_data, val_data],early_stopping_rounds=1000,verbose_eval = 5000,evals_result=evals_result)

    

    p_valid = lgb_clf.predict(X_valid)

    oof[val_idx] = p_valid

    predictions += lgb_clf.predict(X_test)/noOfFolds

    val_score = roc_auc_score(y_valid, p_valid)

    val_aucs.append(val_score)

    

roc_auc_score(train_df.target, oof)

f = pd.DataFrame()

f['ID_code'] = test_df.ID_code.values

f['target'] = predictions

f.to_csv('rankbadcompetition.csv',index=False)