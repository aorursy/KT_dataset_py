# Data manipulation and set-up

import numpy as np

import pandas as pd



# Visualisation

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(font_scale=1)



# Modelling (including set-up & evaluation)

from sklearn.model_selection import KFold, StratifiedKFold

import lightgbm as lgb

from sklearn.metrics import roc_auc_score



# Ignore warnings

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

warnings.filterwarnings('ignore')
random_state = 42

np.random.seed(random_state)

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
# Data augmentation

def augment(x,y,t=2):

    xs,xn = [],[]

    for i in range(t):

        mask = y>0

        x1 = x[mask].copy()

        ids = np.arange(x1.shape[0])

        for c in range(x1.shape[1]):

            np.random.shuffle(ids)

            x1[:,c] = x1[ids][:,c]

        xs.append(x1)



    for i in range(t//2):

        mask = y==0

        x1 = x[mask].copy()

        ids = np.arange(x1.shape[0])

        for c in range(x1.shape[1]):

            np.random.shuffle(ids)

            x1[:,c] = x1[ids][:,c]

        xn.append(x1)



    xs = np.vstack(xs)

    xn = np.vstack(xn)

    ys = np.ones(xs.shape[0])

    yn = np.zeros(xn.shape[0])

    x = np.vstack([x,xs,xn])

    y = np.concatenate([y,ys,yn])

    return x,y
# Model parameters

lgb_params = {

    "objective" : "binary",

    "metric" : "auc",

    "boosting": 'gbdt',

    "max_depth" : -1,

    "num_leaves" : 13,

    "learning_rate" : 0.01,

    "bagging_freq": 5,

    "bagging_fraction" : 0.4,

    "feature_fraction" : 0.05,

    "min_data_in_leaf": 80,

    "min_sum_heassian_in_leaf": 10,

    "tree_learner": "serial",

    "boost_from_average": "false",

    "bagging_seed" : random_state,

    "verbosity" : 1,

    "seed": random_state}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

oof = train[['ID_code', 'target']]

oof['predict'] = 0

predictions = test[['ID_code']]

val_aucs = []

feature_importance = pd.DataFrame()
features = [col for col in train.columns if col not in ['target', 'ID_code']]

X_test = test[features].values
for fold, (trn_idx, val_idx) in enumerate(skf.split(train, train['target'])):

    X_train, y_train = train.iloc[trn_idx][features], train.iloc[trn_idx]['target']

    X_valid, y_valid = train.iloc[val_idx][features], train.iloc[val_idx]['target']

    

    N = 5

    p_valid,yp = 0,0

    for i in range(N):

        X_t, y_t = augment(X_train.values, y_train.values)

        X_t = pd.DataFrame(X_t)

        X_t = X_t.add_prefix('var_')

    

        trn_data = lgb.Dataset(X_t, label=y_t)

        val_data = lgb.Dataset(X_valid, label=y_valid)

        evals_result = {}

        lgb_clf = lgb.train(lgb_params,

                        trn_data,

                        100000,

                        valid_sets = [trn_data, val_data],

                        early_stopping_rounds=3000,

                        verbose_eval = 1000,

                        evals_result=evals_result

                       )

        p_valid += lgb_clf.predict(X_valid)

        yp += lgb_clf.predict(X_test)

    fold_importance = pd.DataFrame()

    fold_importance["feature"] = features

    fold_importance["importance"] = lgb_clf.feature_importance()

    fold_importance["fold"] = fold + 1

    feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    oof['predict'][val_idx] = p_valid/N

    val_score = roc_auc_score(y_valid, p_valid)

    val_aucs.append(val_score)

    

    predictions['fold{}'.format(fold+1)] = yp/N
# Submission

predictions['target'] = np.mean(predictions[[col for col in predictions.columns if col not in ['ID_code', 'target']]].values, axis=1)

predictions.to_csv('lgb_all_predictions.csv', index=None)

sub = pd.DataFrame({"ID_code":test["ID_code"].values})

sub["target"] = predictions['target']

sub.to_csv("lgb_submission.csv", index=False)

oof.to_csv('lgb_oof.csv', index=False)