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
import lightgbm as lgb

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

import warnings

warnings.filterwarnings('ignore')

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
pd.set_option('display.max_columns',500)
train_df.head()
test_df.head()
print(train_df.shape,test_df.shape)
import xgboost as xgb

from matplotlib import pyplot
XGBOOST_PARAM = {

    'random_state' : 42,

    'n_estimators' : 25, 

    'learning_rate': 0.15,   ### importtant

    'num_leaves': 36,        ### important

    'max_depth': 6,         ### important

    'metric' : ['auc'],

    'reg_alpha' : 2.03,     ### important

    'reg_lambda' : 4.7,     ### important  

    'feature_fraction' : 0.8, #colsample_bytree, important

    'feature_fraction_seed' : 42,   

    'max_bins' : 100,         

    'min_split_gain': 0.0148,   

    'min_child_weight' : 7.835, #min_sum_hessian_in_leaf 

    'min_data_in_leaf' : 1000, #min_child_samples

    'random_state' : 1981, # Updated from 'seed'

    'subsample' : .912, #also known as Bagging fraction!

    'subsample_freq' : 200, # also known as bagging frequency!

    'boost_from_average' : False,

    'verbose_eval' : 10,

    'is_unbalance' : True ### to address class imbalance

    }
XGBGBDT = xgb.XGBClassifier(**XGBOOST_PARAM,

                            silent=0,

                            )
features = [c for c in train_df.columns if c not in ['ID_code', 'target']]

target = train_df['target']

folds = StratifiedKFold(n_splits=3, shuffle=False, random_state=2319)

oof = np.zeros(len(train_df))

predictions = np.zeros(len(test_df))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):

    print("Fold {}".format(fold_))

    trn_X = train_df.iloc[trn_idx][features]

    trn_Y = target.iloc[trn_idx]

    val_X = train_df.iloc[val_idx][features]

    val_Y = target.iloc[val_idx]

    

    

    clf = XGBGBDT.fit(trn_X, trn_Y,

                      eval_set=[(trn_X, trn_Y),(val_X, val_Y)],

                        eval_metric='auc',early_stopping_rounds=100,

                        verbose=25)

            

    oof[val_idx] = clf.predict_proba(train_df.iloc[val_idx][features])[:,1]

    predictions += clf.predict_proba(test_df[features])[:,1] / folds.n_splits

print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))

sub = pd.DataFrame({"ID_code": test_df.ID_code.values})

sub["target"] = predictions

sub.to_csv("submission_XGB.csv", index=False)
# feature importance

print(clf.feature_importances_)

# plot

pyplot.bar(range(len(clf.feature_importances_)), clf.feature_importances_)

pyplot.show()
feature_importance_df = pd.DataFrame()

param = {

    'bagging_freq': 5,  'bagging_fraction': 0.331,  'boost_from_average':'false',   

    'boost': 'gbdt',    'feature_fraction': 0.0405, 'learning_rate': 0.0083,

    'max_depth': -1,    'metric':'auc',             'min_data_in_leaf': 80,     

    'min_sum_hessian_in_leaf': 10.0,'num_leaves': 13,  'num_threads': 8,            

    'tree_learner': 'serial',   'objective': 'binary',       'verbosity': 1

}

folds = StratifiedKFold(n_splits=3, shuffle=False, random_state=42)

oof = np.zeros(len(train_df))

predictions = np.zeros(len(test_df))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):

    print("Fold {}".format(fold_))

    trn_data = lgb.Dataset(train_df.iloc[trn_idx][features], label=target.iloc[trn_idx])

    val_data = lgb.Dataset(train_df.iloc[val_idx][features], label=target.iloc[val_idx])

    clf = lgb.train(param, trn_data, 5000, valid_sets = [trn_data, val_data], verbose_eval=500, early_stopping_rounds = 250)

    oof[val_idx] = clf.predict(train_df.iloc[val_idx][features], num_iteration=clf.best_iteration)

    predictions += clf.predict(test_df[features], num_iteration=clf.best_iteration) / folds.n_splits

    

    ### variable importance plot utility

    fold_importance_df = pd.DataFrame()

    fold_importance_df["feature"] = features

    fold_importance_df["importance"] = clf.feature_importance()

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    

print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))

sub = pd.DataFrame({"ID_code": test_df.ID_code.values})

sub["target"] = predictions

sub.to_csv("submission_LGBM.csv", index=False)
import matplotlib.pyplot as plt

import seaborn as sns



cols = (feature_importance_df[["feature", "importance"]]

        .groupby("feature")

        .mean()

        .sort_values(by="importance", ascending=False)[:1000].index)

best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]



plt.figure(figsize=(14,26))

sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance",ascending=False))

plt.title('LightGBM Features (averaged over folds)')

plt.tight_layout()

plt.savefig('lgbm_importances.png')


from catboost import CatBoostClassifier



MAX_ROUNDS = 650

OPTIMIZE_ROUNDS = False

LEARNING_RATE = 0.05

model = CatBoostClassifier(

    learning_rate=LEARNING_RATE, 

    depth=6, 

    l2_leaf_reg = 14, 

    iterations = MAX_ROUNDS,

#    verbose = True,

    loss_function='Logloss'

)
features = [c for c in train_df.columns if c not in ['ID_code', 'target']]

target = train_df['target']

folds = StratifiedKFold(n_splits=3, shuffle=False, random_state=2319)

oof = np.zeros(len(train_df))

predictions = np.zeros(len(test_df))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):

    print("Fold {}".format(fold_))

    trn_X = train_df.iloc[trn_idx][features]

    trn_Y = target.iloc[trn_idx]

    val_X = train_df.iloc[val_idx][features]

    val_Y = target.iloc[val_idx]

    

    

    clf = model.fit(trn_X, trn_Y,

                      eval_set=[(trn_X, trn_Y),(val_X, val_Y)],

                        use_best_model=True)

            

    oof[val_idx] = clf.predict_proba(train_df.iloc[val_idx][features])[:,1]

    predictions += clf.predict_proba(test_df[features])[:,1] / folds.n_splits

print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))

sub = pd.DataFrame({"ID_code": test_df.ID_code.values})

sub["target"] = predictions

sub.to_csv("submission_catboost.csv", index=False)