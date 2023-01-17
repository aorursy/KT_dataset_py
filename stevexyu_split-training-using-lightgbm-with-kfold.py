import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split,StratifiedKFold

from sklearn.metrics import roc_auc_score, roc_curve,confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score

import lightgbm as lgb
train = pd.read_csv("../input/train.csv")

test = pd.read_csv('../input/test.csv')

print("train shape:", train.shape, "test.shape:", test.shape)
sns.countplot(train['target'])
total1 = train["target"].value_counts()[1]

print("There are {} target values with 1, is about {}% of total data".format(total1, 100 * total1/train.shape[0]))
features = train.columns.values[2:202]

train_df = train[features]

test_df = test[features]

print(features.shape, train_df.shape, test_df.shape)
train0 = train.loc[train['target'] == 0]

train1 = train.loc[train['target'] == 1]

print(train0.shape, train1.shape)

splitNum = 3

t0PerSplit = train0.shape[0] // splitNum

print(t0PerSplit)

splits = []

for i in range(splitNum-1):

    splits.append(pd.concat([train0[i*t0PerSplit:(i+1)*t0PerSplit], train1]).sample(frac=1))

    print(splits[i].shape)

splits.append(pd.concat([train0[(splitNum-1)*t0PerSplit:], train1]).sample(frac=1))

print(splits[splitNum-1].shape)

splits[splitNum-1].iloc[:,1].values[:100]
scaler = StandardScaler()

#x_train = scaler.fit_transform(x_train)

x_test = scaler.fit_transform(test[features])
param = {

    'bagging_freq': 5,

    'bagging_fraction': 0.333,

    'boost_from_average':'false',

    'boost': 'gbdt',

    'feature_fraction': 0.041,

    'learning_rate': 0.0083,

    'max_depth': -1,  

    'metric':'auc',

    'min_data_in_leaf': 80,

    'min_sum_hessian_in_leaf': 10.0,

    'num_leaves': 13,

    'num_threads': 8,

    'tree_learner': 'serial',

    'objective': 'binary', 

    'verbosity': 1

}
predictions = np.zeros(len(test))  

for split_ in range(splitNum):

    folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=2019)

    train_s = splits[split_][features]

    train_s =  scaler.transform(train_s)

    target_s = splits[split_]["target"]

    oof = np.zeros(len(train_s))

    prediction = np.zeros(len(test))

    feature_importance_df = pd.DataFrame()

    print(train_s.shape, target_s.shape, oof.shape)

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_s, target_s.values)):

        print("Split:", split_,  "Fold:",fold_, trn_idx.shape, val_idx.shape)

        trn_data = lgb.Dataset(train_s[trn_idx], label=target_s.iloc[trn_idx])

        val_data = lgb.Dataset(train_s[val_idx], label=target_s.iloc[val_idx])



        num_round = 1000000

        clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 3000)

        oof[val_idx] = clf.predict(train_s[val_idx], num_iteration=clf.best_iteration)

    

        fold_importance_df = pd.DataFrame()

        fold_importance_df["Feature"] = features

        fold_importance_df["importance"] = clf.feature_importance()

        fold_importance_df["fold"] = fold_ + 1

        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        print("CV score: {:<8.5f}".format(roc_auc_score(target_s, oof)))

        

        prediction += clf.predict(x_test, num_iteration=clf.best_iteration) / folds.n_splits

        

    predictions += prediction / splitNum

    
sub_df = pd.DataFrame({"ID_code":test["ID_code"].values})

sub_df["target"] = predictions

sub_df.to_csv("submission_lgb_split3.csv", index=False)
!ls
!ls .