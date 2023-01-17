from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import pandas as pd

import lightgbm as lgb

from sklearn.model_selection import KFold

from sklearn import model_selection, preprocessing, metrics

import os

print(os.listdir("../input"))

print(os.listdir("../input/lanl-master-s-features-creating-0"))

train_X_0 = pd.read_csv("../input/lanl-master-s-features-creating-0/train_X_features_865.csv")

train_X_1 = pd.read_csv("../input/lanl-master-s-features-creating-1/train_X_features_865.csv")

y_0 = pd.read_csv("../input/lanl-master-s-features-creating-0/train_y.csv", index_col=False,  header=None)

y_1 = pd.read_csv("../input/lanl-master-s-features-creating-1/train_y.csv", index_col=False,  header=None)

train_X = pd.concat([train_X_0, train_X_1], axis=0)

train_X = train_X.reset_index(drop=True)

print(train_X.shape)

train_X.head()



y = pd.concat([y_0, y_1], axis=0)

y = y.reset_index(drop=True)

y[0].shape



train_y = pd.Series(y[0].values)





test_X = pd.read_csv("../input/lanl-master-s-features-creating-0/test_X_features_10.csv")

# del X["seg_id"], test_X["seg_id"]



scaler = StandardScaler()

train_columns = train_X.columns



train_X[train_columns] = scaler.fit_transform(train_X[train_columns])

test_X[train_columns] = scaler.transform(test_X[train_columns])
features = train_X.columns

train_X['target'] = 0

test_X['target'] = 1
train_test = pd.concat([train_X, test_X], axis =0)



target = train_test['target'].values
param = {'num_leaves': 50,

         'min_data_in_leaf': 30, 

         'objective':'binary',

         'max_depth': 5,

         'learning_rate': 0.006,

         "min_child_samples": 20,

         "boosting": "gbdt",

         "feature_fraction": 0.9,

         "bagging_freq": 1,

         "bagging_fraction": 0.9 ,

         "bagging_seed": 27,

         "metric": 'auc',

         "verbosity": -1}



folds = KFold(n_splits=5, shuffle=True, random_state=15)

oof = np.zeros(len(train_test))





for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_test.values, target)):

    print("fold n°{}".format(fold_))

    trn_data = lgb.Dataset(train_test.iloc[trn_idx][features], label=target[trn_idx])

    val_data = lgb.Dataset(train_test.iloc[val_idx][features], label=target[val_idx])



    num_round = 30000

    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 1400)

    oof[val_idx] = clf.predict(train_test.iloc[val_idx][features], num_iteration=clf.best_iteration)
feature_imp = pd.DataFrame(sorted(zip(clf.feature_importance(),features)), columns=['Value','Feature'])

plt.figure(figsize=(30, 30))

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).head(20))

plt.title('GDBT Features')

plt.tight_layout()

plt.show()



feature_imp.sort_values(by="Value", ascending=False).to_csv('feat.csv')