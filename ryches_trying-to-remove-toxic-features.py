import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import lightgbm as lgb

from sklearn.model_selection import KFold

from sklearn import model_selection, preprocessing, metrics

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import shap

import os

print(os.listdir("../input"))

from sklearn import preprocessing

import xgboost as xgb

import gc





import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/standalone-train-and-test-preprocessing/train.csv')

test = pd.read_csv('../input/standalone-train-and-test-preprocessing/test.csv')
adversarial_features = ["TransactionDT", "id_31", "D15", "id_13", "D11", "D4", "D10", "TransactionAmt", "V75", 

                        "dist1", "addr1", "D13", "D2", "id_01", "C13", "C9", "C11", "C12", "D1", "C14", "D2", "id_02", "C6", "id_30",

                        "D5", "D6", "D14", "C2", "id_20","DeviceInfo"]

train = train.drop(adversarial_features, axis = 1)

test = test.drop(adversarial_features, axis = 1)
train.shape

test.shape
features = test.columns
train = train[features]
train['target'] = 0

test['target'] = 1
# for col in test:

#     if (test[col].dtype == "object") or ("id" in col):

#         print(col, train[col].nunique())

#         diffs = list(set(train[col].unique()).difference(set(test[col].unique())))

#         train.loc[train[col].isin(diffs) , col] = train[col].unique()[0]

#         test.loc[test[col].isin(diffs) , col] = train[col].unique()[0]
train_test = pd.concat([train, test], axis =0)

target = train_test['target'].values
object_columns = np.load('../input/standalone-train-and-test-preprocessing/object_columns.npy')
del train, test
gc.collect()
# Label Encoding

for f in train_test.columns:

    if train_test[f].dtype=='object' or train_test[f].dtype=='object': 

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(train_test[f].values) + list(train_test[f].values))

        train_test[f] = lbl.transform(list(train_test[f].values))

train, test = model_selection.train_test_split(train_test, test_size=0.33, random_state=42, shuffle=True)
del train_test

gc.collect()
train_y = train['target'].values

test_y = test['target'].values

del train['target'], test['target']

gc.collect()
train = lgb.Dataset(train, label=train_y)

test = lgb.Dataset(test, label=test_y)

param = {'num_leaves': 50,

         'min_data_in_leaf': 30, 

         'objective':'binary',

         'max_depth': 5,

         'learning_rate': 0.2,

         "min_child_samples": 20,

         "boosting": "gbdt",

         "feature_fraction": 0.75,

         "bagging_freq": 1,

         "bagging_fraction": 0.75 ,

         "bagging_seed": 44,

         "metric": 'auc',

         "verbosity": -1}
num_round = 100

clf = lgb.train(param, train, num_round, valid_sets = [train, test], verbose_eval=50, early_stopping_rounds = 50)
feature_imp = pd.DataFrame(sorted(zip(clf.feature_importance(),features)), columns=['Value','Feature'])



plt.figure(figsize=(20, 10))

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).head(20))

plt.title('LightGBM Features')

plt.tight_layout()

plt.show()

plt.savefig('lgbm_importances-01.png')