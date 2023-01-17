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



%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/standalone-train-and-test-preprocessing/train.csv')

test = pd.read_csv('../input/standalone-train-and-test-preprocessing/test.csv')
train.shape
test.shape
print(train.TransactionDT.min(), train.TransactionDT.max())

print(test.TransactionDT.min(), test.TransactionDT.max())
features = test.drop('TransactionDT', axis=1).columns
train = train[features]

test = test[features]
train['target'] = 0

test['target'] = 1
train_test = pd.concat([train, test], axis =0)



target = train_test['target'].values
object_columns = np.load('../input/standalone-train-and-test-preprocessing/object_columns.npy')
del train, test

gc.collect()
# Label Encoding

for f in object_columns:

    lbl = preprocessing.LabelEncoder()

    lbl.fit(list(train_test[f].values) )

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

         "feature_fraction": 0.9,

         "bagging_freq": 1,

         "bagging_fraction": 0.9 ,

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
del train, test

gc.collect()
train = pd.read_csv('../input/standalone-train-and-test-preprocessing/train.csv')

test = pd.read_csv('../input/standalone-train-and-test-preprocessing/test.csv')
fig, ax = plt.subplots(1, 2, figsize=(16,10))

train_id_31 = train.id_31.value_counts().iloc[:30]

test_id_31 = test.id_31.value_counts().iloc[:30]

sns.barplot(y=train_id_31.index, x=train_id_31.values, ax=ax[0])

sns.barplot(y=test_id_31.index, x=test_id_31.values, ax=ax[1])

plt.tight_layout()

plt.show()
fig, ax = plt.subplots(1, 2, figsize=(16,10))

train_d10 = train.D10.value_counts().iloc[1:30]

test_d10 = test.D10.value_counts().iloc[1:30]

sns.barplot(x=train_d10.index, y=train_d10.values, ax=ax[0])

sns.barplot(x=test_d10.index, y=test_d10.values, ax=ax[1])

plt.tight_layout()

plt.show()
fig, ax = plt.subplots(1, 2, figsize=(16,10))

train_d15 = train.D15.value_counts().iloc[1:30]

test_d15 = test.D15.value_counts().iloc[1:30]

sns.barplot(x=train_d15.index, y=train_d15.values, ax=ax[0])

sns.barplot(x=test_d15.index, y=test_d15.values, ax=ax[1])

plt.tight_layout()

plt.show()
fig, ax = plt.subplots(1, 2, figsize=(16,10))

sns.distplot(train.id_13.fillna(-9).values, ax=ax[0])

sns.distplot(test.id_13.fillna(-9).values, ax=ax[1])

plt.tight_layout()

plt.show()