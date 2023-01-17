# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install missingpy
!pip install git+https://github.com/t0biK/data_science_utils.git
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

from tqdm import tqdm_notebook

from IPython.core.display import HTML

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold, KFold, TimeSeriesSplit, train_test_split

from sklearn.metrics import roc_auc_score

from sklearn.tree import DecisionTreeClassifier

from sklearn import tree

import graphviz

import lightgbm as lgb

import warnings

from tqdm.notebook import tqdm

import data_science_utils.feature_extraction_util as fe

from data_science_utils.plot_util import *

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import GridSearchCV

from missingpy import MissForest
test_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv')

test_identity.columns = test_identity.columns.str.replace("-", "_")

train_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')

test_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')

train_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
train = pd.merge(train_identity, train_transaction, on='TransactionID', how='right')

test = pd.merge(test_identity, test_transaction, on='TransactionID', how='right')
test_identity, train_identity, test_transaction, train_transaction = [None]*4
numerical_columns = ['id_0'+str(i) for i in range(1,10)]+['id_10', 'id_11']
test.loc[:,~test.columns.isin(numerical_columns)] = test.loc[:,~test.columns.isin(numerical_columns)].fillna(-999)

train.loc[:,~train.columns.isin(numerical_columns)] = train.loc[:,~train.columns.isin(numerical_columns)].fillna(-999)
imputer = MissForest()

test.loc[:,test.columns.isin(numerical_columns)] = imputer.fit_transform(test.loc[:,test.columns.isin(numerical_columns)])

train.loc[:,train.columns.isin(numerical_columns)] = imputer.fit_transform(train.loc[:,train.columns.isin(numerical_columns)])
params = {'objective': 'binary', "boosting_type": "gbdt", "subsample": 1, "bagging_seed": 11, "metric": 'auc', 'num_boost_round':100, 'verbose':-1}

#train, test = fe.correct_features_with_covariate_shift(train, test, params, train.columns.drop(['isFraud', 'TransactionID', 'TransactionDT']))
useful_features = ['TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'dist1',

                   'P_emaildomain', 'R_emaildomain', 'C1', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13',

                   'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'M2', 'M3',

                   'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V17',

                   'V19', 'V20', 'V29', 'V30', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V40', 'V44', 'V45', 'V46', 'V47', 'V48',

                   'V49', 'V51', 'V52', 'V53', 'V54', 'V56', 'V58', 'V59', 'V60', 'V61', 'V62', 'V63', 'V64', 'V69', 'V70', 'V71',

                   'V72', 'V73', 'V74', 'V75', 'V76', 'V78', 'V80', 'V81', 'V82', 'V83', 'V84', 'V85', 'V87', 'V90', 'V91', 'V92',

                   'V93', 'V94', 'V95', 'V96', 'V97', 'V99', 'V100', 'V126', 'V127', 'V128', 'V130', 'V131', 'V138', 'V139', 'V140',

                   'V143', 'V145', 'V146', 'V147', 'V149', 'V150', 'V151', 'V152', 'V154', 'V156', 'V158', 'V159', 'V160', 'V161',

                   'V162', 'V163', 'V164', 'V165', 'V166', 'V167', 'V169', 'V170', 'V171', 'V172', 'V173', 'V175', 'V176', 'V177',

                   'V178', 'V180', 'V182', 'V184', 'V187', 'V188', 'V189', 'V195', 'V197', 'V200', 'V201', 'V202', 'V203', 'V204',

                   'V205', 'V206', 'V207', 'V208', 'V209', 'V210', 'V212', 'V213', 'V214', 'V215', 'V216', 'V217', 'V219', 'V220',

                   'V221', 'V222', 'V223', 'V224', 'V225', 'V226', 'V227', 'V228', 'V229', 'V231', 'V233', 'V234', 'V238', 'V239',

                   'V242', 'V243', 'V244', 'V245', 'V246', 'V247', 'V249', 'V251', 'V253', 'V256', 'V257', 'V258', 'V259', 'V261',

                   'V262', 'V263', 'V264', 'V265', 'V266', 'V267', 'V268', 'V270', 'V271', 'V272', 'V273', 'V274', 'V275', 'V276',

                   'V277', 'V278', 'V279', 'V280', 'V282', 'V283', 'V285', 'V287', 'V288', 'V289', 'V291', 'V292', 'V294', 'V303',

                   'V304', 'V306', 'V307', 'V308', 'V310', 'V312', 'V313', 'V314', 'V315', 'V317', 'V322', 'V323', 'V324', 'V326',

                   'V329', 'V331', 'V332', 'V333', 'V335', 'V336', 'V338', 'id_01', 'id_02', 'id_03', 'id_05', 'id_06', 'id_09',

                   'id_11', 'id_12', 'id_13', 'id_14', 'id_15', 'id_17', 'id_19', 'id_20', 'id_30', 'id_31', 'id_32', 'id_33',

                   'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo']
train['TransactionAmt_decimal'] = ((train['TransactionAmt'] - train['TransactionAmt'].astype(int)) * 1000).astype(int)

test['TransactionAmt_decimal'] = ((test['TransactionAmt'] - test['TransactionAmt'].astype(int)) * 1000).astype(int)



# Count encoding for card1 feature. 

# Explained in this kernel: https://www.kaggle.com/nroman/eda-for-cis-fraud-detection

train['card1_count_full'] = train['card1'].map(pd.concat([train['card1'], test['card1']], ignore_index=True).value_counts(dropna=False))

test['card1_count_full'] = test['card1'].map(pd.concat([train['card1'], test['card1']], ignore_index=True).value_counts(dropna=False))



# https://www.kaggle.com/fchmiel/day-and-time-powerful-predictive-feature

train['Transaction_day_of_week'] = np.floor((train['TransactionDT'] / (3600 * 24) - 1) % 7)

test['Transaction_day_of_week'] = np.floor((test['TransactionDT'] / (3600 * 24) - 1) % 7)

train['Transaction_hour'] = np.floor(train['TransactionDT'] / 3600) % 24

test['Transaction_hour'] = np.floor(test['TransactionDT'] / 3600) % 24



# Some arbitrary features interaction

for feature in ['id_02__id_20', 'id_02__D8', 'D11__DeviceInfo', 'DeviceInfo__P_emaildomain', 'P_emaildomain__C2', 

                'card2__dist1', 'card1__card5', 'card2__id_20', 'card5__P_emaildomain', 'addr1__card1']:



    f1, f2 = feature.split('__')

    train[feature] = train[f1].astype(str) + '_' + train[f2].astype(str)

    test[feature] = test[f1].astype(str) + '_' + test[f2].astype(str)



    le = LabelEncoder()

    le.fit(list(train[feature].astype(str).values) + list(test[feature].astype(str).values))

    train[feature] = le.transform(list(train[feature].astype(str).values))

    test[feature] = le.transform(list(test[feature].astype(str).values))

    

for feature in ['id_34', 'id_36']:

    if feature in useful_features:

        # Count encoded for both train and test

        train[feature + '_count_full'] = train[feature].map(pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))

        test[feature + '_count_full'] = test[feature].map(pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))

        

for feature in ['id_01', 'id_31', 'id_33', 'id_35', 'id_36']:

    if feature in useful_features:

        # Count encoded separately for train and test

        train[feature + '_count_dist'] = train[feature].map(train[feature].value_counts(dropna=False))

        test[feature + '_count_dist'] = test[feature].map(test[feature].value_counts(dropna=False))
params = {'num_leaves': 491,

          'min_child_weight': 0.03454472573214212,

          'feature_fraction': 0.3797454081646243,

          'bagging_fraction': 0.4181193142567742,

          'min_data_in_leaf': 106,

          'objective': 'binary',

          'max_depth': -1,

          'learning_rate': 0.0066,

          'n_estimators': 1000,

          "boosting_type": "gbdt",

          "bagging_seed": 11,

          "metric": 'auc',

          "verbosity": -1,

          'reg_alpha': 0.3899927210061127,

          'reg_lambda': 0.6485237330340494,

          'random_state': 47

         }
X_train, X_test, y_train, y_test = train_test_split(train.drop(columns=['isFraud', 'TransactionID', 'TransactionDT']), train['isFraud'], test_size=0.33)
for col in X_train.select_dtypes(include='object').columns:

    le = LabelEncoder()

    le.fit(list(X_train[col].astype(str).values) + list(X_test[col].astype(str).values) + list(test[col].astype(str).values))

    X_train[col] = le.transform(list(X_train[col].astype(str).values))

    X_test[col] = le.transform(list(X_test[col].astype(str).values))

    test[col] = le.transform(list(test[col].astype(str).values))
clf = lgb.LGBMClassifier(**params)

clf.fit(X_train, y_train)
cols_to_drop = ['TransactionID', 'TransactionDT']
sub = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv')

sub['isFraud'] = clf.predict_proba(test.drop(columns=cols_to_drop))[:, 1]

sub['TransactionID'] = test.TransactionID

sub.to_csv('submission.csv', index=False)