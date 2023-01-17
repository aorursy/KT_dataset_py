# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # data Visualization

import seaborn as sns # data Visualization



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Importing Datasets

df_train = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv')

df_test = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv')

df_sub = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/sample_submission.csv')
sns.countplot(x=df_train['target'], data=df_train, palette='seismic')

plt.title("TARGET DISTRIBUTION", fontsize = 20)

plt.xlabel("Target Values", fontsize = 15)

plt.ylabel("Count", fontsize = 15)

plt.show()
df_train.sort_index(inplace=True)
df_train.head()
y_train = df_train['target']

test_id = df_test['id']

df_train.drop(['target', 'id'], axis=1, inplace=True)

df_test.drop('id', axis=1, inplace=True)
cat_feat_to_encode = df_train.columns.tolist()

smoothing=0.20

import category_encoders as ce

oof = pd.DataFrame([])

from sklearn.model_selection import StratifiedKFold

for tr_idx, oof_idx in StratifiedKFold(n_splits=5, random_state= 1024, shuffle=True).split(df_train, y_train):

    ce_target_encoder = ce.TargetEncoder(cols = cat_feat_to_encode, smoothing=smoothing)

    ce_target_encoder.fit(df_train.iloc[tr_idx, :], y_train.iloc[tr_idx])

    oof = oof.append(ce_target_encoder.transform(df_train.iloc[oof_idx, :]), ignore_index=False)

ce_target_encoder = ce.TargetEncoder(cols = cat_feat_to_encode, smoothing=smoothing)

ce_target_encoder.fit(df_train, y_train)

df_train = oof.sort_index()

df_test = ce_target_encoder.transform(df_test)
x_train = df_train.iloc[:,:].values

x_test = df_test.iloc[:,:].values
import lightgbm as lgb

classifier = lgb.LGBMClassifier(**{

                    'learning_rate': 0.05,

                    'feature_fraction': 0.1,

                    'min_data_in_leaf' : 12,

                    'max_depth': 3,

                    'reg_alpha': 1,

                    'reg_lambda': 1,

                    'objective': 'binary',

                    'metric': 'auc',

                    'n_jobs': -1,

                    'n_estimators' : 5000,

                    'feature_fraction_seed': 42,

                    'bagging_seed': 42,

                    'boosting_type': 'gbdt',

                    'verbose': 1,

                    'is_unbalance': True,

                    'boost_from_average': False})
classifier.fit(x_train, y_train)
y_pred = classifier.predict_proba(x_test)[:,1]
#Sumbmission the result

df_sub = pd.DataFrame()

df_sub['id'] = test_id

df_sub['target'] = y_pred

df_sub.to_csv('submission.csv', index=False)
df_sub.head(20)