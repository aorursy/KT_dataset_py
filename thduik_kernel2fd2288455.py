# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')

test = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')
sample = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')
train.head()
train.columns
def plot_new_feature_distribution(df1, df2, label1, label2, features):

    i = 0

    sns.set_style('whitegrid')

    plt.figure()

    fig, ax = plt.subplots(2,4,figsize=(18,10))



    for feature in features:

        i += 1

        plt.subplot(2,4,i)

        sns.kdeplot(df1[feature], bw=0.5,label=label1)

        sns.kdeplot(df2[feature], bw=0.5,label=label2)

        plt.xlabel(feature, fontsize=11)

        locs, labels = plt.xticks()

        plt.tick_params(axis='x', which='major', labelsize=8)

        plt.tick_params(axis='y', which='major', labelsize=8)

    plt.show();

features = train.columns

df_0 = train[train.target == 0]

df_1 = train[train.target == 1]

for i_lol in range(2,197,8):

    plot_new_feature_distribution(df_0, df_1, 'target_0', 'target_1', features[i_lol:i_lol+8])

    print (features[i_lol])
dty = train.dtypes
dty[dty == object]
import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold
train.head()
test.head(1)
y = train['target']

X = train.iloc[:, 2:]



X_test = test.iloc[:, 1:]

test_id_arr = test['ID_code']
X.head()
#lr, min_child, bagging_fraction

params = {'objective': 'binary',

          'bagging_fraction':0.6,

          'bagging_freq':7,

          'num_leaves': 128,

          'min_child_samples': 30,       

          'max_depth': 13,

          'learning_rate': 0.13,

          "boosting_type": "gbdt",

          "subsample_freq": 1,

          "subsample": 0.9,

          "bagging_seed": 11,

          "metric": 'mae',

          "verbosity": -1,

          'reg_alpha': 0.15,

          'reg_lambda': 0.3,

          'colsample_bytree': 1.0

         }





kfold = StratifiedKFold(n_splits = 5, random_state = 12212)

y_pred_res = np.zeros(X_test.shape[0])

for (train_index, test_index) in kfold.split(X, y):

    train_X, train_y = X.loc[train_index], y.loc[train_index]

    valid_X, valid_y = X.loc[test_index], y.loc[test_index]

    

    model = lgb.LGBMClassifier(**params, n_estimators = 3000, n_jobs = -1)

    model.fit(train_X, train_y, eval_set = [(train_X, train_y), (valid_X, valid_y)], eval_metric = 'auc', early_stopping_rounds = 100,verbose = 10)

    y_pred_valid = model.predict_proba(valid_X)

    y_pred_test = model.predict_proba(X_test)

    y_pred_test = np.array([i[1] for i in y_pred_test])

    y_pred_res += np.array(y_pred_test)
res_arr = y_pred_res/5
imp_dict = model.feature_importances_

cols = train_X.columns

df_dict = {'feature':cols, 'imp':imp_dict}

fea_imp = pd.DataFrame(data = df_dict)
fea_imp = fea_imp.sort_values('imp', ascending = False)
fea_imp.head(30)
res = res_arr

res_dict = {'ID_code':test_id_arr, 'target':res}

resdf = pd.DataFrame(data = res_dict)

resdf.to_csv("submission.csv", index=False)