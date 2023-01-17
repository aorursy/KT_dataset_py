# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train = pd.read_csv('../input/artificial-data-leaks/train.csv')

print('Train data shape:',df_train.shape)

df_train.head(10)
df_test = pd.read_csv('../input/artificial-data-leaks/test.csv')

print('Test data shape:',df_test.shape)

df_test.head(10)
print('Distributions')

fig, axs = plt.subplots(ncols=4, nrows=3, figsize=(15,12))

for x in range(10):

    sns.distplot(df_train['col{}'.format(x)], hist=True, kde=True, ax=axs[x//4,x%4])

sns.distplot(df_train['target'], hist=True, kde=True, ax=axs[2,2])

pass
print('Correlations all close to zero')

f = plt.figure(figsize=(11, 11))

plt.matshow(df_train.corr(), fignum=f.number)

plt.xticks(range(df_train.shape[1]), df_train.columns, fontsize=14, rotation=45)

plt.yticks(range(df_train.shape[1]), df_train.columns, fontsize=14)

cb = plt.colorbar()

plt.show()

df_train.corr()
import lightgbm as lgb

from sklearn import metrics



RS = 0

ROUNDS = 500

TARGET = 'target'



params = {

    'objective': 'binary',

    'metric': 'auc',

    'boosting': 'gbdt',

    'learning_rate': 0.1,

    'verbose': 0,

    'num_leaves': 64,

    'bagging_fraction': 0.8,

    'bagging_seed': RS,

    'feature_fraction': 0.9,

    'feature_fraction_seed': RS,

    'max_bin': 100,

    'max_depth': 5

}



x_train = lgb.Dataset(df_train.drop(TARGET, axis=1), df_train[TARGET])

model = lgb.train(params, x_train, num_boost_round=ROUNDS)

preds = model.predict(df_test.drop(TARGET, axis=1))



score = metrics.roc_auc_score(df_test[TARGET], preds)

print('Test AUC score:',score)



fig, axs = plt.subplots(ncols=2, figsize=(15,6))

lgb.plot_importance(model, importance_type='split', ax=axs[0], title='Feature importance (split)')

lgb.plot_importance(model, importance_type='gain', ax=axs[1], title='Feature importance (gain)')

pass



#Baseline not-tuned model, raw features: AUC 0.74632

#All leaks found, same model parameters: AUC 0.93927
