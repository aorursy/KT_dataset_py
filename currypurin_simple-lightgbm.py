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
# トレーニングデータとテストデータの読み込み

train = pd.read_csv('../input/application_train.csv')

test = pd.read_csv('../input/application_test.csv')



train.shape, test.shape, 
# 列名を表示

train.columns
# 初めの5行を表示

train.head()
# 各列のデータの型を表示

train.dtypes
# targetの0と1をカウント

train['TARGET'].value_counts()
# EXTが含まれるtrainの列をリストで表示

[col for col in train.columns if 'EXT' in col ]
# yはtrainのTARGET列

y = train['TARGET']



# trainとtestの3列だけを抽出

train = train.loc[:, ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']]

test = test.loc[:, ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']]
# trainの初めの5行を表示

display(train.head())



# testの初めの5行を表示

display(test.head())
# ライブラリのimport

from sklearn.model_selection import train_test_split

import lightgbm as lgb

from sklearn import metrics
# trainを0.8:0.2に分割

X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.2, random_state=42)
# インスタンス化

gbm = lgb.LGBMClassifier(objective='binary',

                         metric='auc',

                        )



# 学習

gbm.fit(X_train, y_train,

        eval_set = [(X_test, y_test)],

        early_stopping_rounds=50,

        verbose=5)



# X_testを推論

y_pred = gbm.predict_proba(X_test, num_iteration=gbm.best_iteration_)    
y_pred
# testデータを推論

target = gbm.predict_proba(test, num_iteration=gbm.best_iteration_)[:,1]



# サンプルサブミットファイルの読み込み

sample_submit = pd.read_csv('../input/sample_submission.csv')



# サンプルサブミットファイルのTARGETをtargetにする

sample_submit['TARGET'] = target
sample_submit.head()
# sample_submitをsimple_light_gbm.csvとして書き出し

sample_submit.to_csv('simple_light_gbm.csv', index=False)