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
train = pd.read_csv('../input/train.csv')
#　データの中身を確認

train.head(5)
train.describe()
##　男女の情報を０，１に置き換える

train['fin_Sex'] = train['Sex'].map(lambda x: 1 if x=='female'else 0)
train.head(5)
# 説明変数と目的変数を選ぶ

## ここでは数値になっている列だけを使う

X = train[['Pclass','Age','SibSp','Parch','Fare','fin_Sex']]

y = train['Survived']



#訓練データとテストデータに分ける

##　ここでは半分ずつに分ける

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.5, random_state=0)
# LightGBMを使って学習

# https://lightgbm.readthedocs.io/en/latest/Python-Intro.html

import lightgbm

train_data = lightgbm.Dataset(X_train, y_train)

eval_data = lightgbm.Dataset(X_test, y_test, reference=train_data)

params = {'ovjective': 'binary', 'metric': 'auc'}

model = lightgbm.train(params, train_data, valid_sets=eval_data)

model.save_model('model.txt')
#　予測する

y_pred = model.predict(X_test, numiteration=model.best_iteration)

binary_y_pred = np.where(y_pred > 0.5, 1, 0)
# 制度を計算する

accuracy = sum(y_test == binary_y_pred) / len(y_test)

print(accuracy)
# テストデータをトレインデータと同じ形に

test = pd.read_csv('../input/test.csv')

test['fin_Sex'] = test['Sex'].map(lambda x: 1 if x=='female'else 0)

test['fin_Embarked'] = test['Embarked'].map(lambda x: 1 if x=='Q'else 2 if x=='S' else 0)

test_id = test[['PassengerId']]

test = test[['Pclass','Age','SibSp','Parch','Fare','fin_Sex']]

submission_pred = model.predict(test, numiteration=model.best_iteration)

submission_pred = np.where(submission_pred > 0.5, 1, 0)
# 提出データの形式を確認

submission = pd.read_csv('../input/gender_submission.csv')

print(submission.head(5), '\n', submission.shape)
# 提出形式に整える

submission = test_id

submission['Survived'] = submission_pred

print(submission.head(5), '\n', submission.shape)
# 提出ファイルを作る

from datetime import datetime

from pytz import timezone

date_now = datetime.now(timezone('Asia/Tokyo')).strftime("%Y%m%d%H%M")

submission.to_csv(date_now + 'submission_file.csv', index =False)