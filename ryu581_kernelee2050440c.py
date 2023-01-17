# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
# テストデータの読み込み

train_df = pd.read_csv('../input/titanic/train.csv')

test_df = pd.read_csv('../input/titanic/test.csv')

gender_submission = pd.read_csv('../input/titanic/gender_submission.csv')
# 独立変数(X)と従属変数(Y)を分ける

X_train = train_df.loc[:, ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']]

y_train = train_df.loc[:, 'Survived']
# 独立変数の中から、何を使うか特徴量を選ぶ(不要な変数を削除する)

X_train = X_train.drop(['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis=1)

test_df = test_df.drop(['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis=1)
# 欠損値をハンドリングする

# どの変数が欠損値を含んでいるかを確認する

X_train.isnull().sum()
# 欠損値をどのように処理するかを決める (欠損値を含む行ごと削除? / 欠損値をなにかの値で置き換え？) ここは、全体の平均値を欠損値に当てはめる

age_mean = X_train['Age'].mean()

age_mean
# 上で出したAgeの平均値を欠損値に当てはめる

X_train = X_train.fillna(age_mean)

# X_train = X_train.fillna(X_train.mean())でも多分可能

test_df = test_df.fillna(test_df.mean())
# Sex変数において、男性ならば0, 女性ならば1と置き換える 'Sex_rev'のカラムを追加する

X_train['Sex_rev'] = X_train['Sex'].map({'male': 0, 'female': 1}).astype(int)

test_df['Sex_rev'] = test_df['Sex'].map({'male': 0, 'female': 1}).astype(int)
# もともとの'Sex'カラムを削除する

X_train = X_train.drop('Sex', axis=1)

test_df = test_df.drop('Sex', axis=1)
# ParchとSurvivedにどんな関係があるのかを見る

pclass_crosstab = pd.crosstab(y_train, X_train['Pclass'])

pclass_crosstab

# Pclassが３だと生存確率が下がり、１，２だと生存確率が高い　したがって、pclassを逆数処理したものを特徴量とする
# Pclassの特徴量を逆数にし、新たにカラムを追加する

X_train['Pclass_rev'] = 1 / X_train.loc[:, 'Pclass']

test_df['Pclass_rev'] = 1 / test_df.loc[:, 'Pclass']

# もともとのPclassのカラムを削除する

X_train = X_train.drop('Pclass', axis=1)

test_df = test_df.drop('Pclass', axis=1)
# 'SibSP'とSurvivedの関係を見る

SibSp_crosstab = pd.crosstab(y_train, X_train['SibSp'])

SibSp_crosstab
Parch_ct = pd.crosstab(y_train, X_train['Parch'])

Parch_ct

test_df.isnull().sum()
X_train['Parch_rev'] = X_train['Parch'].map({0:0, 1:1, 2:1, 3:0, 4:0, 5:0, 6:0}).astype(int)

test_df['Parch_rev'] = test_df['Parch'].replace({0:0, 1:1, 2:1, 3:0, 4:0, 5:0, 6:0})
X_train = X_train.drop('Parch', axis=1)

test_df = test_df.drop('Parch', axis=1)
test_df
clf = LogisticRegression()
clf.fit(X_train, y_train)
clf.score(X_train, y_train)
y_pred = clf.predict(test_df)
y_pred
sub = gender_submission

sub['Survived'] = list(map(int, y_pred))

sub.to_csv('submission.csv', index=False)