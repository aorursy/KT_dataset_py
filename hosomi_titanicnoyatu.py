
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
import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
import seaborn as sns
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# Cabinがnullかどうかのフラグ
train['CabinIsNan'] = [False if val == val else True for val in train['Cabin'].tolist()]
test['CabinIsNan'] = [False if val == val else True for val in test['Cabin'].tolist()]

# Ageがnullかどうかのフラグ
#train['AgeIsNan'] = [False if val == val else True for val in train['Age'].tolist()]
#test['AgeIsNan'] = [False if val == val else True for val in test['Age'].tolist()]

# 家族構成
train['NumFamily'] = train['SibSp']+train['Parch']+1
test['NumFamily'] = test['SibSp']+test['Parch']+1

# ボッチ
train['IsAlone'] = [x == 1 for x in train.NumFamily]
test['IsAlone'] = [x == 1 for x in test.NumFamily]
# 名前の処理
train['Title'] = train.Name.str.extract(' ([A-Za-z]+).', expand=False) 
train['Title'] = train['Title'].replace('Mlle', 'Miss')
train['Title'] = train['Title'].replace('Ms', 'Miss')
train['Title'] = train['Title'].replace('Mme', 'Mrs')
train['Title'] = [val if val in ['Mr', 'Mrs', 'Miss', 'Ms', 'Master'] else 'Others' for val in train['Title']]

test['Title'] = test.Name.str.extract(' ([A-Za-z]+).', expand=False) 
test['Title'] = test['Title'].replace('Mlle', 'Miss')
test['Title'] = test['Title'].replace('Ms', 'Miss')
test['Title'] = test['Title'].replace('Mme', 'Mrs')
test['Title'] = [val if val in ['Mr', 'Mrs', 'Miss', 'Ms', 'Master'] else 'Others' for val in test['Title']]
# Cabinは先頭のA~Gの文字列で分類
#train['CabinType'] = [x[0] if x == x else 'Others' for x in train.Cabin]
#test['CabinType'] = [x[0] if x == x else 'Others' for x in test.Cabin]

# 2つ以上のもいるので個数をカウントしておく
# train['NumCabin'] = [len(x.split(' ')) if x == x else 0 for x in train.Cabin]
# test['NumCabin'] = [len(x.split(' ')) if x == x else 0 for x in test.Cabin]
# 防水壁のことを考慮し、CabinTypeがC以上、D、E、F以下、その他を分類
def classify_cabin(x):
    if x in ['A', 'B', 'C']:
        return 'first'
    elif x == 'D':
        return 'second'
    elif x == 'E':
        return 'third'
    elif x == 'Others':
        return 'Unknown'
    else:
        return 'fourth'
train['Deck'] = [classify_cabin(x) for x in train.CabinType]
test['Deck'] = [classify_cabin(x) for x in test.CabinType]
# Ticketは先頭の文字列で分類
def get_first_string(x):
    if x == x:
        arr = x.split(' ')
        if len(arr) > 1:
            s = arr[0]
            if s in ['WE/P', 'W.E.P.']:
                return 'WEP'
            elif 'PC' in s:
                return 'PC'
            elif 'STON' in s:
                return 'STON'
            elif 'SOTON' in s:
                return 'SOTON'
            elif 'SC' in s or 'S.C.' in s:
                return 'SC'
            elif 'SW' in s:
                return 'SW'
            elif 'W' in s and 'C' in s:
                return 'WC'
            elif 'S.P.' in s:
                return 'SP'
            elif 'S.O.' in s:
                return 'SO'
            elif 'PP' in s:
                return 'PP'
            elif 'PC' in s:
                return 'PC'
            elif 'F.C.' in s:
                return 'FC'
            elif 'Fa' in s:
                return 'FA'
            elif 'CA' in s or 'C.A' in s:
                return 'CA'
            return s[0]
        return 'general'
    return 'NaN'

train.Ticket = [get_first_string(x) for x in train.Ticket]
test.Ticket = [get_first_string(x) for x in test.Ticket]
# PClassは本来グレードを表す属性=オブジェクト
train.Pclass = ['c_' + str(x) for x in train.Pclass]
test.Pclass = ['c_' + str(x) for x in test.Pclass]

# Sexを2値化
train.replace({'male':0, 'female':1}, inplace=True)
test.replace({'male':0, 'female':1}, inplace=True)
# testのFareの欠損値を同じPclassの人の平均で埋める
fare_mean = pd.concat([train[train.Pclass=='c_3'].Fare, test[(~test.Fare.isnull())&(test.Pclass=='c_3')].Fare]).mean()
test.loc[152, 'Fare'] = fare_mean
# 年齢の欠損地は敬称（Title）ごとの平均値で埋める
age_mean = train.groupby(['Title']).Age.mean()
for k, v in age_mean.items():
    v = float(round(v))
    train.loc[(train.Title.values == k) & (train.Age.isnull()), 'Age'] = v
    test.loc[(test.Title.values == k) & (test.Age.isnull()), 'Age'] = v
# trainの乗船場所の欠損はPclass=1、Fare=80で一番それっぽいのがCなのでCで埋める
train.loc[train.Embarked.isnull(), 'Embarked'] = train[train.Pclass==1].Embarked.mode()
# 外れ値の除去
train = train.drop(493, axis=0)

# Name, Cabinを除去
train = train.drop(['Name', 'Cabin', 'Ticket'], axis=1)
test = test.drop(['Name', 'Cabin', 'Ticket'], axis=1)
target = train.Survived
train = train.drop('Survived', axis=1)
# concatしてからone-hot-encoding
df = pd.concat([train, test])
df['FareBin'] = pd.cut(df.Fare, 5, labels=False)
df['AgeBin'] = pd.cut(df.Age, 5, labels=False)
df = pd.get_dummies(df)
# 分離
train = df.iloc[:len(train), :].copy()
test = df.iloc[len(train):, :].copy()
#train['Survived'] = target
#df_corr = train.corr()
#print(df_corr['Survived'])
#sns.heatmap(df_corr, vmax=1, vmin=-1, center=0)
# IDを控えておく
test_id = test.PassengerId
# IDを除去
train = train.drop('PassengerId', axis=1)
test = test.drop('PassengerId', axis=1)
# Fare, Age, SibSp, Parch, NumFamily, NumCabinの標準化
for col in ['Fare', 'Age', 'SibSp', 'Parch', 'NumFamily']:#, 'NumCabin']:
    mean = train[col].mean()
    std = train[col].std()
    train[col] = (train[col] - mean) / std
    test[col] = (test[col] - mean) / std
# 結果保管用
results = pd.DataFrame(index=test.index)
# ロジスティック回帰
logi = LogisticRegression(C=0.01)
logi.fit(train, target)
results['Survived'] = logi.predict(test)
results['PassengerId'] = test_id
results[['PassengerId', 'Survived']].to_csv('submission.csv', index=False)
