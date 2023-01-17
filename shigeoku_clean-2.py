import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.utils import np_utils

%matplotlib inline

import check_miss_value
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
train.describe()
test.describe()
train.corr()
train.Sex.values.reshape(-1, 1).shape
train.info()
from sklearn.preprocessing import LabelEncoder
train.Sex.value_counts()
# LE1 = LabelEncoder()
# train.Sex = LE1.fit_transform(train.Sex.values.reshape(-1, 1))
train['Sex'].value_counts()
train.Sex.value_counts()
# train.Ticket.value_counts()
# Ticket は後で削除する
train.Embarked.value_counts()
# LE2 = LabelEncoder()
from sklearn.preprocessing import Imputer
# impubter = Imputer(missing_values=np.nan, strategy='most_frequent')
# 文字列型は Imputerで入れられないので、最頻値を直接入れる
train.Embarked = train.Embarked.fillna('S')
train.Embarked.value_counts()
# train.Embarked = LE2.fit_transform(train.Embarked.values.reshape(-1, 1))
train.Embarked.value_counts()
train.corr()[train.corr() > 0.2]
train.corr()[train.corr() < -0.2]
# PassengerId, Name, Ticket, Cabin　を削除
train = train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
check_miss_value.check_miss_value(train)
train.Age = train.Age.fillna(train.Age.median())
check_miss_value.check_miss_value(train)
# testデータ
check_miss_value.check_miss_value(test)
test.info()
test.Age.median()
test.Age = test.Age.fillna(test.Age.median())
test.Fare = test.Fare.fillna(test.Fare.median())
check_miss_value.check_miss_value(test)
# Name, Ticket, Cabin　を削除
test = test.drop(['Name', 'Ticket', 'Cabin'], axis=1)
# test.Sex = LE1.fit_transform(test.Sex.values.reshape(-1, 1))
# test.Embarked = LE2.fit_transform(test.Sex.values.reshape(-1, 1))
test.Sex.value_counts()
test.Embarked.value_counts()
test.info()
test.corr()
train.info()
test.info()
train.Age.min(), train.Age.max(), test.Age.min(), test.Age.max()
# 年齢を 10 最刻みのレンジにする
train['Age_bin'] = train['Age'] // 10
test['Age_bin'] = test['Age'] // 10
train.Fare.min(), train.Fare.max(), test.Fare.min(), test.Fare.max()
# 運賃を 20 $ 毎のレンジにする
train['Fare_bin'] = train['Fare'] // 25
test['Fare_bin'] = test['Fare'] // 25
# カテゴリカルを変換
train = pd.get_dummies(train, columns=['Sex', 'Embarked', 'Pclass'])
test = pd.get_dummies(test, columns=['Sex', 'Embarked', 'Pclass'])
# picle ファイルに出力
train.to_pickle('pd_train.pk2')
test.to_pickle('pd_test.pk2')
train.corr()
train.corr()[train.corr() > 0.2]
train.corr()[train.corr() < -0.2]
train.corr()[:][['Survived', 'Sex_female', 'Sex_male', 'Pclass_1', 'Pclass_2', 'Pclass_3']]
