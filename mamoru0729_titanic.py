%matplotlib inline

import numpy as np

import pandas as pd
# 学習用データ

train = pd.read_csv("../input/titanic/train.csv")

# テスト用データ

test = pd.read_csv("../input/titanic/test.csv")

#提出用サンプル

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")
data = pd.concat([train, test], sort=True)

print(len(train))

print(len(test))

print(len(data))
data.head()
data.columns
#欠損地の有無を確認

data.isnull()
# 欠損地の数を確認

data.isnull().sum()
# 特微量エンジニアリング



# Pclass チケットクラス

    

# Sex

data['Sex'].replace(['male','female'],[0, 1], inplace=True)



# Embark 出航港-タイタニックへ乗った港-

data['Embarked'].fillna(('S'), inplace=True)

data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



# Fere 料金

data['Fare'].fillna(np.mean(data['Fare']), inplace=True)



# Age

data['Age'].fillna(data['Age'].median(), inplace=True)
# 余分な列をカット

delete_columns = ['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin']

data.drop(delete_columns, axis = 1, inplace = True)
# データ処理のために結合したTrainデータとTestデータを分離

train = data[:len(train)]

test = data[len(train):]
# データ塊を作成

y_train = train['Survived']

X_train = train.drop('Survived', axis = 1)



X_test = test.drop('Survived', axis = 1)
# データを見てみる

print(X_train.head())

print(y_train.head())

print(X_test.head())
# 予測モデルの作成

from sklearn.linear_model import LogisticRegression



clf = LogisticRegression(penalty='l2', solver="sag", random_state=0)

clf.fit(X_train, y_train)
# Test用データを用いて予測

y_pred = clf.predict(X_test)

y_pred[:20]
# 提出用CSVを作成
# データフレームを作成

sub = pd.DataFrame(pd.read_csv("../input/titanic/test.csv")['PassengerId'])

sub.head()
# 予測結果を追加

sub['Survived'] = list(map(int, y_pred))

sub.head()
#CSVファイルを作成

sub.to_csv("submission.csv", index = False)