import numpy as np

import pandas as pd



import os

print(os.listdir("../input"))
!ls ../input/titanic
train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")
train.head()

# PassengerId – 乗客識別ユニークID

# Survived – 生存フラグ（0=死亡、1=生存）

# Pclass – チケットクラス(1=上層クラス、２=中級クラス、３=下層クラス)

# Name – 乗客の名前

# Sex – 性別（male=男性、female＝女性）

# Age – 年齢

# SibSp – タイタニックに同乗している兄弟/配偶者の数

# parch – タイタニックに同乗している親/子供の数

# ticket – チケット番号

# fare – 料金

# cabin – 客室番号

# Embarked – 出港地（タイタニックへ乗った港） C=Cherbourg, Q=Queenstown, S=Southampton
test.head()
data = pd.concat([train, test], sort=False)

data.head()
print(len(train), len(test), len(data))
data.isnull().sum()
data.describe()
data['Sex'].replace(['male','female'], [0, 1], inplace=True)

data
data['Embarked'].fillna(('S'), inplace=True)

data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

data
data['Fare'].fillna(np.mean(data['Fare']), inplace=True)

data
age_avg = data['Age'].mean()

age_std = data['Age'].std()



data['Age'].fillna(np.random.randint(age_avg - age_std, age_avg + age_std), inplace=True)

data
delete_columns = ['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin']

data.drop(delete_columns, axis=1, inplace=True)

data
train = data[:len(train)]

test = data[len(train):]
y_train = train['Survived']

X_train = train.drop('Survived', axis = 1)

X_test = test.drop('Survived', axis = 1)
X_train.head()
y_train.head()
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(penalty='l2', solver="sag", random_state=0)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

y_pred[:20]
sub = gender_submission

sub['Survived'] = list(map(int, y_pred))

sub.to_csv("submission.csv", index=False)