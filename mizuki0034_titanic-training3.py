# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# 前回のAgeを処理する部分までを実行



import numpy as np

import pandas as pd



train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")



data = pd.concat([train, test], sort=False)



data['Sex'].replace(['male','female'], [0, 1], inplace=True)

data['Embarked'].fillna(('S'), inplace=True)

data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

data['Fare'].fillna(np.mean(data['Fare']), inplace=True)
#中央値でデータを補完

data['Age'].fillna(data['Age'].median(), inplace=True)
# その他の特徴量エンジニアリングの部分の処理



delete_columns = ['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin']

data.drop(delete_columns, axis=1, inplace=True)



train = data[:len(train)]

test = data[len(train):]



y_train = train['Survived']

X_train = train.drop('Survived', axis=1)

X_test = test.drop('Survived', axis=1)
X_train.head()

y_train.head()
#ロジスティック回帰による学習

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(penalty='l2', solver="sag", random_state=0)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
#以下仮説検証　家族の人数が多いと生存率低い?

# 改めてデータを読み込み直す

train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")



data = pd.concat([train, test], sort=False)



data['Sex'].replace(['male','female'], [0, 1], inplace=True)

data['Embarked'].fillna(('S'), inplace=True)

data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

data['Fare'].fillna(np.mean(data['Fare']), inplace=True)

data['Age'].fillna(data['Age'].median(), inplace=True)
data['FamilySize'] = data['Parch'] + data['SibSp'] + 1

train['FamilySize'] = data['FamilySize'][:len(train)]

test['FamilySize'] = data['FamilySize'][len(train):]



import seaborn as sns

sns.countplot(x='FamilySize', data = train, hue='Survived')
data['IsAlone'] = 0

data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1



train['IsAlone'] = data['IsAlone'][:len(train)]

test['IsAlone'] = data['IsAlone'][len(train):]
# その他の特徴量エンジニアリングの部分の処理

delete_columns = ['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin']

data.drop(delete_columns, axis=1, inplace=True)



train = data[:len(train)]

test = data[len(train):]



y_train = train['Survived']

X_train = train.drop('Survived', axis=1)

X_test = test.drop('Survived', axis=1)
X_train.head()
sub = gender_submission



clf.fit(X_train, y_train)

y_pred_familysize_isalone = clf.predict(X_test)



sub['Survived'] = list(map(int, y_pred_familysize_isalone))

sub.to_csv("submission_familysize_isalone.csv", index = False)



sub.head()