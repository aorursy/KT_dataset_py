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
import pandas as pd
df_train = pd.read_csv('../input/train.csv')
# テストデータの読み込み
df_test = pd.read_csv('../input/test.csv')
df_gender_submission = pd.read_csv('../input/gender_submission.csv')
df_train.head(5)
print(df_train.shape)
print(df_test.shape)
print(df_gender_submission.shape)
print(df_train.columns)
print('-----')
print(df_test.columns)
df_train.info()
#df_train.info()により、各列の欠損値 (NaN)1の数とデータの型がわかります。
df_test.info()
df_train['Age'].mean() # 年齢の平均値を算出
df_train['Age'] = df_train['Age'].fillna(30)
df_test['Age'] = df_test['Age'].fillna(30)
df_train['Embarked'].value_counts()
df_train['Embarked']= df_train['Embarked'].fillna('S')
FareMean = df_test['Fare'].mean()
FareMean
df_test['Fare'] = df_test['Fare'].fillna(FareMean)
df_test['Fare'].isnull().sum()
genders = {'male': 0, 'female': 1} # 辞書を作成  # Sexをgendersを用いて変換
df_train['Sex'] = df_train['Sex'].map(genders)
df_test['Sex'] = df_test['Sex'].map(genders)
df_train = pd.get_dummies(df_train, columns=['Embarked'])
df_test = pd.get_dummies(df_test, columns = ['Embarked'])
df_train.drop(['Name', 'Cabin', 'Ticket'], axis=1, inplace=True)
df_test.drop(['Name', 'Cabin', 'Ticket'], axis=1, inplace=True)
print('--df_trainの欠損値--')
print(df_train.isnull().sum()) # df_trainの欠損値を表示
print('-'*10 )
print('--df_testの欠損値--')
print(df_test.isnull().sum()) # df_testの欠損値を表示
df_train.head(5)
X_train = df_train.drop(columns=['PassengerId', 'Survived']) # 不要な列を削除
X_train.head(3)
y_train = df_train['Survived'] # Y_trainは、df_trainのSurvived列
y_train[:5]
X_test = df_test.drop('PassengerId', axis=1).copy()
X_test.head(3)
from sklearn.ensemble import RandomForestClassifier # ランダムフォレストのインスタンスを作成
forest = RandomForestClassifier(random_state=1) # X_trainからY_trainを予測するように学習
forest.fit(X_train,y_train)
# 正解率を表示
acc_log = round(forest.score(X_train, y_train) * 100, 2)
print(round(acc_log,2,), '%')
y_pred = forest.predict(X_test)

submission = pd.DataFrame({
'PassengerId': df_test['PassengerId'],
'Survived': y_pred })

submission.head(4)
submission.to_csv('submission.csv', index=False)
