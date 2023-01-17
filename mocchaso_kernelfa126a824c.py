# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.svm import SVC

from sklearn.metrics import accuracy_score



import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
sns.set_style('whitegrid')

%matplotlib inline
# 学習データ、テストデータの読み込み

df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

df_test = pd.read_csv('/kaggle/input/titanic/test.csv')
# 性別で可視化

sns.countplot('Sex',data=df_train)
# 乗客の性別をチケットクラスで層別化

sns.countplot('Sex',data=df_train,hue='Pclass')
# 乗客のチケットクラスを性別で層別

sns.countplot('Pclass',data=df_train,hue='Sex')
# 名前、客室番号、チケット番号のカラムは除外

df_train = df_train.drop('Name', axis=1)

df_train = df_train.drop('Ticket', axis=1)

df_train = df_train.drop('Cabin', axis=1)

df_test = df_test.drop('Name', axis=1)

df_test = df_test.drop('Ticket', axis=1)

df_test = df_test.drop('Cabin', axis=1)

df_train.head(5)
# 性別、乗船港のカラムを数値化

X_train = df_train.replace({'Sex': {'male': 0, 'female': 1}})

X_train = X_train.replace({'Embarked': {'S': 1, 'C': 2, 'Q': 3}})

df_test = df_test.replace({'Sex': {'male': 0, 'female': 1}})

df_test = df_test.replace({'Embarked': {'S': 1, 'C': 2, 'Q': 3}})

# 欠損値の置換

# 除去すると行数が足りなくなり、データの不整合が発生する

X_train = X_train.fillna(0)

df_test = df_test.fillna(0)

X_train.head(5)
# 学習データについて、特徴量と正解ラベルを分離

Y_train = X_train['Survived']

X_train = X_train.drop('Survived', axis=1)

X_train.head(5)
# 線形SVMで学習

model = SVC(kernel='linear', random_state=None)

model.fit(X_train, Y_train)
# 予測モデルの評価

# 学習データに対する精度

pred_train = model.predict(X_train)

accuracy_train = accuracy_score(Y_train, pred_train)

print('学習データに対する精度 = {}'.format(accuracy_train))
# テストデータに対しても予測

pred_test = model.predict(df_test)
# 予測結果の提出用にCSV出力

submission = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': pred_test})

submission.to_csv('/kaggle/working/submission.csv', index=False)