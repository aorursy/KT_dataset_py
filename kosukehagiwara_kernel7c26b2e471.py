# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.ensemble import RandomForestClassifier 

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#各種データの取り込み

train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

gender_submission = pd.read_csv('../input/titanic/gender_submission.csv')
#どんなデータがあるのか、欠損値の有無を確認

print(train.info())

print()

print(test.info())
train.head(10)
test.head(10)
train = train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

test_PassengerId = test['PassengerId']#後で使用するためここで取得しておく

test = test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
print(test_PassengerId)
#一緒に乗船していた兄弟、配偶者の数 + 一緒に乗船していた親、子供、孫の数 + 本人

train['Family_size'] = train['SibSp'] + train['Parch'] + 1 #

test['Family_size'] = test['SibSp'] + test['Parch'] +1
print(train.isnull().sum())#トレーニングデータの欠損値をカウントする

print()

print(test.isnull().sum())#テストデータの欠損値をカウントする
train['Age']=train['Age'].fillna(train['Age'].median()) #中央値で埋める

test['Age']=test['Age'].fillna(test['Age'].median()) #中央値で埋める



test['Fare']=test['Fare'].fillna(test['Fare'].median()) #中央値で埋める



train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode().iloc[0]) #最頻値で埋める
print(train.isnull().sum())

print()

print(test.isnull().sum())
print(train.info())

print()

print(test.info())
train = pd.get_dummies(train)



test = pd.get_dummies(test)
train.head(10)
test.head(10)
train_set, test_set = train_test_split(train, test_size = 0.3, random_state = 0)



X_train = train_set.iloc[:,1:] #全ての行の２列目以降を説明変数とする

y_train = train_set.iloc[:, 0] #全ての行の１列目（Survived）を目的変数とする



X_test = test_set.iloc[:,1:] #同様

y_test = test_set.iloc[:, 0] #同様
model1 = RandomForestClassifier(n_estimators = 100, random_state=0) #ランダムフォレストでモデルの作成

model1.fit(X_train, y_train)



test_pred1 = model1.predict(X_test) #分割したテストデータの予測



accuracy_score(y_test, test_pred1) #テストデータの正解率
import xgboost as xgb #xgboostをインポート

from xgboost import XGBClassifier



model2 = XGBClassifier(random_state=0) #xgboostでモデルの作成

model2.fit(X_train, y_train)



test_pred2 = model2.predict(X_test) #分割したテストデータの予測



accuracy_score(y_test, test_pred2) #テストデータの正解率
import lightgbm as gbm #LightGBMをインポート

from lightgbm import LGBMClassifier



model3 = LGBMClassifier(random_state=0) #LightGBMでモデルの作成

model3.fit(X_train, y_train)



test_pred3 = model3.predict(X_test) #分割したテストデータの予測



accuracy_score(y_test, test_pred2) #テストデータの正解率
test_predict = model2.predict(test)

print('予測データのサイズ : {}'.format(test_predict.shape)) #予測データのサイズを確認

print()

print('予測データの中身を表示')

print(test_predict) #予測データの中身を確認
solution = pd.DataFrame(test_predict, test_PassengerId, columns = ['Survived']) #最初に取得したtest_PassengerIdを使用する



solution.to_csv('solution.csv', index_label = ['PassengerId'])
features = X_train.columns

importances =model1.feature_importances_



df = pd.DataFrame({'features':features, 'importances':importances}).sort_values('importances', ascending=False)

df.reset_index(drop=True)