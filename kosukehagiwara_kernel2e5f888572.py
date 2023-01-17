# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style='darkgrid')

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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
#欠損値をカウント、パーセンテージを出力する関数
def count_missing_rate(df):
    count = 0
    for column in df.columns:
        total = df[column].isnull().sum()#欠損値のカウント
        percent = round(total/len(df[column])*100,2)#データ数に対する欠損値の割合
        if count == 0:
            df1 = pd.DataFrame([[total,percent]], columns=['total', 'percent'], index=[column])
            count+=1
        else:#作成したカラム毎のDataFrameを結合
            df2 = pd.DataFrame([[total, percent]], columns=['total', 'percent'], index=[column])
            df1 = pd.concat([df1, df2], axis=0)
            count+=1
    return df1
count_missing_rate(train)
count_missing_rate(test)
train = train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
test_PassengerId = test['PassengerId']#後で使用するためにここで取得しておく
test = test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
#一緒に乗船していた兄弟、配偶者の数 + 一緒に乗船していた親、子供、孫の数 + 本人
train['Family_size'] = train['SibSp'] + train['Parch'] + 1 
test['Family_size'] = test['SibSp'] + test['Parch'] +1
#新たに "Family_size"を作成したので "SibSp", "Parch"を削除する
train = train.drop(['SibSp', 'Parch'], axis=1)
test = test.drop(['SibSp', 'Parch'], axis=1)
#統計量の確認
train.describe()
plt.figure(figsize=(5,7.5))
sns.boxplot(data=train, y='Fare')#乗車料金の箱ひげ図
sns.distplot(train.Fare)#乗車料金のヒストグラム
drop_train = train.dropna(subset=['Age'])#年齢のヒストグラム
sns.distplot(drop_train.Age)
drop = train.index[train['Fare']>(31+(31-7.91)*1.5)]#"Fare"の外れ値のインデックスを取得
train=train.drop(drop)#外れ値をドロップして再度trainに入れる
train
train.describe()
train = train.dropna()#trainデータの欠損値を含む行を削除
train.describe()
#Embarkedの欠損値を最頻値で埋めていく
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode().iloc[0])
# "Age" と "Fare" を中央値で埋めていく
test['Age']=test['Age'].fillna(test['Age'].median())
test['Fare']=test['Fare'].fillna(test['Fare'].median())
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

test_pred1 = model1.predict(X_test)#分割したテストデータの予測

accuracy_score(y_test, test_pred1)#テストデータの正解率
import xgboost as xgb #xgboostをインポート

xgb_params = {'objective' : 'binary:logistic',
              'eval_metric' : 'logloss'
             }

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

model2 = xgb.train(xgb_params, dtrain)

pred2 = model2.predict(dtest)
pred2
import lightgbm as lgbm #LightGBMをインポート
from lightgbm import LGBMClassifier

lgbm_params = {'objective':'binary',
              'metric':'binary_logloss',
              'verbosity':-1}

#テストデータに入力するためのパラメータを辞書として記述する
#binaryは二項分類 binary_loglossクロスエントロピー
#学習途中の中継的な情報は表示しないため-1を指定する

dtrain = lgbm.Dataset(X_train, y_train) #LightGBMでモデルの作成

model3 = lgbm.train(lgbm_params, dtrain) #パラメータを入力してテストデータの予測

pred3 = model3.predict(X_test) #テストデータを予測して出力
pred3
score_list = []
for i in [x / 100 for x in range(1, 101)]:
    pred = ((pred2*i + pred3*(1-i))) > 0.5
    pred = pred.astype(int)
    score = accuracy_score(y_test, pred)#テストデータの正解率
    score_list.append(score)
    
i = score_list.index(max(score_list))
pred = ((pred2*i + pred3*(1-i))) > 0.5
pred = pred.astype(int)
print('正答率が最大の時の割合はxgb%d: lgbm%d'%(i, 100-i))
print(max(score_list))
pred
features = X_train.columns
importances =model1.feature_importances_

df = pd.DataFrame({'features':features, 'importances':importances}).sort_values('importances', ascending=False)
df.reset_index(drop=True)
train = train.drop(['Embarked_C', 'Embarked_S', 'Embarked_Q'], axis=1)
test = test.drop(['Embarked_C', 'Embarked_S', 'Embarked_Q'], axis=1)
train_set, test_set = train_test_split(train, test_size = 0.3, random_state = 0)

X_train = train_set.iloc[:,1:] #全ての行の２列目以降を説明変数とする
y_train = train_set.iloc[:, 0] #全ての行の１列目（Survived）を目的変数とする

X_test = test_set.iloc[:,1:] #同様
y_test = test_set.iloc[:, 0] #同様
model1 = RandomForestClassifier(n_estimators = 100, random_state=0) #ランダムフォレストでモデルの作成
model1.fit(X_train, y_train)

test_pred1 = model1.predict(X_test)#分割したテストデータの予測

accuracy_score(y_test, test_pred1)#テストデータの正解率
import xgboost as xgb #xgboostをインポート

xgb_params = {'objective' : 'binary:logistic',
              'eval_metric' : 'logloss'
             }

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

model2 = xgb.train(xgb_params, dtrain)

pred2 = model2.predict(dtest)
pred2
import lightgbm as lgbm #LightGBMをインポート
from lightgbm import LGBMClassifier

lgbm_params = {'objective':'binary',
              'metric':'binary_logloss',
              'verbosity':-1}

#テストデータに入力するためのパラメータを辞書として記述する
#binaryは二項分類 binary_loglossクロスエントロピー
#学習途中の中継的な情報は表示しないため-1を指定する

dtrain = lgbm.Dataset(X_train, y_train) #LightGBMでモデルの作成

model3 = lgbm.train(lgbm_params, dtrain) #パラメータを入力してテストデータの予測

pred3 = model3.predict(X_test) #テストデータを予測して出力
pred3
score_list = []
for i in [x / 100 for x in range(1, 101)]:
    pred = ((pred2*i + pred3*(1-i))) > 0.5
    pred = pred.astype(int)
    score = accuracy_score(y_test, pred)#テストデータの正解率
    score_list.append(score)
    
i = score_list.index(max(score_list))
pred = ((pred2*i + pred3*(1-i))) > 0.5
pred = pred.astype(int)
print('正答率が最大の時の割合はxgb%d: lgbm%d'%(i, 100-i))
print(max(score_list))
pred
import xgboost as xgb #xgboostをインポート

xgb_params = {'objective' : 'binary:logistic',
              'eval_metric' : 'logloss'
             }

dtrain = xgb.DMatrix(X_train, label=y_train)#LightGBMで使用するデータセットの作成
model2 = xgb.train(xgb_params, dtrain) #パラメータを入力してモデルの作成

pred2 = model2.predict(xgb.DMatrix(test)) #テストデータを予測して出力


import lightgbm as lgbm #LightGBMをインポート
from lightgbm import LGBMClassifier
                      
lgbm_params = {'objective':'binary',
              'metric':'binary_logloss',
              'verbosity':-1}

#テストデータに入力するためのパラメータを辞書として記述する
#binaryは二項分類 binary_loglossクロスエントロピー
#学習途中の中継的な情報は表示しないため-1を指定する

dtrain = lgbm.Dataset(X_train, y_train) #LightGBMで使用するデータセットの作成
model3 = lgbm.train(lgbm_params, dtrain) #パラメータを入力してモデルの作成

pred3 = model3.predict(test) #テストデータを予測して出力


pred = ((pred2*0.25 + pred3*(0.75))) > 0.5
pred = pred.astype(int)
pred
solution = pd.DataFrame(pred, test_PassengerId, columns = ['Survived']) #最初に取得したtest_PassengerIdを使用する

solution.to_csv('solution.csv', index_label = ['PassengerId'])

