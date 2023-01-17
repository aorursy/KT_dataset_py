# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/titanic/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
import re
# データをロード
test_dir = '/kaggle/input/titanic/test.csv'
train_dir = '/kaggle/input/titanic/train.csv'
gender_submission_dir = '/kaggle/input/titanic/gender_submission.csv'
titanic_test = pd.read_csv(test_dir, encoding='utf-8')
titanic_train = pd.read_csv(train_dir, encoding='utf-8')
gender_submission = pd.read_csv(gender_submission_dir, encoding='utf-8')
# データ可視化
print(titanic_train.describe())
print(titanic_train.sort_values('Age'))

# 欠損値計算
print(titanic_train.isna().sum())
print('Test Data:\n{}'.format(titanic_test.isna().sum()))

titanic_train
# 前処理

# 欠損値を補完
train_Mr = titanic_train[titanic_train['Name'].str.contains(' Mr. ')]
train_Mrs = titanic_train[titanic_train['Name'].str.contains(' Mrs. ')]
train_Miss = titanic_train[titanic_train['Name'].str.contains(' Miss. ')]
train_Master = titanic_train[titanic_train['Name'].str.contains(' Master. ')]
test_Mr = titanic_test[titanic_test['Name'].str.contains(' Mr. ')]
test_Mrs = titanic_test[titanic_test['Name'].str.contains(' Mrs. ')]
test_Miss = titanic_test[titanic_test['Name'].str.contains(' Miss. ')]
test_Master = titanic_test[titanic_test['Name'].str.contains(' Master. ')]

# 各平均値を求める
train_mr_mean = int(train_Mr['Age'].dropna().mean())
train_mrs_mean = int(train_Mrs['Age'].dropna().mean())
train_miss_mean = int(train_Miss['Age'].dropna().mean())
train_master_mean = int(train_Master['Age'].dropna().mean())
train_all_median = int(titanic_train['Age'].dropna().median())
test_mr_mean = int(test_Mr['Age'].dropna().mean())
test_mrs_mean = int(test_Mrs['Age'].dropna().mean())
test_miss_mean = int(test_Miss['Age'].dropna().mean())
test_master_mean = int(test_Master['Age'].dropna().mean())
test_all_median = int(titanic_test['Age'].dropna().median())

# 求めた平均値を使って欠損値を穴埋めする
titanic_train['Age'][train_Mr['Age']].fillna(train_mr_mean,inplace=True)
titanic_train['Age'][train_Mrs['Age']].fillna(train_mrs_mean, inplace=True)
titanic_train['Age'][train_Miss['Age']].fillna(train_miss_mean, inplace=True)
titanic_train['Age'][train_Master['Age']].fillna(train_master_mean, inplace=True)
titanic_train['Age'].fillna(train_all_median, inplace=True)
titanic_test['Age'][test_Mr['Age']].fillna(test_mr_mean,inplace=True)
titanic_test['Age'][test_Mrs['Age']].fillna(test_mrs_mean, inplace=True)
titanic_test['Age'][test_Miss['Age']].fillna(test_miss_mean, inplace=True)
titanic_test['Age'][test_Master['Age']].fillna(test_master_mean, inplace=True)
titanic_test['Age'].fillna(test_all_median, inplace=True)
# print('train(Age) num of Nan %s' % titanic_train['Age'].isna().sum())
# print('test(Age) num of Nan %s' % titanic_test['Age'].isna().sum())
# print('Curent nan :\n{}'.format(titanic_train.isna().sum()))

# データのCabinを調査&整理
cabin_bool = titanic_train['Cabin'].isna()

# Cabinの欠損値をUに変更
titanic_train['Cabin'].fillna('U',inplace=True)
titanic_test['Cabin'].fillna('U', inplace=True)

# Cabinを整理(train)
print(titanic_train['Cabin'].unique())
titanic_train_A = titanic_train[titanic_train['Cabin'].str.match('A')]
titanic_train_B = titanic_train[titanic_train['Cabin'].str.match('B')]
titanic_train_C = titanic_train[titanic_train['Cabin'].str.match('C')]
titanic_train_D = titanic_train[titanic_train['Cabin'].str.match('D')]
titanic_train_E = titanic_train[titanic_train['Cabin'].str.match('E')]
titanic_train_F = titanic_train[titanic_train['Cabin'].str.match('F\d+')]
titanic_train_G = titanic_train[titanic_train['Cabin'].str.match('G')]
titanic_train_H = titanic_train[titanic_train['Cabin'].str.match('H')]
titanic_train_T = titanic_train[titanic_train['Cabin'].str.match('T')]

titanic_train.loc[titanic_train['Cabin'].str.match('A'),'Cabin'] = 'A'
titanic_train.loc[titanic_train['Cabin'].str.match('B'),'Cabin'] = 'B'
titanic_train.loc[titanic_train['Cabin'].str.match('C'),'Cabin'] = 'C'
titanic_train.loc[titanic_train['Cabin'].str.match('D'),'Cabin'] = 'D'
titanic_train.loc[titanic_train['Cabin'].str.match('E'),'Cabin'] = 'E'
titanic_train.loc[titanic_train['Cabin'].str.match('F\d+'),'Cabin'] = 'F'
titanic_train.loc[titanic_train['Cabin'].str.match('\w\sE+'), 'Cabin'] = 'F'
titanic_train.loc[titanic_train['Cabin'].str.match('G'),'Cabin'] = 'G'
titanic_train.loc[titanic_train['Cabin'].str.match('\w\sG+'),'Cabin'] = 'G'
titanic_train.loc[titanic_train['Cabin'].str.match('H'),'Cabin'] = 'H'
titanic_train.loc[titanic_train['Cabin'].str.match('T'),'Cabin'] = 'U'
print('renamed titanic[Cabin]:\n{}'.format(titanic_train['Cabin'].unique()))

# test
print(titanic_test['Cabin'].unique())
titanic_test_A = titanic_test[titanic_test['Cabin'].str.match('A')]
titanic_test_B = titanic_test[titanic_test['Cabin'].str.match('B')]
titanic_test_C = titanic_test[titanic_test['Cabin'].str.match('C')]
titanic_test_D = titanic_test[titanic_test['Cabin'].str.match('D')]
titanic_test_E = titanic_test[titanic_test['Cabin'].str.match('E')]
titanic_test_F = titanic_test[titanic_test['Cabin'].str.match('F\d+')]
titanic_test_G = titanic_test[titanic_test['Cabin'].str.match('G')]
titanic_test_H = titanic_test[titanic_test['Cabin'].str.match('H')]
titanic_test_T = titanic_test[titanic_test['Cabin'].str.match('T')]

titanic_test.loc[titanic_test['Cabin'].str.match('A'),'Cabin'] = 'A'
titanic_test.loc[titanic_test['Cabin'].str.match('B'),'Cabin'] = 'B'
titanic_test.loc[titanic_test['Cabin'].str.match('C'),'Cabin'] = 'C'
titanic_test.loc[titanic_test['Cabin'].str.match('D'),'Cabin'] = 'D'
titanic_test.loc[titanic_test['Cabin'].str.match('E'),'Cabin'] = 'E'
titanic_test.loc[titanic_test['Cabin'].str.match('F\d+'),'Cabin'] = 'F'
titanic_test.loc[titanic_test['Cabin'].str.match('\w\sE+'), 'Cabin'] = 'F'
titanic_test.loc[titanic_test['Cabin'].str.match('G'),'Cabin'] = 'G'
titanic_test.loc[titanic_test['Cabin'].str.match('\w\sG+'),'Cabin'] = 'G'
titanic_test.loc[titanic_test['Cabin'].str.match('H'),'Cabin'] = 'H'
titanic_test.loc[titanic_test['Cabin'].str.match('T'),'Cabin'] = 'U'
print('renamed titanic[Cabin]:\n{}'.format(titanic_test['Cabin'].unique()))

# survivedとCabinとの関係を調査（死亡0）
titanic_train[titanic_train['Survived']==0]

# Ticketの文字を正規表現で整理・分類
num_ticket = titanic_train[titanic_train['Ticket'].str.match('\d+')]
num_ticket['Ticket'] = num_ticket['Ticket'].apply(lambda x : int(x))
obj_ticket = titanic_train[titanic_train['Ticket'].str.match('[a-zA-Z]+')]
obj_ticket_A = obj_ticket[obj_ticket['Ticket'].str.match('A.+')]
obj_ticket_PC = obj_ticket[obj_ticket['Ticket'].str.match('PC.+')]
obj_ticket_STON = obj_ticket[obj_ticket['Ticket'].str.match('STON.+')]
obj_ticket_PP = obj_ticket[obj_ticket['Ticket'].str.match('PP.+')]
obj_ticket_C = obj_ticket[obj_ticket['Ticket'].str.match('C\s+')]
obj_ticket_CA = obj_ticket[obj_ticket['Ticket'].str.match('C\.+A\.+')]

num_ticket_test = titanic_test[titanic_test['Ticket'].str.match('\d+')]
num_ticket_test['Ticket'] = num_ticket_test['Ticket'].apply(lambda x : int(x))
obj_ticket_test = titanic_test[titanic_test['Ticket'].str.match('[a-zA-Z]+')]
obj_ticket_A_test = obj_ticket_test[obj_ticket_test['Ticket'].str.match('A.+')]
obj_ticket_PC_test = obj_ticket_test[obj_ticket_test['Ticket'].str.match('PC.+')]
obj_ticket_STON_test = obj_ticket_test[obj_ticket_test['Ticket'].str.match('STON.+')]
obj_ticket_PP_test = obj_ticket_test[obj_ticket_test['Ticket'].str.match('PP.+')]
obj_ticket_C_test = obj_ticket_test[obj_ticket_test['Ticket'].str.match('C\s+')]
obj_ticket_CA_test = obj_ticket_test[obj_ticket_test['Ticket'].str.match('C\.+A\.+')]

# 正規表現で分けたデータを新しいラベルをつける(1-9)
obj_ticket1 = obj_ticket.copy()
obj_ticket['Ticket'] = '7'
obj_ticket.loc[obj_ticket1['Ticket'].str.match('A.+'),'Ticket'] = '1'
obj_ticket.loc[obj_ticket1['Ticket'].str.match('PC.+'), 'Ticket'] ='2'
obj_ticket.loc[obj_ticket1['Ticket'].str.match('STON.+'), 'Ticket'] = '3'
obj_ticket.loc[obj_ticket1['Ticket'].str.match('PP.+'), 'Ticket'] = '4'
obj_ticket.loc[obj_ticket1['Ticket'].str.match('C\s+'), 'Ticket'] = '5'
obj_ticket.loc[obj_ticket1['Ticket'].str.match('C\.+A\.+'), 'Ticket'] = '6'
num_ticket1 = num_ticket.copy()
num_ticket.loc[num_ticket1['Ticket']  < 10000, 'Ticket'] = '8'
num_ticket.loc[(num_ticket1['Ticket'] >= 10000) & (num_ticket1['Ticket'] < 15000), 'Ticket'] = '9'
num_ticket.loc[(num_ticket1['Ticket'] >= 15000) & (num_ticket1['Ticket'] < 20000), 'Ticket'] = '10'
num_ticket.loc[(num_ticket1['Ticket'] >= 20000) & (num_ticket1['Ticket'] < 25000), 'Ticket'] = '11'
num_ticket.loc[(num_ticket1['Ticket'] >= 25000) & (num_ticket1['Ticket'] < 30000), 'Ticket'] = '12'
num_ticket.loc[num_ticket1['Ticket'] >= 30000, 'Ticket'] = '13'
train = pd.concat([obj_ticket, num_ticket], axis=0)

obj_ticket_test1 = obj_ticket_test.copy()
obj_ticket_test['Ticket'] = '7'
obj_ticket_test.loc[obj_ticket_test1['Ticket'].str.match('A.+'),'Ticket'] = '1'
obj_ticket_test.loc[obj_ticket_test1['Ticket'].str.match('PC.+'), 'Ticket'] ='2'
obj_ticket_test.loc[obj_ticket_test1['Ticket'].str.match('STON.+'), 'Ticket'] = '3'
obj_ticket_test.loc[obj_ticket_test1['Ticket'].str.match('PP.+'), 'Ticket'] = '4'
obj_ticket_test.loc[obj_ticket_test1['Ticket'].str.match('C\s+'), 'Ticket'] = '5'
obj_ticket_test.loc[obj_ticket_test1['Ticket'].str.match('C\.+A\.+'), 'Ticket'] = '6'
num_ticket_test1 = num_ticket_test.copy()
num_ticket_test.loc[num_ticket_test1['Ticket']  < 10000, 'Ticket'] = '8'
num_ticket_test.loc[(num_ticket_test1['Ticket'] >= 10000) & (num_ticket_test1['Ticket'] < 15000), 'Ticket'] = '9'
num_ticket_test.loc[(num_ticket_test1['Ticket'] >= 15000) & (num_ticket_test1['Ticket'] < 20000), 'Ticket'] = '10'
num_ticket_test.loc[(num_ticket_test1['Ticket'] >= 20000) & (num_ticket_test1['Ticket'] < 25000), 'Ticket'] = '11'
num_ticket_test.loc[(num_ticket_test1['Ticket'] >= 25000) & (num_ticket_test1['Ticket'] < 30000), 'Ticket'] = '12'
num_ticket_test.loc[num_ticket_test1['Ticket'] >= 30000, 'Ticket'] = '13'
test = pd.concat([obj_ticket_test, num_ticket_test], axis=0)

# Embarkedの欠損値を削除
train['Embarked'].fillna('S', inplace=True)  # 'S' is the most common symbol 
test['Embarked'].fillna('S', inplace=True)

# Test dataのFareの欠損値を中央値で補完
test['Fare'].fillna(test['Fare'].median(), inplace=True)

# Data後で使うのでソート
train.sort_values('PassengerId', inplace=True)
test.sort_values('PassengerId', inplace=True)

train['Family'] = train['SibSp'] + train['Parch']
test['Family'] = test['SibSp'] + test['Parch']

# ワンホットラベル化
def get_dummy(df):
    df['Pclass'] = df['Pclass'].astype(np.str)
    temp = pd.get_dummies(df, columns = ['Pclass','Sex', 'Ticket', 'Cabin', 'Embarked'],drop_first=False)
    temp['PassengerId'] = df['PassengerId']
    return temp

train_dm = get_dummy(train)
test_dm = get_dummy(test)

# いらない値を削除
train_dm.drop(columns=['PassengerId', 'Name', 'Parch', 'SibSp'],inplace=True)
test_dm.drop(columns=['PassengerId', 'Name', 'Parch', 'SibSp'],inplace=True)

# 訓練用データのtarget（死亡生存）を分ける
train_target = train_dm['Survived']
train_data = train_dm[train_dm.columns[train_dm.columns != 'Survived']]
# LightGBM勾配Boosting木で訓練
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

# 交差検証
skf = StratifiedKFold(n_splits=10,
                      shuffle=True,
                      random_state=0)

# LightGBMモデルで訓練
model = lgb.LGBMClassifier(silent=False)

# GridSearchで検索するパラメーター
param_grid = {"max_depth": [2, 3, 5, 10],
              "learning_rate" : [0.01, 0.05, 0.1],
              "num_leaves": [10, 100, 300, 900],
              "n_estimators": [100, 200, 500]
             }

grid_result = GridSearchCV(estimator=model,
                           param_grid=param_grid,
                           scoring='balanced_accuracy',
                           cv=skf,
                           verbose=3,
                           return_train_score=True,
                           n_jobs=-1)

grid_result.fit(train_data, train_target)


# LightGBM訓練 + 予測値
X_train, X_val, y_train, y_val = tts(train_data, train_target,
                                           shuffle=True,
                                           stratify=train_target)

lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_val = lgb.Dataset(X_val, label=y_val)

lgb_param = {
    'objective' : 'binary',
    'metrics' : 'binary_logloss',
}

# 辞書結合（もしかするともっといい方法があるかも）
params = grid_result.best_params_
params.update(lgb_param)

model = lgb.train(params,
                  lgb_train,
                  valid_sets=lgb_val,
                  num_boost_round=10000,
                  early_stopping_rounds=1000
                  )
# GridSearchCVのベストスコア（交差検証済み）
score = grid_result.best_score_
print('Best Score:{:.3f}'.format(score))
print('Best parametor :\n{}'.format(grid_result.best_params_))

# 予測値
y_pred_proba = model.predict(test_dm,num_iteration=model.best_iteration)
y_pred = (y_pred_proba > 0.5).astype(int)
print('prediction :\n{}'.format(y_pred))

# 提出用ファイルを作成&saving
test = titanic_test
test['Survived'] = y_pred.astype(int)
test[['PassengerId', 'Survived']].to_csv('submission.csv', encoding='utf-8', index=False)

# 確認
for dirname, _, filenames in os.walk('./'):
    for filename in filenames:
        print(os.path.join(dirname, filename))