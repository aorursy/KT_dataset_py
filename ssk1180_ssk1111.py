import pandas as pd

import numpy as np




# CSVファイルの読み込み

def load_data():

    return pd.read_csv("../input/titanic/train.csv"), pd.read_csv("../input/titanic/test.csv")



train_data, test_data = load_data()
datas = {'train': train_data,

        'test': test_data}
def reload_data():

    train_data, test_data = load_data()

    return {'train': train_data,'test': test_data}
# データの内容：タイタニック号の乗客についての情報と、その乗客が死亡したか否か

# PassengerId: 乗客番号

# Survived; 生存フラグ（1なら生存、0なら死亡）

# Pclass: チケットのクラス（数字が小さい方が高級）

# Name: 乗客の名前

# Sex: 性別

# Age: 年齢

# SibSp: 乗船している配偶者や兄弟の数

# Parch: 乗船している親や子供の数

# Ticket: チケット番号

# Fare: 運賃

# Cabin: 客室番号

# Embarked: 乗船した港の名前

train_data.head()
# 訓点・テストデータの各カラムのNanの数を数える

for k,data in datas.items():

    print('Nan in {}:'.format(k),data.isnull().sum())

    print('='*20)
# データのクリーニングと特徴量の生成

# datas = reload_data()

for k,data in datas.items():

    # 余分な列を削除

    data.drop(['PassengerId', 'Name','Ticket', 'Embarked','Cabin'], axis=1,inplace=True)

    

    # Age列の欠損値を中央値で埋める

    data['Age'].fillna(data['Age'].median(), inplace=True)

    

    # SibSpとParchを合計して、家族の数を表すFamilyを生成

    data['Family'] = data['SibSp'] + data['Parch']

    data.drop(['SibSp', 'Parch'], axis=1,inplace=True)

    

    # 性別を文字列から整数値に変更

    data['Sex'] = data['Sex'].replace({"female": 0, "male": 1})
for k,data in datas.items():

    print(k)

    print(data.head())

    print('='*60)
# 目標データYと入力データXに分離

# 最初の列が目標データ

trainX = datas['train'].iloc[:,1:].values

trainY = datas['train'].iloc[:,0].values

# testX = datas['test'].iloc[:,1:].values

# testY = datas['test'].iloc[:,0].values

testX = datas['test'].values
# (データ数、特徴量数)

print('trainX: ', trainX.shape)

print('trainY: ', trainY.shape)

print('testX: ', testX.shape)

# print('testY: ', testY.shape)
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score

CV=3
%%time

# ロジスティック回帰

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

parameters = [{'penalty': ['l2'], 'C':[10**x for x in range(-4,4)]},

             {'penalty': ['l1'], 'C':[10**x for x in range(-4,4)], 'solver':['liblinear']}]



lr_clf = GridSearchCV(model, parameters, cv=CV)

lr_clf.fit(trainX,trainY)

print(lr_clf.best_params_)

print(lr_clf.best_score_)
%%time

# 決定木

from sklearn import tree

model = tree.DecisionTreeClassifier(random_state=0)

parameters = [{'max_depth': [x for x in range(1,10)]}] 



dt_clf = GridSearchCV(model, parameters, cv=CV)

dt_clf.fit(trainX,trainY)

print(dt_clf.best_params_)

print(dt_clf.best_score_)
%%time

# Wall time: 7.99 s

# Adaboost

from sklearn.ensemble import AdaBoostClassifier

from sklearn import tree

model = AdaBoostClassifier(n_estimators=100, random_state=0)

parameters = [{"base_estimator" : [tree.DecisionTreeClassifier(max_depth=x*2) 

                                  for x in range(1, 10)]}] 



ab_clf = GridSearchCV(model, parameters, cv=CV)

ab_clf.fit(trainX,trainY)

print(ab_clf.best_params_)

print(ab_clf.best_score_)
%%time

# Wall time: 1.24 s

# 勾配ブースティング決定木

from xgboost import XGBClassifier

model = XGBClassifier(random_state=0, n_estimators=100)

parameters = [{"max_depth" : [x for x in range(1,10)]}] 



xgb_clf = GridSearchCV(model, parameters, cv=CV)

xgb_clf.fit(trainX,trainY)

print(xgb_clf.best_params_)

print(xgb_clf.best_score_)
predY = xgb_clf.predict(testX)
submission = pd.read_csv('../input/titanic/gender_submission.csv')

submission
submission['Survived'] = predY

submission.to_csv('my_submission.csv',index=False)