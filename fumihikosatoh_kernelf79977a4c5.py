import numpy as np 

import pandas as pd 

import os



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier



# CSV読み込み

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

combine = [train_df, test_df]



# 情報表示

train_df.info()

test_df.info()
# Age欠損値算出

train_temp = train_df[['Pclass','Sex','Age']]

test_temp = test_df[['Pclass','Sex','Age']]

temp_df = pd.concat([train_temp, test_temp])

grouped = temp_df.groupby(['Pclass', 'Sex'])

grouped.mean()
# データ変換

for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == 0) & (dataset.Pclass == 1), 'Age'] = 41

    dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == 0) & (dataset.Pclass == 2), 'Age'] = 31

    dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == 0) & (dataset.Pclass == 3), 'Age'] = 26

    dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == 1) & (dataset.Pclass == 1), 'Age'] = 37

    dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == 1) & (dataset.Pclass == 2), 'Age'] = 28

    dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == 1) & (dataset.Pclass == 3), 'Age'] = 22

train_df.info()

test_df.info()
# 不要列削除

train_df = train_df.drop(['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin'], axis=1)

test_df = test_df.drop(['Name', 'Ticket', 'Fare', 'Cabin'], axis=1)

train_df.info()

test_df.info()



# 説明変数、目的変数に分割

X = train_df.drop("Survived", axis=1) # 説明変数

Y = train_df["Survived"]              # 目的変数

X_pred  = test_df.drop("PassengerId", axis=1).copy() #説明変数



# テストデータ分割

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=1)
# グリッドサーチ RandomForest

#rfc = RandomForestClassifier()

#params = {

#    'n_estimators'     : [48, 49, 50, 51, 52],

#    'n_jobs'           : [-1],

#    'random_state'     : [0],

#    'max_depth'        : [5],

#    'min_samples_leaf' : [3],

#}

#params = {

#    'n_estimators'     : [50, 55, 60, 65, 70],

#    'n_jobs'           : [-1],

#    'random_state'     : [0],

#    'max_depth'        : [3, 4, 5, 6, 7],

#    'min_samples_leaf' : [2, 3, 4, 5, 6],

#}

#gs = GridSearchCV(rfc, params, cv=5, scoring='accuracy')



#gs.fit(X_train, Y_train)

#print(gs.best_score_)

#print('Best Parm 01 : ' + str(gs.best_params_))



#clf = gs.best_estimator_

#clf.fit(X_train, Y_train)

#print('Test accuracy: %.3f' % clf.score(X_test, Y_test))
# グリッドサーチ xgboost

#param1 = {

# 'max_depth':[x for x in range(3,10,2)],

# 'min_child_weight':[x for x in range(1,6,2)]

#}

#param2 = {

# 'max_depth':[1,2,3,4],

#}

#param3 = {

#    'gamma': [i/10.0 for i in range(0,5)]

#}

#param4 = {

#    'gamma': [0.455, 0.465, 0.475, 0.485, 0.495]

#}

#param5 = {

# 'subsample':[i/10.0 for i in range(5,10)],

# 'colsample_bytree':[i/10.0 for i in range(5,10)]

#}

#param6 = {

# 'reg_alpha':[0.001, 0.01, 0.1, 1, 10]

#}

#param7 = {

# 'reg_alpha':[0.0005, 0.002, 0.004, 0.006, 0.008]

#}

#param8 = {

# 'n_estimators':[105, 115, 125, 135, 145]

#}

#gsx = GridSearchCV(estimator = XGBClassifier(

#    learning_rate = 0.1, 

#    n_estimators=125, 

#    max_depth=3,

#    min_child_weight=5,

#    gamma=0.21,

#    subsample=0.9,

#    colsample_bytree=0.8,

#    objective= 'binary:logistic', 

#    nthread=4,

#    scale_pos_weight=1, 

#    seed=0

#), param_grid = param7, scoring='roc_auc', n_jobs=-1, iid=False, cv=5)

#gsx.fit(X_train,Y_train)

#gsx.best_params_, gsx.best_score_
# 学習 & 分類 RandomForest

#random_forest = RandomForestClassifier(n_estimators=49,n_jobs=-1,random_state=0,max_depth=5,min_samples_leaf=3)

#random_forest = RandomForestClassifier(n_estimators=49,n_jobs=-1,random_state=1,max_depth=4)

#random_forest.fit(X_train, Y_train)

#Y_pred = random_forest.predict(X_pred)

#random_forest.score(X_train, Y_train)

#acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

#acc_random_forest



# 学習 & 分類 xgboost

xgbc = XGBClassifier(

    learning_rate = 0.075, 

    n_estimators=125,        #150

    reg_alpha=0.004,         #0.65

    max_depth=3,             #3

    min_child_weight=5,      #5

    gamma=0.495,             #0.21

    subsample=0.9,           #0.9

    colsample_bytree=0.7,    #0.8

    objective= 'binary:logistic', 

    nthread=4,

    scale_pos_weight=1, 

    seed=0)

xgbc.fit(X_train, Y_train)

y_test = xgbc.predict(X_test)

Y_pred = xgbc.predict(X_pred)

acc = accuracy_score(Y_test, y_test)

print('Accuracy:', acc)
# csv書き出し

submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('submission.csv', index=False)