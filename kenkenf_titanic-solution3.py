import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train["Fsize"] = train["SibSp"] + train["Parch"] + 1

test["Fsize"] = test["SibSp"] + test["Parch"] + 1
train['IsAlone'] = 1 #initialize to yes/1 is alone

train['IsAlone'].loc[train['Fsize'] > 1] = 0 # now update to no/0 if family size is greater than 1

test['IsAlone'] = 1 #initialize to yes/1 is alone

test['IsAlone'].loc[test['Fsize'] > 1] = 0 # now update to no/0 if family size is greater than 1
train_title = [i.split(",")[1].split(".")[0].strip() for i in train["Name"]]

train["Title"] = pd.Series(train_title)

train["Title"].head()
test_title = [i.split(",")[1].split(".")[0].strip() for i in test["Name"]]

test["Title"] = pd.Series(test_title)

test["Title"].head()
# Convert to categorical values Title 

train["Title"] = train["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

train["Title"] = train["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})

train["Title"] = train["Title"].astype(int)
# Convert to categorical values Title 

test["Title"] = test["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

test["Title"] = test["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})

test["Title"] = test["Title"].astype(int)
train['Single'] = train['Fsize'].map(lambda s: 1 if s == 1 else 0)

train['SmallF'] = train['Fsize'].map(lambda s: 1 if  s == 2  else 0)

train['MedF'] = train['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)

train['LargeF'] = train['Fsize'].map(lambda s: 1 if s >= 5 else 0)
test['Single'] = test['Fsize'].map(lambda s: 1 if s == 1 else 0)

test['SmallF'] = test['Fsize'].map(lambda s: 1 if  s == 2  else 0)

test['MedF'] = test['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)

test['LargeF'] = test['Fsize'].map(lambda s: 1 if s >= 5 else 0)
train.head()
test.head()
g = sns.heatmap(train[["Survived","Pclass","SibSp","Parch","Age","Fare","Sex","Cabin","Embarked","Fsize","IsAlone","Title","Single"]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
test_passid = test['PassengerId']
train.drop(["Name","Ticket","Cabin"], axis=1, inplace=True)

test.drop(["Name","Ticket","Cabin"], axis=1, inplace=True)
# Ageの穴埋め(中央値)

train["Age"] = train["Age"].fillna(train["Age"].median())

test["Age"] = test["Age"].fillna(test["Age"].median())

# Embarkedは最頻値であるSを代入

train["Embarked"] = train["Embarked"].fillna("S")

# Fareの穴埋め(中央値)

test["Fare"] = test["Fare"].fillna(test["Fare"].median())
train.head()
test.head()
train.drop(["SibSp","Parch"], axis=1, inplace=True)

test.drop(["SibSp","Parch"], axis=1, inplace=True)
train.head()
test.head()
# カテゴリカル変数(Sex,Embarked)をダミー変数で表示

train = train.join(pd.get_dummies(train["Sex"],prefix="sex"))

test = test.join(pd.get_dummies(test["Sex"],prefix="sex"))



train = train.join(pd.get_dummies(train["Embarked"],prefix="emberk"))

test = test.join(pd.get_dummies(test["Embarked"],prefix="emberk"))

train = train.join(pd.get_dummies(train["Title"],prefix="title"))

test = test.join(pd.get_dummies(test["Title"],prefix="title"))
# 使用後のカテゴリカル変数の削除

train.drop(["Sex", "Embarked","Title","Fsize"], axis=1, inplace=True)

test.drop(["Sex", "Embarked","Title","Fsize"], axis=1, inplace=True)
train.head()
test.head()
# PassengerIdを削除

train.drop("PassengerId", axis=1, inplace=True)

test.drop("PassengerId", axis=1, inplace=True)
X_train = train.drop(['Survived'], axis=1) # X_trainはtrainのSurvived列以外

Y_train = train['Survived'] # Y_trainはtrainのSurvived列
train.head()
test.head()
import lightgbm as lgb

from sklearn.model_selection import train_test_split 

from sklearn.metrics import accuracy_score
X_train
Y_train
train_x, valid_x, train_y, valid_y = train_test_split(X_train, Y_train, test_size=0.4, random_state=0)
gbm = lgb.LGBMClassifier(objective='binary')
gbm.fit(train_x, train_y, eval_set = [(valid_x, valid_y)], 

        early_stopping_rounds=20, # 20回連続でlossが下がらなかったら終了 

        verbose=10 # 10round毎に、lossを表示

)
# valid_xについて推論

oof = gbm.predict(valid_x, num_iteration=gbm.best_iteration_)

print(round(accuracy_score(valid_y, oof)*100,2))
gbm_pred = gbm.predict(test, num_iteration=gbm.best_iteration_) 

gbm_submit = pd.DataFrame({

        "PassengerId": test_passid,

        "Survived": gbm_pred

    })

gbm_submit.to_csv("gbm_submission.csv", index=False)



gbm_submit.to_csv('gbm_submission.csv', index=False)