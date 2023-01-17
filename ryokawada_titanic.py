# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix, log_loss

from sklearn.model_selection import KFold

import itertools



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")

train.head(3)
train_x = train.drop(["Survived"], axis=1)

train_x2 = train.drop(["Survived"], axis=1)

train_y = train["Survived"]
train_x = train_x.drop(["PassengerId","Name","Ticket","Cabin"], axis=1)

train_x2 = train_x2.drop(["PassengerId","Name","Ticket","Cabin"], axis=1)

test_x = test.drop(["PassengerId","Name","Ticket","Cabin"], axis=1)

test_x2 = test.drop(["PassengerId","Name","Ticket","Cabin"], axis=1)
# xgboost用

from sklearn.preprocessing import LabelEncoder



# カテゴリ変数にlabel encoding

for columns in ["Sex","Embarked"]:

    le = LabelEncoder()

    le.fit(train_x[columns].fillna("NA"))

    

    # 変換

    train_x[columns] = le.transform(train_x[columns].fillna("NA"))

    test_x[columns] = le.transform(test_x[columns].fillna("NA"))

    

print(train_x.shape)

train_x.head()
# Logistic regression用

# one-hot encoding

from sklearn.preprocessing import OneHotEncoder



cat_cols = ['Sex', 'Embarked', 'Pclass']

ohe = OneHotEncoder(categories='auto', sparse=False)

ohe.fit(train_x2[cat_cols].fillna('NA'))
# one-hot encodingのダミー変数の列名を作成

ohe_columns = []

for i, c in enumerate(cat_cols):

    ohe_columns += [f'{c}_{v}' for v in ohe.categories_[i]]
# one-hot encodingによる変換

ohe_train_x2 = pd.DataFrame(ohe.transform(train_x2[cat_cols].fillna('NA')), columns=ohe_columns)

ohe_test_x2 = pd.DataFrame(ohe.transform(test_x2[cat_cols].fillna('NA')), columns=ohe_columns)
# one-hot encoding済みの変数を除外する

train_x2 = train_x2.drop(cat_cols, axis=1)

test_x2 = test_x2.drop(cat_cols, axis=1)
# one-hot encodingで変換された変数を結合する

train_x2 = pd.concat([train_x2, ohe_train_x2], axis=1)

test_x2 = pd.concat([test_x2, ohe_test_x2], axis=1)
# 数値変数の欠損値を学習データの平均で埋める

num_cols = ["Age","SibSp","Parch","Fare"]

for col in num_cols:

    train_x2[col].fillna(train_x2[col].mean(), inplace=True)

    test_x2[col].fillna(test_x2[col].mean(), inplace=True)
# 変数Fareを対数変換する

train_x2["Fare"] = np.log1p(train_x2["Fare"])

test_x2["Fare"] = np.log1p(test_x2["Fare"])
model_xgb = XGBClassifier(n_estimators=20,

                          random_state=0)



model_xgb.fit(train_x,train_y)

pred_xgb = model_xgb.predict_proba(test_x)[:,1]
model_lr = LogisticRegression(solver="lbfgs",max_iter=300)

model_lr.fit(train_x2,train_y)

pred_lr = model_lr.predict_proba(test_x2)[:,1]
pred = pred_xgb * 0.8 + pred_lr * 0.2

pred_label = np.where(pred > 0.5, 1, 0)
submission = pd.DataFrame({"PassengerId": test["PassengerId"],

                          "Survived": pred_label})



submission.to_csv("submission.csv", index=False)