# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
##自分なりのEDAをしてみる

import matplotlib.pyplot as plt

import seaborn as sns
##学習データを特徴量と目的変数に分ける

train_x=train.drop(["Survived"],axis=1)

train_y=train["Survived"]

##テストデータはそのまま

test_x=test.copy()
##変数間の関係を見ていく

##GradientBoostingDecisionTreeの適用を目指す

from sklearn.preprocessing import LabelEncoder

##PassengerIdは予測に寄与しない変数なので削除する

train_x=train_x.drop(["PassengerId"],axis=1)

test_x=test_x.drop(["PassengerId"],axis=1)



##Name,Ticket,Cabinも一度削除,,,Cabinは使えそうだが、、、

train_x=train_x.drop(["Name","Ticket","Cabin"],axis=1)

test_x=test_x.drop(["Name","Ticket","Cabin"],axis=1)



##それぞれのカテゴリカル変数にstr→floatの変換を行う（GBDTでは文字列を入れるとエラーが出るため）

list1=["Sex","Embarked"]

for c in list1:

    ##学習データに基づいてどう変換するのか定める

    le=LabelEncoder()

    le.fit(train_x[c].fillna("NA"))

    ##学習データ、テストデータの変換

    train_x[c] = le.transform(train_x[c].fillna("NA"))

    test_x[c] = le.transform(test_x[c].fillna("NA"))



from xgboost import XGBClassifier



##モデルの作成と学習

model=XGBClassifier(n_estimators=20,random_state=71)

model.fit(train_x,train_y)

##テストデータの予測値を確率で出力

pred=model.predict_proba(test_x)[:,1]



##テストデータの予測値を二値に変換

pred_label=np.where(pred>0.5,1,0)

#提出ファイルの作成

submission=pd.DataFrame({"PassengerId":test["PassengerId"],"Survived":pred_label})

submission.to_csv("submission_first.csv",index=False)
