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

submission = pd.read_csv("../input/gender_submission.csv")

test = pd.read_csv("../input/test.csv", index_col='PassengerId')

train = pd.read_csv("../input/train.csv", index_col='PassengerId')
# 데이터 시각화 패키지

%matplotlib inline



import seaborn as sns

import matplotlib.pyplot as plt
#성별분석



sns.countplot(data=train, x='Sex', hue='Survived')
pd.pivot_table(train, index='Sex', values='Survived')
#Pclass 분석



sns.countplot(data=train, x="Pclass", hue="Survived")
pd.pivot_table(train, index='Pclass', values='Survived')
#Embarked 분석

sns.countplot(data=train, x="Embarked", hue="Survived")
pd.pivot_table(train, index="Embarked", values="Survived")
# 나이와 운임의 상관관계 분석

# implot 이용



sns.lmplot(data=train, x="Age", y="Fare", hue="Survived", fit_reg=False)
# 아웃라이어 제거

low_fare = train[train["Fare"] < 500]

train.shape, low_fare.shape

sns.lmplot(data=low_fare, x="Age", y="Fare", hue="Survived", fit_reg=False)
# 운임요금 100달러 이상도 아웃라이어로 간주하자

low_low_fare = train[train["Fare"] < 100]



train.shape, low_fare.shape, low_low_fare.shape
sns.lmplot(data=low_low_fare, x="Age", y="Fare", hue="Survived", fit_reg=False)
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1 # +1은 자기 자신

train[["SibSp", "Parch", "FamilySize"]].head()
sns.countplot(data=train, x="FamilySize", hue="Survived")
train.loc[train["FamilySize"] == 1, "FamilyType"] = "Single"

train.loc[(train["FamilySize"] > 1) & (train["FamilySize"] < 5), "FamilyType"] = "Nuclear"

train.loc[train["FamilySize"] >= 5, "FamilyType"] = "Big"

train[["FamilySize", "FamilyType"]].head(10)
sns.countplot(data=train, x="FamilyType", hue="Survived")
pd.pivot_table(data=train, index="FamilyType", values="Survived")
train["Name"].head()
# 이름을 입력하면 가운데 부분(타이틀)을 반환해주는 함수를 만들어보자



def get_title(name):

    

    return name.split(',')[1].split('.')[0]



# 함수를 Name컬럼에 적용시킨뒤 중복값을 제거하자(unique)



train['Name'].apply(get_title).unique()
train.loc[train["Name"].str.contains("Mr"), "Title"] = "Mr"

train.loc[train["Name"].str.contains("Miss"), "Title"] = "Miss"

train.loc[train["Name"].str.contains("Mrs"), "Title"] = "Mrs"

train.loc[train["Name"].str.contains("Master"), "Title"] = "Master"

train[["Name", "Title"]].head(10)
sns.countplot(data=train, x="Title", hue="Survived")
pd.pivot_table(train, index="Title", values="Survived")
# 남자는 0, 여자는 1



train.loc[train["Sex"] == "male", "Sex_encode"] = 0

train.loc[train["Sex"] == "female", "Sex_encode"] = 1



train[["Sex", "Sex_encode"]].head()
test.loc[test["Sex"] == "male", "Sex_encode"] = 0

test.loc[test["Sex"] == "female", "Sex_encode"] = 1



test[["Sex", "Sex_encode"]].head()
# Fare 컬럼의 Nan값 0으로 채워주기

# 사본 컬럼을 하나 더 만들어서 사용하자



train["Fare_fillin"] = train["Fare"]

train[["Fare", "Fare_fillin"]].head()

test["Fare_fillin"] = test["Fare"]



test[["Fare", "Fare_fillin"]].head()
test.loc[test["Fare"].isnull(), "Fare_fillin"] = 0



test.loc[test["Fare"].isnull(), ["Fare", "Fare_fillin"]]
train["Embarked_C"] = train["Embarked"] == "C"

train["Embarked_S"] = train["Embarked"] == "S"

train["Embarked_Q"] = train["Embarked"] == "Q"



train[["Embarked", "Embarked_C", "Embarked_S", "Embarked_Q"]].head()
test["Embarked_C"] = test["Embarked"] == "C"

test["Embarked_S"] = test["Embarked"] == "S"

test["Embarked_Q"] = test["Embarked"] == "Q"



test[["Embarked", "Embarked_C", "Embarked_S", "Embarked_Q"]].head()
train["Child"] = train["Age"] < 15



train[["Age", "Child"]].head(10)
test["Child"] = test["Age"] < 15



test[["Age", "Child"]].head(10)
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1



train[["SibSp", "Parch", "FamilySize"]].head()
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1



test[["SibSp", "Parch", "FamilySize"]].head()
#  가족수에 따라 범주를 나누어준다.



train["Single"] = train["FamilySize"] == 1

train["Nuclear"] = (train["FamilySize"] > 1) & (train["FamilySize"] < 5)

train["Big"] = train["FamilySize"] >= 5



train[["FamilySize", "Single", "Nuclear", "Big"]].head(10)
test["Single"] = test["FamilySize"] == 1

test["Nuclear"] = (test["FamilySize"] > 1) & (test["FamilySize"] < 5)

test["Big"] = test["FamilySize"] >= 5



test[["FamilySize", "Single", "Nuclear", "Big"]].head(10)
train["Master"] = train["Name"].str.contains("Master")

train[["Name", "Master"]].head(10)
test["Master"] = test["Name"].str.contains("Master")

test[["Name", "Master"]].head(10)
feature_names = ["Pclass", "Sex_encode", "Fare_fillin",

                 "Embarked_C", "Embarked_S", "Embarked_Q",

                 "Child", "Single", "Nuclear", "Big", "Master"]

feature_names



# Embarked가 셋으로 나뉨

# Child는 15세 미만 true/false

# sipsp, parch가 family size로 처리됌
# 생존여부를 Label로 지정



label_name = "Survived"

label_name
# 지정된 컬럼들만 모아서 X_train이라고 새로 하나 만든다

X_train = train[feature_names]



X_train.head()
X_test = test[feature_names]



X_test.head()
y_train = train[label_name]



y_train.head()
from sklearn.tree import DecisionTreeClassifier



model = DecisionTreeClassifier(max_depth=8, random_state=0)

model
model.fit(X_train, y_train)
import graphviz

from sklearn.tree import export_graphviz



dot_tree = export_graphviz(model,

                           feature_names=feature_names,

                           class_names=["Perish", "Survived"],

                           out_file=None)



graphviz.Source(dot_tree)
predictions = model.predict(X_test)



predictions[0:10]
# 제출



submission = pd.read_csv("../input/gender_submission.csv", index_col="PassengerId")



print(submission.shape)

submission.head()

submission["Survived"] = predictions

print(submission.shape)

submission.head()
submission.to_csv("./decision-tree_0.81818.csv")