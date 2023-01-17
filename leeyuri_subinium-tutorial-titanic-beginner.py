# 필요한 라이브러리를 우선 불러옵니다.



## 데이터 분석 관련

import pandas as pd

from pandas import Series, DataFrame

import numpy as np



## 데이터 시각화 관련

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid') # matplotlib의 스타일에 관련한 함

## 그래프 출력에 필요한 IPython 명령어

%matplotlib inline 



## Scikit-Learn의 다양한 머신러닝 모듈을 불러옵니다.

## 분류 알고리즘 중에서 선형회귀, 서포트벡터머신, 랜덤포레스트, K-최근접이웃 알고리즘을 사용해보려고 합니다.

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier
# 데이터를 우선 가져와야합니다.

train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")



# 데이터 미리보기

train_df.head()

train_df.info()

print('-'*20)

test_df.info()
train_df = train_df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)

test_df = test_df.drop(['Name','Ticket'], axis=1)
train_df['Pclass'].value_counts()
pclass_train_dummies = pd.get_dummies(train_df['Pclass'])

pclass_test_dummies = pd.get_dummies(test_df['Pclass'])



train_df.drop(['Pclass'], axis=1, inplace=True)

test_df.drop(['Pclass'], axis=1, inplace=True)



train_df = train_df.join(pclass_train_dummies)

test_df = test_df.join(pclass_test_dummies)
sex_train_dummies = pd.get_dummies(train_df['Sex'])

sex_test_dummies = pd.get_dummies(test_df['Sex'])



sex_train_dummies.columns = ['Female', 'Male']

sex_test_dummies.columns = ['Female', 'Male']



train_df.drop(['Sex'], axis=1, inplace=True)

test_df.drop(['Sex'], axis=1, inplace=True)



train_df = train_df.join(sex_train_dummies)

test_df = test_df.join(sex_test_dummies)
train_df["Age"].fillna(train_df["Age"].mean() , inplace=True)

test_df["Age"].fillna(train_df["Age"].mean() , inplace=True)
test_df["Fare"].fillna(0, inplace=True)
train_df = train_df.drop(['Cabin'], axis=1)

test_df = test_df.drop(['Cabin'], axis=1)
train_df['Embarked'].value_counts()
test_df['Embarked'].value_counts()
train_df["Embarked"].fillna('S', inplace=True)

test_df["Embarked"].fillna('S', inplace=True)
embarked_train_dummies = pd.get_dummies(train_df['Embarked'])

embarked_test_dummies = pd.get_dummies(test_df['Embarked'])



embarked_train_dummies.columns = ['S', 'C', 'Q']

embarked_test_dummies.columns = ['S', 'C', 'Q']



train_df.drop(['Embarked'], axis=1, inplace=True)

test_df.drop(['Embarked'], axis=1, inplace=True)



train_df = train_df.join(embarked_train_dummies)

test_df = test_df.join(embarked_test_dummies)
X_train = train_df.drop("Survived",axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop("PassengerId",axis=1).copy()
# Logistic Regression



logreg = LogisticRegression()

logreg.fit(X_train, Y_train)



Y_pred = logreg.predict(X_test)



logreg.score(X_train, Y_train)
# Support Vector Machines



svc = SVC()



svc.fit(X_train, Y_train)



Y_pred = svc.predict(X_test)



svc.score(X_train, Y_train)
# Random Forests



random_forest = RandomForestClassifier(n_estimators=100)



random_forest.fit(X_train, Y_train)



Y_pred = random_forest.predict(X_test)



random_forest.score(X_train, Y_train)
knn = KNeighborsClassifier(n_neighbors = 3)



knn.fit(X_train, Y_train)



Y_pred = knn.predict(X_test)



knn.score(X_train, Y_train)
# Random Forests



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)



submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('titanic.csv', index=False)