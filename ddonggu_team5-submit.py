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
#데이터 불러오기

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

combine = [train, test]
train.head()
#"train 데이터 셋에 비어있는 데이터 확인"

train.isnull().sum()
#"test 데이터 셋에 비어있는 데이터 확인"

test.isnull().sum()
# 사망 생존자수 알아보자

f, ax = plt.subplots(nrows=1, ncols=1)

sns.countplot(data=train, x="Survived", ax=ax)

ax.set_title("Dead vs Survived")

plt.show()


f, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 5))

# 남녀별 생존률

sns.barplot(data=train, x="Sex", y="Survived", ax=ax[0])

ax[0].set_title("Survival rate by sex")

# 남녀별 사망, 생존자 수

sns.countplot(data=train, x="Sex", hue="Survived", ax=ax[1])

ax[1].set_title("Sex: Dead vs Survived")

plt.show()
#성별과 생존의 상관관계 분석

train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
f, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 5))



sns.countplot(data=train, x="Pclass", hue="Survived", ax=ax[0])

ax[0].set_title("Pclass: Dead vs Survived")



sns.factorplot(data=train, x="Pclass", y="Survived", hue="Sex", ax=ax[1])

plt.close(2)

plt.show()


#"Pclass 별 생존확률"

pd.pivot_table(train, index="Pclass", values="Survived").sort_values(by='Survived', ascending=False)
#"Pclass & 성별별 생존확률"

pd.pivot_table(train, index=["Pclass","Sex"], values="Survived").T
for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(train['Title'], train['Sex'])
for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

    'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
f, ax = plt.subplots(nrows=1, ncols=1)

sns.countplot(data=train, x="Title", hue="Survived", ax=ax)

ax.set_title("Title: Dead vs Survived")

plt.show()
#"호칭별 생존확률"

pd.pivot_table(train, index="Title", values="Survived").T
# 비어있는 값이 존재하므로 채워주어야 한다.

#"비어있는 데이터 개수 출력"

train["Age"].isnull().sum()
# age의 평균에서 분산값만 큼 빼고, 더한 값들 사이에서 랜덤으로 빈값을 채워넣는다.

age_avg = train["Age"].mean()

age_std = train['Age'].std()

age_null_count = train['Age'].isnull().sum()

age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

train['Age'][np.isnan(train['Age'])] = age_null_random_list

train['Age'] = train['Age'].astype(int)

pd.cut(train['Age'], 5)
#age값이 다 채워진걸 알 수 있다.

train['Age']
# 다양한 관점에서 age값을 봐보자

f, ax = plt.subplots(2, 2, figsize=(18, 10))



Age_0 = train[train["Survived"] == 0]

sns.distplot(Age_0["Age"], ax=ax[0][0])

ax[0][0].set_title("Age distribution of deaths")



Age_1 = train[train["Survived"] == 1]

sns.distplot(Age_1["Age"], ax=ax[0][1])

ax[0][1].set_title("Age distribution of survivors")



sns.violinplot(data=train, x="Sex", y="Age", hue="Survived", split=True, ax=ax[1][0])

ax[1][0].set_title("Sex & Age vs Survived")

ax[1][0].set_yticks(range(0, 110, 10))



sns.violinplot(data=train, x="Pclass", y="Age", hue="Survived", split=True, ax=ax[1][1])

ax[1][1].set_title("Pclass & Age vs Survived")

ax[1][1].set_yticks(range(0, 110, 10))

print("가장 높은 운임비: ", train["Fare"].max())

print("가낭 낮은 운임비: ", train["Fare"].min())

print("운임비의 평균: ", train["Fare"].mean())
# Pclass별 운임비의 분포를 알아보자.

f, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))



sns.distplot(train[train["Pclass"] == 1].Fare, ax=ax[0])

ax[0].set_title("Fares in Pclass 1")

ax[0].set_xticks(range(0, 200, 50))



sns.distplot(train[train["Pclass"] == 2].Fare, ax=ax[1])

ax[1].set_title("Fares in Pclass 2")

ax[1].set_xticks(range(0, 200, 50))



sns.distplot(train[train["Pclass"] == 3].Fare, ax=ax[2])

ax[2].set_title("Fares in Pclass 3")

ax[2].set_xticks(range(0, 200, 50))

plt.show()
# 운임비를 4등분 해주어 각 구간의 평균을 구하자.

train["Fare_quater"] = pd.qcut(train["Fare"], 4)

#"Fare_quater별 생존 확률"

pd.pivot_table(train, index="Fare_quater", values="Survived").T
# 운임비를 4등분 피벗 테이블 자료를 바탕으로 운임비를 카테고리화한 컬럼을 추가하자.

train['Fare_cat']=0

train.loc[train['Fare']<=7.91,'Fare_cat'] = 0

train.loc[(train['Fare']>7.91) & (train['Fare']<=14.454),'Fare_cat'] = 1

train.loc[(train['Fare']>14.454) & (train['Fare']<=31),'Fare_cat'] = 2

train.loc[(train['Fare']>31) & (train['Fare']<=513),'Fare_cat'] = 3
#운임비 구간별 생존자수

f, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 5))



sns.countplot(data=train, x="Fare_cat", hue="Survived", ax=ax[0])

ax[0].set_title("Fare_cat: Dead vs Survived")



sns.factorplot(data= train, x="Fare_cat", y="Survived", hue="Sex", ax=ax[1])



plt.close(2)

plt.show()


f, ax = plt.subplots(nrows=2, ncols=2, figsize=(18, 10))



sns.countplot(data=train, x="Embarked", hue="Survived", ax=ax[0][0])

ax[0][0].set_title("Embarked: Dead vs Survived")



sns.countplot(data=train, x="Embarked", hue="Sex", ax=ax[0][1])

ax[0][1].set_title("Embarked: male vs female")



sns.countplot(data=train, x="Embarked", hue="Pclass", ax=ax[1][0])

ax[1][0].set_title("Embarked: Pcalss 1 vs 2 vs 3")



sns.countplot(data=train, x="Embarked", hue="Fare_cat", ax=ax[1][1])

ax[1][1].set_title("Embarked: Fare_cat 0 vs 1 vs 2 vs 3")



plt.show()
#"선착장별 생존율"

pd.pivot_table(train, index="Embarked", values="Survived").T
# 형재/자매/배우자 + 부모/자식 + 1을 하여 가족 구성원 크기를 Family 컬럼으로 만들어준다.

train["Family"] = train["SibSp"] + train["Parch"] + 1

train[["SibSp", "Parch", "Family"]].head(3)
#Family size별 생존자수

f, ax = plt.subplots(nrows=2, ncols=2, figsize=(18, 10))



sns.countplot(data=train, x="Family", hue="Survived", ax=ax[0][0])

ax[0][0].set_title("Family: Dead vs Survived")



train["Family_single"] = train["Family"] == 1

train["Family_small"] = (train["Family"] > 1) & (train["Family"] < 5)

train["Family_big"] = train["Family"] >= 5



sns.countplot(data=train, x="Family_single", hue="Survived", ax=ax[0][1])

ax[0][1].set_title("Family_single: Dead vs Survived")



sns.countplot(data=train, x="Family_small", hue="Survived", ax=ax[1][0])

ax[1][0].set_title("Family_small: Dead vs Survived")



sns.countplot(data=train, x="Family_big", hue="Survived", ax=ax[1][1])

ax[1][1].set_title("Family_big: Dead vs Survived")

plt.show()
#"혼자 탑승한 경우 생존율"

pd.pivot_table(train, index="Family_single", values="Survived").T
#"2~4명이 탑승한 경우 생존율"

pd.pivot_table(train, index="Family_small", values="Survived").T
#"5명 이상이 탑승한 경우 생존율"

pd.pivot_table(train, index="Family_big", values="Survived").T
#남자는0 여자는 1로 한다.

for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)



train.head()
#age를 5구간으로 나눠서 그 구간에 인덱스를 넣어준다.

train['AgeBand'] = pd.cut(train['Age'], 5)

train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
for dataset in combine:

    dataset['Age value']=0

    dataset.loc[ dataset['Age'] <= 16, 'Age value'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age value'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age value'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age value'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age value']

train.head()
#Fare의 결측치 처리

test['Fare'].fillna(test['Fare'].dropna().median(), inplace=True)

test.head()
# 운임비를 4등분 해주어 각 구간의 평균을 구합니다.

test["Fare_quater"] = pd.qcut(test["Fare"], 4)
# 운임비를 4등분 피벗 테이블 자료를 바탕으로 운임비를 카테고리화한 컬럼을 추가한다.

test['Fare_cat']=0

test.loc[test['Fare']<=7.91,'Fare_cat'] = 0

test.loc[(test['Fare']>7.91) & (test['Fare']<=14.454),'Fare_cat'] = 1

test.loc[(test['Fare']>14.454) & (test['Fare']<=31),'Fare_cat'] = 2

test.loc[(test['Fare']>31) & (test['Fare']<=513),'Fare_cat'] = 3

test.head()
# Embarked 결측치 처리

freq_port = train.Embarked.dropna().mode()[0]

freq_port
# Embarked 의 최빈값인 S로 결측치를 채운다.

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
#Embarked 의 데이터 수치화

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



train.head(3)
test.head(3)
# 위에 train 데이터를 처리 한 것과 같이 처리한다.

test["Family"] = test["SibSp"] + test["Parch"] + 1

test["Family_single"] = test["Family"] == 1

test["Family_small"] = (test["Family"] > 1) & (test["Family"] < 5)

test["Family_big"] = test["Family"] >= 5

test.head()
train.head()
# 각 title에 인덱스를 부여한다.

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



train.head()
test.head()
feature_list = ["Sex", "Pclass", "Title", "Age value", "Fare_cat", "Embarked", "Family_single", "Family_small", "Family_big"]

#"Feature로 사용할 컬럼들 입니다."

print(feature_list)
label = "Survived"

#"Label로 사용할 컬럼 입니다."

print(label)
# x_train 값

x_train = train[feature_list]

x_train.head(3)
#x_test 값

x_test = test[feature_list]

x_test.head(3)
#y_train 값

y_train = train[label]

y_train.head(3)
# Support Vector Machines



svc = SVC()

svc.fit(x_train, y_train)

y_pred = svc.predict(x_test)

acc_svc = round(svc.score(x_train, y_train) * 100, 2)

acc_svc
#knn

knn = KNeighborsClassifier(n_neighbors = 9)

knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

acc_knn = round(knn.score(x_train, y_train) * 100, 2)

acc_knn

# n 값을 여러가지로 바꿔봤는데 9일때가 가장컷으므로 9값으로 정했다.
# Decision Tree



decision_tree = DecisionTreeClassifier()

decision_tree.fit(x_train, y_train)

y_pred = decision_tree.predict(x_test)

acc_decision_tree = round(decision_tree.score(x_train, y_train) * 100, 2)

acc_decision_tree
# Random Forest



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(x_train, y_train)

y_pred = random_forest.predict(x_test)

random_forest.score(x_train, y_train)

acc_random_forest = round(random_forest.score(x_train, y_train) * 100, 2)

acc_random_forest
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN',  

              'Random Forest',   

              'Decision Tree'],

    'Score': [acc_svc, acc_knn, 

              acc_random_forest, 

              acc_decision_tree]})

models.sort_values(by='Score', ascending=False)
# 파일을 제출 했을때 가장큰 점수가 나온 분석기법은 random_forest가 아닌 svm 이였으므로 svm의 예측치를 제출한다.

#( 아마도 오버피팅의 문제 때문에 이런 것 같다.) 

svc = SVC()

svc.fit(x_train, y_train)

y_pred = svc.predict(x_test)
submission = pd.read_csv("../input/gender_submission.csv", index_col="PassengerId")

submission["Survived"] = y_pred

submission.to_csv("submit_team5.csv",index = False)