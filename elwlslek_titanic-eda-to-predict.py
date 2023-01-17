import numpy as np

import pandas as pd

import os

import warnings

from IPython.display import Image

warnings.filterwarnings("ignore")
# 데이터 로드

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

total_data = [train, test]

# 학습, 테스트 데이터 병합 ( 전처리 위해 )
train.describe(include='all')
for column in ['Survived','Pclass','SibSp','Parch','Cabin','Embarked']:

    print(column)

    print(train[column].unique())
def check_missing_data(data):

    total = data.isnull().sum()

    percent = (data.isnull().sum() / data.isnull().count() * 100)

    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    types = []

    for col in data.columns:

        dtype = str(data[col].dtype)

        types.append(dtype)

    tt['Types'] = types

    return np.transpose(tt)
check_missing_data(train)

# Age, Cabin, Embarked 결측치 확인
check_missing_data(test)

# Age, Fare, Cabin 결측치 확인
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set() ## setting seaborn default for plots
def bar_chart(feature):

    survived = train[train['Survived']==1][feature].value_counts()

    dead = train[train['Survived']==0][feature].value_counts()

    df = pd.DataFrame([survived,dead])

    df.index = ['Survived','dead']

    df.plot(kind='bar',stacked=True,figsize=(10,5))
bar_chart("Sex")

# 여성이 남성보다 많이 살아남음
bar_chart("Pclass")

# 생존자에서는 차이가 없지만, 사망자는 대부분 3등칸, 2등칸, 1등칸 순으로 많다.
bar_chart("SibSp")
bar_chart("Parch")
bar_chart("Embarked")

# 선착장은 영향이 없는 것으로 보임
train['Name'].tolist()[:10]

# Name 컬럼으로부터 결혼여부, 성별, 확인 가능
# title_re 가 없는 경우에대한 처리가 가능

# for data in total_data:

#     data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.',expand=True)

import re

def get_title(name):

    title_re = re.search(" ([A-Za-z]+)\.", name)

    if title_re:

        return title_re.group(1)

    return ""
for data in total_data:

    data['Title'] = data['Name'].apply(get_title)

# Mr : man

# Miss : unmarried woman

# Mrs : married woman

# master : boy

# Ms : not indicate her marital status.
title_list = []

for data in total_data:

    title_list += data['Title'].value_counts().index.tolist()

title_list = list(set(title_list))

title_list
# Mr : 1 // Miss : 2 // Mrs : 3 // Master : 4 // Ohter : 5

# 비슷한 의미 맵핑 -> Mlle, Ms, Mme

non_common_titles = ['Rev','Major','Countess','Col','Capt','Lady','Don','Sir','Dr','Jonkheer','Dona']

for data in total_data:

    data['Title'] = data['Title'].replace(non_common_titles, 'Other')

    data['Title'] = data['Title'].replace("Mlle","Miss")

    data['Title'] = data['Title'].replace("Ms","Miss")

    data['Title'] = data['Title'].replace("Mme","Mrs")
title_mapping = {'Mr':1,'Miss':2,'Mrs':3,'Master':4,'Other':5}

for data in total_data:

    data['Title'] = data['Title'].map(title_mapping)

    # map 함수에 매핑되지 않는 row는 nan 처리하기 때문에

    data['Title'] = data['Title'].fillna(0)
for data in total_data:

    title_na_count = (data['Title'] == 0).isnull().sum()

    print(title_na_count)
bar_chart("Title")
for data in total_data:

    data['Sex'] = data['Sex'].map({'female':0,'male':1}).astype(int) # map 함수에 매핑되지 않는 row는 nan 처리
# 결측치 처리

for data in total_data:

    age_avg = data['Age'].mean()

    age_std = data['Age'].std()

    age_null_count = data['Age'].isnull().sum()

    # 표준편차는 평균으로 부터 원래 데이타에 대한 오차범위의 근사값 ( 분산의 제곱근 으로 근사한다. )

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    data['Age'][np.isnan(data['Age'])] = age_null_random_list

    data['Age'] = data['Age'].astype(int)
np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
train['Age'].isnull().sum()
test['Age'].isnull().sum()
facet = sns.FacetGrid(train,hue="Survived",aspect=4)

facet.map(sns.kdeplot,"Age",shade=True)

facet.set(xlim=(0, train['Age'].max()))

facet.add_legend()



plt.show()
facet = sns.FacetGrid(train,hue="Survived",aspect=2)

facet.map(sns.kdeplot,"Age",shade=True)

facet.set(xlim=(0, train['Age'].max()))

facet.add_legend()

plt.xlim(0,20)

plt.show()
facet = sns.FacetGrid(train,hue="Survived",aspect=2)

facet.map(sns.kdeplot,"Age",shade=True)

facet.set(xlim=(0, train['Age'].max()))

facet.add_legend()

plt.xlim(20,30)

plt.show()
facet = sns.FacetGrid(train,hue="Survived",aspect=2)

facet.map(sns.kdeplot,"Age",shade=True)

facet.set(xlim=(0, train['Age'].max()))

facet.add_legend()

plt.xlim(30,40)

plt.show()
facet = sns.FacetGrid(train,hue="Survived",aspect=3)

facet.map(sns.kdeplot,"Age",shade=True)

facet.set(xlim=(0, train['Age'].max()))

facet.add_legend()

plt.xlim(40,60)

plt.show()
facet = sns.FacetGrid(train,hue="Survived",aspect=2)

facet.map(sns.kdeplot,"Age",shade=True)

facet.set(xlim=(0, train['Age'].max()))

facet.add_legend()

plt.xlim(60,80)

plt.show()
# Age : continous

for data in total_data:

    data.loc[data['Age'] <= 11,"Age"] = 0

    data.loc[(data['Age'] > 11) & (data['Age'] <= 28),"Age"] = 1

    data.loc[(data['Age'] > 28) & (data['Age'] <= 38),"Age"] = 2

    data.loc[(data['Age'] > 38) & (data['Age'] <= 48),"Age"] = 3

    data.loc[(data['Age'] > 48) & (data['Age'] <= 60),"Age"] = 4

    data.loc[(data['Age'] > 60),"Age"] = 5
Pclass1 = train[train['Pclass']==1]['Embarked'].value_counts()

Pclass2 = train[train['Pclass']==2]['Embarked'].value_counts()

Pclass3 = train[train['Pclass']==3]['Embarked'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3],index=['1st class','2nd class','3rd class'])

df
df.plot(kind='bar',stacked=False,figsize=(10,5))

plt.show()
for data in total_data:

    data['Embarked'] = data['Embarked'].fillna("S")

    data['Embarked'] = data['Embarked'].map({"S":0,"C":1,"Q":1}).astype(int)
train['Fare'].isnull().sum(), test['Fare'].isnull().sum()
# 테스트 데이터에서 결측치 제거

test['Fare'].fillna(test.groupby("Pclass")['Fare'].transform("median"),inplace=True)
facet = sns.FacetGrid(train,hue="Survived",aspect=4)

facet.map(sns.kdeplot,"Fare",shade=True)

facet.set(xlim=(0,train['Fare'].max()))

facet.add_legend()



plt.show()
facet = sns.FacetGrid(train,hue="Survived",aspect=4)

facet.map(sns.kdeplot,"Fare",shade=True)

facet.set(xlim=(0,train['Fare'].max()))

facet.add_legend()



plt.xlim(0,40)

plt.show()
for data in total_data:

    data.loc[data['Fare']<=8,'Fare'] = 0,

    data.loc[(data['Fare']>8) & (data['Fare']<=17),'Fare'] = 1

    data.loc[(data['Fare']>17) & (data['Fare']<=35),'Fare'] = 2

    data.loc[data['Fare']>35,'Fare'] = 3
train['Cabin'].value_counts()
for data in total_data:

    data['Cabin'] = data['Cabin'].str[:1]
Pclass1 = train[train['Pclass'] == 1]['Cabin'].value_counts()

Pclass2 = train[train['Pclass'] == 2]['Cabin'].value_counts()

Pclass3 = train[train['Pclass'] == 3]['Cabin'].value_counts()
df = pd.DataFrame([Pclass1,Pclass2,Pclass3],index=['1st class','2nd class','3rd class'])

df
df.plot(kind="bar",stacked=False,figsize=(10,5))

plt.show()
cabin_mapping = {'A':0,'B':0.4,'C':0.8,'D':1.2,'E':1.6,'F':2.0,'G':2.4,'T':2.8}

for data in total_data:

    data['Cabin'] = data['Cabin'].map(cabin_mapping)
for data in total_data:

    data['Cabin'].fillna(value = data.groupby("Pclass")['Cabin'].transform("median"), inplace=True)
check_missing_data(train)
check_missing_data(test)
for data in total_data:

    data['FamilySize'] = data['SibSp'] + train['Parch'] + 1
facet = sns.FacetGrid(train, hue="Survived",aspect=4) 

facet.map(sns.kdeplot,'FamilySize',shade= True) 

facet.set(xlim=(0, train['FamilySize'].max())) 

facet.add_legend()

plt.xlim(0)

plt.show()
(train['FamilySize'] == 0).isnull().sum()
for data in total_data:

    data['IsAlone'] = 0

    data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1
train.head()
drop_elements = ['PassengerId', 'Name', 'Ticket', 'SibSp']

for data in total_data:

    data.drop(drop_elements,axis=1,inplace=True)
train_data = train.drop("Survived",axis=1)

label = train['Survived']

train_data.shape, label.shape
from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

k_fold = KFold(n_splits=10, shuffle=True, random_state=42)
model = KNeighborsClassifier(n_neighbors = 13)

score = cross_val_score(model, train_data, label, cv=k_fold, n_jobs=1, scoring='accuracy')

# kNN Score

round(np.mean(score)*100, 2)
Image(url= "https://tensorflowkorea.files.wordpress.com/2017/06/2-4.png?w=1250")
model = DecisionTreeClassifier()

score = cross_val_score(model, train_data, label, cv=k_fold, n_jobs=1, scoring='accuracy')

round(np.mean(score)*100, 2)
model = RandomForestClassifier(n_estimators=13)

score = cross_val_score(model, train_data, label, cv=k_fold, n_jobs=1, scoring='accuracy')

round(np.mean(score)*100, 2)
from sklearn.naive_bayes import GaussianNB



model = GaussianNB()

score = cross_val_score(model, train_data, label, cv=k_fold, n_jobs=1, scoring='accuracy')

round(np.mean(score)*100, 2)
from sklearn.naive_bayes import BernoulliNB



model = BernoulliNB()

score = cross_val_score(model, train_data, label, cv=k_fold, n_jobs=1, scoring='accuracy')

round(np.mean(score)*100, 2)
from sklearn.naive_bayes import MultinomialNB



model = MultinomialNB()

score = cross_val_score(model, train_data, label, cv=k_fold, n_jobs=1, scoring='accuracy')

round(np.mean(score)*100, 2)
model = SVC()

score = cross_val_score(model, train_data, label, cv=k_fold, n_jobs=1, scoring='accuracy')

round(np.mean(score)*100, 2)
model = SVC()

model.fit(train_data, label)

prediction = model.predict(test)
test = pd.read_csv("../input/test.csv")
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": prediction

    })



submission.to_csv('submission.csv', index=False)
submission = pd.read_csv('submission.csv')

submission.head()