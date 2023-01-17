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


import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# Common Tools

from sklearn.preprocessing import LabelEncoder

from collections import Counter



#Algorithms

from sklearn import ensemble, tree, svm, naive_bayes, neighbors, linear_model, gaussian_process, neural_network

import xgboost as xgb

from xgboost.sklearn import XGBClassifier



# Model

from sklearn.metrics import accuracy_score, f1_score, auc, roc_curve, roc_auc_score

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score

#from sklearn.ensemble import VotingClassifier



#Configure Defaults

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
#나이에서 nan값을 가진 사람들의 특징을 보자

age_nan_rows = train[train['Age'].isnull()]



age_nan_rows.head()
#성별처리 : 성별을 남자는1, 여자는 0으로 입력

from sklearn.preprocessing import LabelEncoder

train['Sex'] = LabelEncoder().fit_transform(train['Sex'])

test['Sex'] = LabelEncoder().fit_transform(test['Sex'])



train.head(10)
train['Name'] = train['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())

train_titles = train['Name'].unique()

train_titles

test['Name'] = test['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())

test_titles = test['Name'].unique()

test_titles
train_titles
test_titles

train['Age'].fillna(-1, inplace=True)

test['Age'].fillna(-1, inplace=True)



medians = dict()

for title in train_titles:

    median = train.Age[(train["Age"] != -1) & (train['Name'] == title)].median()

    medians[title] = median





for index, row in train.iterrows():

    if row['Age'] == -1:

        train.loc[index, 'Age'] = medians[row['Name']]



for index, row in test.iterrows():

    if row['Age'] == -1:

        test.loc[index, 'Age'] = medians[row['Name']]



train.head(12)

    
#이름별로 산 사람과 죽은 사람을 비교



fig = plt.figure(figsize=(15,6))



i=1

for title in train['Name'].unique():

    fig.add_subplot(3, 6, i)

    plt.title('Title : {}'.format(title))

    train.Survived[train['Name'] == title].value_counts().plot(kind='pie')

    i += 1

#각 성 별로 많이 죽은 성에서 적게 죽은 성까지 순서를 매기면 아래와 같다. 직접 센다

title_replace = {'Don':0, 'Rev':0, 'Capt':0,'Jonkheer':0,'Mr':1,'Dr':2,'Major':3, 'Col':3,

    'Master':4,'Miss':5,'Mrs':6,'Mme':7,'Ms':7,'Lady':7,'Sir':7,'Mlle':7,'the Countess':7}



train['Name'].unique()
test['Name'].unique()
#현재 Dona라는 성이 test에만 있다. 이 사람에 대한 정보를 보면

test[test['Name'] == 'Dona']
train['Name'] = train['Name'].apply(lambda x: title_replace.get(x))

train.head()
test['Name'] = test['Name'].apply(lambda x: title_replace.get(x))

test.isnull().sum()
test[test['Name'].isnull()]
test[test['Sex'] == 0]['Name'].mean()
train[train['Sex'] == 0]['Name'].mean()
test[test['Name'].isnull()]['Sex']
test[test['Name'].isnull()]['Name']
test['Name'] = test['Name'].fillna(value=train[train['Sex'] == 0]['Name'].mean())

test.head()
train_and_test_data = [train,test]
for dataset in train_and_test_data :

    dataset.loc[ dataset['Age']<=10, 'Age'] = 0,

    dataset.loc[(dataset['Age']>10)&(dataset['Age']<=16), 'Age'] = 1,

    dataset.loc[(dataset['Age']>16)&(dataset['Age']<=20), 'Age'] = 2,

    dataset.loc[(dataset['Age']>20)&(dataset['Age']<=26), 'Age'] = 3,

    dataset.loc[(dataset['Age']>26)&(dataset['Age']<=30), 'Age'] = 4,

    dataset.loc[(dataset['Age']>30)&(dataset['Age']<=36), 'Age'] = 5,

    dataset.loc[(dataset['Age']>36)&(dataset['Age']<=40), 'Age'] = 6,

    dataset.loc[(dataset['Age']>40)&(dataset['Age']<=46), 'Age'] = 7,

    dataset.loc[(dataset['Age']>46)&(dataset['Age']<=50), 'Age'] = 8,

    dataset.loc[(dataset['Age']>50)&(dataset['Age']<=60), 'Age'] = 9,

    dataset.loc[ dataset['Age']>60, 'Age'] = 10

#나눈 나이에 대해 죽은사람과 산 사람의 비율을 확인



fig = plt.figure(figsize=(16,5))



i=1

for age in train['Age'].unique():

    fig.add_subplot(3, 6, i)

    plt.title('Age : {}'.format(age))

    train.Survived[train['Age'] == age].value_counts().plot(kind='pie')

    i += 1

    

#이름과 동일한 방법으로 가공한다

age_point_replace = {    0: 8,    1: 6,    2: 2,    3: 4,    4: 1,    5: 7,    6: 3,

    7: 2,    8: 5,    9: 4,    10: 0}   

for dataset in train_and_test_data:

    dataset['age_point'] = dataset['Age'].apply(lambda x: age_point_replace.get(x))

    

train.head(10)
#embarked가 nan인 사람들은 c로 입력, 각각의 항구 이름을 숫자로 표기한다

for dataset in train_and_test_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('C')

    

embarked_mapping = {'S':0, 'C':1, 'Q':2}

for dataset in train_and_test_data:

    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)



train.head()
#가족의 크기를 설정하고, 해당값이 큰 

for dataset in train_and_test_data:

    dataset['size_of_family'] = dataset['SibSp'] + dataset['Parch'] + 1



#타이타닉 침몰1910~1920년대 당시 유럽의 평균 가족크기는 여성 1명당 아이 2명이었으므로, 

#해당하는 가족크기를 가진 남자는 아버지일 가능성이 클 것



devoted_father_mask = (train['size_of_family'] > 3 ) & (train['Sex'] == 1)



devoted_father_mask.head(10)

train['devoted_father'] = 1

train.head(10)

train.loc[devoted_father_mask, 'devoted_father'] = 0 

train[train['devoted_father'] == 0].head()
fig = plt.figure(figsize = (15,5))

i=1

for title in train['Name'].unique() :

    fig.add_subplot(3,6,i)

    plt.title('Title : {}'.format(title))

    train.Survived[train['Name'] == title].value_counts().plot(kind='pie')

    i += 1
#아버지로 분류된 사람들의 생존률. 아래 값보다 산 사람이 더 많다

no1 = train.Survived[train['devoted_father'] == 1].value_counts().plot(kind='pie')
#아버지가 아닌 사람의 생존률. 죽은 사람이 더 많다

no2 = train.Survived[train['devoted_father'] == 0].value_counts().plot(kind='pie')
#유의미한 분류가 나오는 것 같으므로 size of family를 4이상으로 조정하여 테스트해보자

devoted_father_mask = (train['size_of_family'] > 5) & (train['Sex'] == 1)

devoted_father_mask.head(10)
train['devoted_father'] = 1

train.head(10)

train.loc[devoted_father_mask, 'devoted_father'] = 0 

train[train['devoted_father'] == 0].head()
fig = plt.figure(figsize = (15,5))

i=1

for title in train['Name'].unique() :

    fig.add_subplot(3,6,i)

    plt.title('Title : {}'.format(title))

    train.Survived[train['Name'] == title].value_counts().plot(kind='pie')

    i += 1
#아버지로 분류된 사람들의 생존률. 아래 값보다 산 사람이 더 많다

no1 = train.Survived[train['devoted_father'] == 1].value_counts().plot(kind='pie')
#아버지가 아닌 사람의 생존률. 죽은 사람이 더 많고 family size를 조정하지 않았을 때보다 더 deviation이 줄어들었다

no2 = train.Survived[train['devoted_father'] == 0].value_counts().plot(kind='pie')
#test file에도 해당 카테고리를 삽입한다 



test['devoted_father'] = 1

test_devoted_father_mask = (test['size_of_family']>4) & (test['Sex'] == 1)

test.loc[test_devoted_father_mask, 'devoted_father'] = 0 



test.head()
train['size_of_family'].unique()
test['size_of_family'].unique()
fig = plt.figure(figsize=(15,6))



i=1

for size in train['size_of_family'].unique():

    fig.add_subplot(3, 6, i)

    plt.title('Size : {}'.format(size))

    train.Survived[train['size_of_family'] == size].value_counts().plot(kind='pie')

    i += 1
#size of family 역시로 age 등과 같이 순서를 매겨보자

size_replace = {1:3,2:4,3:6,4:7,5:2,6:1,7:4,8:0,11:0}

for dataset in train_and_test_data:

    dataset['fs_point'] = dataset['size_of_family'].apply(lambda x: size_replace.get(x))

    dataset.drop('size_of_family',axis=1,inplace=True)
train.head(10)
train.isnull().sum()
test.isnull().sum()
#pclass에 대한 가공, 생존비율 확인하고 위와 같은 작업을 process해보자

fig = plt.figure(figsize=(15,5))



i=1

for x in train['Pclass'].unique():

    fig.add_subplot(3, 6, i)

    plt.title('Pclass : {}'.format(x))

    train.Survived[train['Pclass'] == x].value_counts().plot(kind='pie')

    i += 1

for dataset in train_and_test_data:

    dataset.loc[dataset['Pclass']==3,'Pclass_point'] = 0

    dataset.loc[dataset['Pclass']==2,'Pclass_point'] = 1

    dataset.loc[dataset['Pclass']==1,'Pclass_point'] = 2
#embarked에 대해서도 동일하게 진행한다

fig = plt.figure(figsize=(15,6))



i=1

for x in train['Embarked'].unique():

    fig.add_subplot(3, 6, i)

    plt.title('Embarked : {}'.format(x))

    train.Survived[train['Embarked'] == x].value_counts().plot(kind='pie')

    i += 1
for dataset in train_and_test_data:

    dataset.loc[dataset['Embarked']==0,'Em_point'] = 0

    dataset.loc[dataset['Embarked']==2,'Em_point'] = 1

    dataset.loc[dataset['Embarked']==1,'Em_point'] = 2
#cabin의 데이터 처리를 위해 내역을 보았다



train['Cabin'].unique()
#cabin값이 nan인 사람들을 N으로 설정하고 cabin별로 생존확률을 확인하면

for data in train_and_test_data:

    data['Cabin'].fillna('N', inplace=True)

    data['Cabin'] = data['Cabin'].apply(lambda x: x[0])

    data['Cabin'].unique()

    data['Fare'].fillna(0,inplace=True)

    data['Fare'] = data['Fare'].apply(lambda x: int(x))
fig = plt.figure(figsize=(15,5))

i=1

for x in train['Cabin'].unique():

    fig.add_subplot(3, 6, i)

    plt.title('Cabin : {}'.format(x))

    train.Survived[train['Cabin'] == x].value_counts().plot(kind='pie')

    i += 1
#fare값을 가공해보자



processing_fare = train['Fare'].unique()

processing_fare.sort()

processing_fare
#fare를 7개의 구간으로 나누고 각각에게 포인트를 줬다



for dataset in train_and_test_data:

    dataset.loc[ dataset['Fare']<=30, 'Fare'] = 0,

    dataset.loc[(dataset['Fare']>30)&(dataset['Fare']<=60), 'Fare'] = 1,

    dataset.loc[(dataset['Fare']>60)&(dataset['Fare']<=90), 'Fare'] = 2,

    dataset.loc[(dataset['Fare']>90)&(dataset['Fare']<=120), 'Fare'] = 3,

    dataset.loc[(dataset['Fare']>120)&(dataset['Fare']<=150), 'Fare'] = 4,

    dataset.loc[(dataset['Fare']>150)&(dataset['Fare']<=180), 'Fare'] = 5,

    dataset.loc[(dataset['Fare']>=180), 'Fare'] = 6
fig = plt.figure(figsize=(15,6))



i=1

for x in train['Cabin'].unique():

    fig.add_subplot(3, 6, i)

    plt.title('Cabin : {}'.format(x))

    train.Fare[train['Cabin'] == x].value_counts().plot(kind='pie')

    i += 1
#N의 값을 가지는 cabin의 사람들에게 존재하는 cabin값으로 넣어주기 위해 각 cabin별로 가장 넓은 범위의 fare양을 가지는 

#fare의 cabin값으로 N을 대체

for dataset in train_and_test_data:

    dataset.loc[(dataset['Cabin'] == 'N')&(dataset['Fare'] == 0), 'Cabin'] = 'G',

    dataset.loc[(dataset['Cabin'] == 'N')&(dataset['Fare'] == 1), 'Cabin'] = 'T',

    dataset.loc[(dataset['Cabin'] == 'N')&(dataset['Fare'] == 2), 'Cabin'] = 'D',

    dataset.loc[(dataset['Cabin'] == 'N')&(dataset['Fare'] == 3), 'Cabin'] = 'B',

    dataset.loc[(dataset['Cabin'] == 'N')&(dataset['Fare'] == 4), 'Cabin'] = 'E',

    dataset.loc[(dataset['Cabin'] == 'N')&(dataset['Fare'] == 5), 'Cabin'] = 'C',

    dataset.loc[(dataset['Cabin'] == 'N')&(dataset['Fare'] == 6), 'Cabin'] = 'B'

    

    
fig = plt.figure(figsize=(15,5))



i=1

for x in train['Cabin'].unique():

    fig.add_subplot(3, 6, i)

    plt.title('Cabin : {}'.format(x))

    train.Fare[train['Cabin'] == x].value_counts().plot(kind='pie')

    i += 1
#cabin별 생존 확률

fig = plt.figure(figsize=(15,5))



i=1

for x in train['Cabin'].unique():

    fig.add_subplot(3, 6, i)

    plt.title('Cabin : {}'.format(x))

    train.Survived[train['Cabin'] == x].value_counts().plot(kind='pie')

    i += 1
for dataset in train_and_test_data:

    dataset.loc[(dataset['Cabin'] == 'G'), 'Cabin_point'] = 0,

    dataset.loc[(dataset['Cabin'] == 'C'), 'Cabin_point'] = 4,

    dataset.loc[(dataset['Cabin'] == 'E'), 'Cabin_point'] = 6,

    dataset.loc[(dataset['Cabin'] == 'T'), 'Cabin_point'] = 1,

    dataset.loc[(dataset['Cabin'] == 'D'), 'Cabin_point'] = 3,

    dataset.loc[(dataset['Cabin'] == 'A'), 'Cabin_point'] = 2,

    dataset.loc[(dataset['Cabin'] == 'B'), 'Cabin_point'] = 7,

    dataset.loc[(dataset['Cabin'] == 'F'), 'Cabin_point'] = 5,
test.head(30)
fig = plt.figure(figsize=(15,6)) 



i=1

for x in train['Fare'].unique():

    fig.add_subplot(3, 6, i)

    plt.title('Fare : {}'.format(x))

    train.Survived[train['Fare'] == x].value_counts().plot(kind='pie')

    i += 1
#fare에 따른 생존 비율을 바탕으로 score replacement

for dataset in train_and_test_data:

    dataset.loc[(dataset['Fare'] == 0), 'Fare_point'] = 0,

    dataset.loc[(dataset['Fare'] == 1), 'Fare_point'] = 2,

    dataset.loc[(dataset['Fare'] == 2), 'Fare_point'] = 1,

    dataset.loc[(dataset['Fare'] == 3), 'Fare_point'] = 5,

    dataset.loc[(dataset['Fare'] == 4), 'Fare_point'] = 4,

    dataset.loc[(dataset['Fare'] == 5), 'Fare_point'] = 6,

    dataset.loc[(dataset['Fare'] == 6), 'Fare_point'] = 3
train.drop(['PassengerId','Pclass','SibSp','Parch','Ticket','Fare','Embarked','Cabin','Age'], axis=1, inplace=True)

test.drop(['Pclass','SibSp','Parch','Ticket','Fare','Embarked','Cabin','Age'], axis=1, inplace=True)
train_data = train.drop('Survived', axis = 1)

target = train['Survived']
test.shape
train.shape
train_data.shape
# Importing Classifier Modules, svc 방법으로 추정하였다.

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score



clf= SVC()

clf.fit(train_data, target)



test_data = test.drop("PassengerId", axis=1).copy()

prediction = clf.predict(test_data)

submission = pd.DataFrame({"PassengerId": test["PassengerId"],"Survived": prediction })

 

submission.to_csv('submission_svc.csv', index=False)

submission = pd.read_csv('submission_svc.csv')

submission.head()


