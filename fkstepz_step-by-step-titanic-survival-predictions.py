import pandas as pd


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train.head()
test.head() #test data set don't have survived col because it's target column
train.shape
test.shape #11cols  = survived is not included in test data set
train.info() #Nan in Age, Cabin columns
test.info()
train.isnull().sum() #you can check Nan value also like this 
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set() #setting seaborn default for plots - 디폴트 값으로 설정
def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts() #count survived people
    dead = train[train['Survived']==0][feature].value_counts() #count dead people
    df = pd.DataFrame([survived,dead]) #데이터프레임으로 묶고
    df.index = ['Survived','Dead'] #add index
    df.plot(kind='bar',stacked=True, figsize=(10,5)) #차트그리기 
bar_chart('Sex') #make polt based on sex
bar_chart('Pclass') 
bar_chart('SibSp') 
bar_chart('Embarked') 
train.head()
train_test_data = [train, test] # train과 test set 합침
train_test_data #train(891 rows) + test(418 rows)가 합쳐진것 확인
for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand = False)
    
train['Title'].value_counts()
test['Title'].value_counts()
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    
    #위 Title Dictionary에 맞게 숫자를 mapping 해준다. 숫자로 바꾸는 이유는 아까 말했던 것처럼
    #대부분 머신러닝 알고리즘들은 텍스트를 읽지 못하기 때문 
train.head()
test.head()
train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)
train.head()
test.head()
bar_chart('Title')
sex_mapping = {"male": 0, "female": 1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)
bar_chart('Sex')
train.head(30)
train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)

#train의 Age 칼럼의 nan값을 train의 title로 gourp을 지어서 해당 그룹의 age칼럼의 median값으로 대체하겠다.
#0 = Mr, 1 = Mrs, 2 = Miss, 3 = Others

test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)
#test 값을 ~ 위와동일

train.head(20)
for dataset in train_test_data:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0,
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1,
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2,
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3,
    dataset.loc[ dataset['Age'] > 62, 'Age'] = 4
train.head()
bar_chart('Age')
Pclass1 = train[train['Pclass']==1]['Embarked'].value_counts()
Pclass2 = train[train['Pclass']==2]['Embarked'].value_counts()
Pclass3 = train[train['Pclass']==3]['Embarked'].value_counts()

#Embarked 칼럼에서 Pclass가 1인 인스턴스의 갯수를 카운트하여 Pclass1 변수에 담는다
#2, 3도 반복

df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))
for dataset in train_test_data: #fill Nan value with 'S'
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
train.head()
embarked_mapping = {"S": 0, "C": 1, "Q": 2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)
train.head() #mapping embaked value with number(non-ordered)
# fill Nan with median value of the group(Pclass)
train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace=True)
test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)

train.head(10)
for dataset in train_test_data:
    dataset.loc[ dataset['Fare'] <= 17, 'Fare'] = 0,
    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1,
    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2,
    dataset.loc[ dataset['Fare'] > 100, 'Fare'] = 3
train.head(20)
train.Cabin.value_counts()
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]
Pclass1 = train[train['Pclass']==1]['Cabin'].value_counts() #Pclass=1에 해당하는 Cabin 값을 카운트
Pclass2 = train[train['Pclass']==2]['Cabin'].value_counts() #반복
Pclass3 = train[train['Pclass']==3]['Cabin'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))
cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping) #mapping cabin col
#fill Nan in cabin col with median value of the Pclass
train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1 #sib = sibling, Parch = parents and child
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1 

train["FamilySize"].max()
test["FamilySize"].max()
#Scale of familysize is 1 - 11 

family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}
for dataset in train_test_data:
    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)
train.head(20)
#drop ticket, sibsp, parch
features_drop = ['Ticket', 'SibSp', 'Parch']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId'], axis=1) #인덱스 필요없음 
train.head(20)
train_data = train.drop('Survived', axis=1)
target = train['Survived']

train_data.shape, target.shape #using survived col as target
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import numpy as np
train.info()
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0) #10개의 fold로 나눈다

clf = KNeighborsClassifier(n_neighbors = 13) #13개의 이웃을 기준으로 측정 
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score) #교차 검증 스코어 
# kNN Score
round(np.mean(score)*100, 2) #10번 시행시 평균 정확도
clf = DecisionTreeClassifier()

clf #특별하게 매개변수를 건드리지 않았으므로 다 디폴트 값이 주어짐
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score) #아까와 동일 
# decision tree Score
round(np.mean(score)*100, 2)
clf = RandomForestClassifier(n_estimators=13) #13개의 decision tree사용
clf
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
# decision tree Score
round(np.mean(score)*100, 2)
clf = GaussianNB()
clf
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
# Naive Bayes Score
round(np.mean(score)*100, 2)
clf = SVC()
clf
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
round(np.mean(score)*100,2)
#SVM이 제일 정확도가 높았으므로 SVM 사용

clf = SVC(C=1, kernel='rbf', coef0=1)
clf.fit(train_data, target) #학습시킬 데이터, 예측해야하는 타겟칼럼

test_data = test.drop("PassengerId", axis=1).copy()
prediction = clf.predict(test_data) 

#drop passengerid col
prediction #result of prediction
import collections, numpy

collections.Counter(prediction)
#dead = 257명 / survived = 161명 
