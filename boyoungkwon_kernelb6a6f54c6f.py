# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
os.chdir('../input')
# Any results you write to the current directory are saved as output.
# 한 셀의 결과 모두 출력
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train.head(20)
test.head(10)
train.shape
test.shape # 컬럼 한개가 적다 = survived
train.info
test.info
test.info()
train.info()
train.isnull().sum() # 이렇게 NaN(null) 값을 표시할 수도 있다
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set() #setting seaborn default for plots - 디폴트 값으로 설정
def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts() #feature에 따라 생존한 value(사람) 카운트
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived, dead]) # 데이터프레임으로 묶고
    df.index = ['Survived','Dead'] # 인덱스 달아주고
    df.plot(kind='bar', stacked=True, figsize=(10,5)) # 차트그리기
bar_chart('Sex')
bar_chart('Pclass')
bar_chart('SibSp')
bar_chart('Embarked') # 승선한 선착장에 따라서 죽었는지 살았는지
train.head(5)
train_test_data = [train, test] # train과 test set 합침
train_test_data #train(891 rows) + test(418 rows) 합쳐진 것 확인
for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand = False)
train['Title'].value_counts()
test['Title'].value_counts()
title_mapping = {"Mr":0, "Mrs":2,"Miss":1,"Master":3,"Dr":3,"Rev":3,"Col":3,"Major":3,"Mlle":3,"Countess":3,"Ms":3,"Lady":3,"Jonkheer":3,"Don":3,"Dona":3,"Mme":3,"Capt":3,"Sir":3}

for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)

    # 위 title dictionary에 맞게 숫자를 mapping해준다. 숫자로 바꾸는 이유는 아까 말했던 것처럼 대부분 머신러닝 알고리즘들은 텍스트를 읽지 못하기 때문.
train.head(5)
test.head(10)
test.info()
train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)
train.head(5)
bar_chart('Title')
sex_mapping = {"male":0, "female":1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)
bar_chart('Sex')
train.head()
train['Age'].fillna(train.groupby('Title')['Age'].transform('median'), inplace=True)

# train의 age 칼럽의 nan값을 train의 title로 group을 지어서 해당 그룹의 age 칼럼의 median값으로 대체한다.
# 0 = Mr, 1 = Mrs, 2 = Miss, 3 = Others
test['Age'].fillna(test.groupby('Title')['Age'].transform('median'), inplace=True)
train.head()
test.info()
for dataset in train_test_data:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0,
    dataset.loc[ (dataset['Age'] > 16)&(dataset['Age'] <= 26), 'Age'] = 1,
    dataset.loc[ (dataset['Age'] > 26)&(dataset['Age'] <= 36), 'Age'] = 2,
    dataset.loc[ (dataset['Age'] > 36)&(dataset['Age'] <= 62), 'Age'] = 3,
    dataset.loc[ (dataset['Age'] > 62), 'Age'] = 4
train.head()
train_test_data
bar_chart('Age')
Pclass1 = train[train['Pclass']==1]['Embarked'].value_counts()
Pclass2 = train[train['Pclass']==2]['Embarked'].value_counts()
Pclass3 = train[train['Pclass']==3]['Embarked'].value_counts()

# Embarked 칼럼에서 Pclass가 1인 인스턴스의 갯수를 카운트하여 Pclass1변수에 담는다.
# 2,3도 반복
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class','3rd class']
df.plot(kind='bar', stacked=True, figsize =(10,5))
# 결과를 볼 때, 전체 탑승객 중 s의 비율이 높기 때문에 Embarked가 Nan이면 그냥 s라고 봐도 무방하다고 가정할 수 있다.

for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
train.head(10)
embarked_mapping ={"S":0,"C":1,"Q":2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)
train.head(5)
test[test['Fare'].isnull()==True]
# NaN인 인스턴스가 속한 Pclass의 median값을 해당 결측치를 가진 인스턴스에 넣어준다. 아까 Age랑 동일
test['Fare'].fillna(test.groupby('Pclass')['Fare'].transform('median'))
train['Fare'].fillna(train.groupby('Pclass')['Fare'].transform('median'))
test.groupby('Pclass')['Fare'].transform('median').head()
test.head()
test.iloc[150:155,:]
test[test['Fare'].isnull()==True]
test.loc[[test['PassengerId']==1044],: ]['Fare']= test[test['Pclass']==3]['Fare'].mean()
test.head()
test.info()
train.info()
for dataset in train_test_data:
    dataset.loc[ dataset['Fare']<=17, 'Fare']=0,
    dataset.loc[(dataset['Fare']>17)&(dataset['Fare']<=30), 'Fare']=1,
    dataset.loc[(dataset['Fare']>30)&(dataset['Fare']<=100), 'Fare']=2,
    dataset.loc[(dataset['Fare']>100), 'Fare']=3
train.head()
train.Cabin.value_counts()
for dataset in train_test_data:
    dataset['Cabin']=dataset['Cabin'].str[:1]
Pclass1 = train[train['Pclass']==1]['Cabin'].value_counts()
Pclass2 = train[train['Pclass']==2]['Cabin'].value_counts()
Pclass3 = train[train['Pclass']==3]['Cabin'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class','3rd class']
df.plot(kind='bar', stacked=True, figsize=(10,5))
cabin_mapping = {'A':0, 'B':0.4,'C':0.8,'D':1.2,'E':1.6,'F':2,'G':2.4,'T':2.8}
for dataset in train_test_data:
    dataset['Cabin']= dataset['Cabin'].map(cabin_mapping)
#Pclass의 median으로 Cabin 결측치 대체
train['Cabin'].fillna(train.groupby('Pclass')['Cabin'].transform('median'), inplace=True)
test['Cabin'].fillna(test.groupby('Pclass')['Cabin'].transform('median'), inplace=True)
train['Familysize'] = train['SibSp']+train['Parch']+1 # sib = 형제자매, parch = 부모자식
test['Familysize'] = test['SibSp'] + test['Parch']+1 # 즉 형제자매 수 + 부모자식 수 + 나 = 우리가족수
train['Familysize'].max()
test['Familysize'].max()
#Familysize 의 범위는 1~11이다. 따라서 위에서 설명한 방식으로 정규화를 해준다.
family_mapping = {1:0, 2:0.4, 3:0.8, 4:1.2, 5:1.6, 6:2, 7:2.4, 8:2.8, 9:3.2, 10:3.6, 11:4}
for dataset in train_test_data:
    dataset['Familysize']=dataset['Familysize'].map(family_mapping)
train.head()
test.head()
# 티켓번호, 형제자매수, 부모가족수 칼럼은 드랍하도록 한다
features_drop=['Ticket','SibSp','Parch']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId'], axis = 1) # 인덱스 필요없음
train.head()
train_data = train.drop('Survived', axis=1)
target = train['Survived']

train_data.shape, target.shape # survived를 떼서 target값으로 준다
target.head()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
train.info()
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
# 10 개의 fold로 나눈다.
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
clf = KNeighborsClassifier(n_neighbors = 13) #13개의 이웃을 기준으로 측정
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring = scoring)
print(score) # 교차 검증 스코어
# kNN Score
round(np.mean(score)*100,2) # 10번 시행시 평균 정확도
clf = DecisionTreeClassifier()
clf # 특별하게 매개변수를 건드리지 않았으므로 다 디폴트 값이 주어짐
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv = k_fold, n_jobs=1, scoring = scoring)
print(score) # 아까와 동일
round(np.mean(score)*100,2)
clf = RandomForestClassifier(n_estimators=13) #13개의 decision tree 사용
clf
scoring = 'accuracy'
score = cross_val_score(clf, train_data,target, cv=k_fold, n_jobs=1, scoring = scoring)
print(score)
# decision tree Score
round(np.mean(score)*100,2)
