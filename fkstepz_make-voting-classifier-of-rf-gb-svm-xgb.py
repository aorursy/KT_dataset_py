import pandas as pd

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train.head(5)
test.head(5) #survived 칼럼만 없다. 왜냐? 타겟이기 때문(종속변수 = 예측해야하는 것이기 때문)
train.shape
test.shape #칼럼 한개가 적다 = survived
train.info() #총 891개가 있어야 결측값(NaN) 없는 것, 그러나 Age, Cabin같은경우 NaN이 많다 
test.info()
train.isnull().sum() #이렇게 NaN(Null) 값을 표시할 수도 있다
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set() #setting seaborn default for plots - 디폴트 값으로 설정
def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts() #feature에 따라 생존한 value(사람) 카운트
    dead = train[train['Survived']==0][feature].value_counts() #feature에 따라 죽은 value(사람) 카운트 
    df = pd.DataFrame([survived,dead]) #데이터프레임으로 묶고
    df.index = ['Survived','Dead'] #인덱스 달아주고 
    df.plot(kind='bar',stacked=True, figsize=(10,5)) #차트그리기 
bar_chart('Sex') #성별에 따라서 죽었는지 살았는지 
bar_chart('Pclass') #클래스에 따라서 죽었는지 살았는지 - 위 Data dictionary 참조
bar_chart('SibSp') #가족수에 따라서 죽었는지 살았는지 - 위 Data dictionary 참조
bar_chart('Embarked') #승선한 선착장에 따라서 죽었는지 살았는지 - 위 Data dictionary 참조
train.head(5)
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
train.head(5)
test.head(5)
train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)
train.head(5)
test.head(5)
bar_chart('Title')
sex_mapping = {"male": 0, "female": 1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)
bar_chart('Sex')
train.head(5)
train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)

#train의 Age 칼럼의 nan값을 train의 title로 gourp을 지어서 해당 그룹의 age칼럼의 median값으로 대체하겠다.
#0 = Mr, 1 = Mrs, 2 = Miss, 3 = Others

test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)
#test 값을 ~ 위와동일F

train.head(5)
for dataset in train_test_data:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0,
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1,
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2,
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3,
    dataset.loc[ dataset['Age'] > 62, 'Age'] = 4
train.head(5)
bar_chart('Age')
Pclass1 = train[train['Pclass']==1]['Embarked'].value_counts()
Pclass2 = train[train['Pclass']==2]['Embarked'].value_counts()
Pclass3 = train[train['Pclass']==3]['Embarked'].value_counts()

#Embarked 칼럼에서 Pclass가 1인 인스턴스의 갯수를 카운트하여 Pclass1 변수에 담는다
#2, 3도 반복

df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
train.head(5)
embarked_mapping = {"S": 0, "C": 1, "Q": 2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)
train.head(5) #마찬가지로 text는 못읽으니 숫자로 매핑해준다
# Nan인 인스턴스가 속한 Pclass의 median값을 해당 결측지를 가진 인스턴스에 넣어준다. 아까 Age랑 동일
train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace=True)
test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)

train.head(5)
for dataset in train_test_data:
    dataset.loc[ dataset['Fare'] <= 17, 'Fare'] = 0,
    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1,
    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2,
    dataset.loc[ dataset['Fare'] > 100, 'Fare'] = 3
train.head(5)
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
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)
#Pclass의 median으로 Cabin 결측치 대체
train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1 #sib = 형제자매, Parch = 부모자식
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1 #즉 형제자매 수 + 부모자식수 + 나 = 우리가족수 

train["FamilySize"].max()
test["FamilySize"].max()
#FamilySize의 범위는 1~11이다. 따라서 위에서 설명한 비닝 방식으로 정규화를 해준다

family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}
for dataset in train_test_data:
    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)
train.head(5)
#티켓번호, 형제자매수, 부모가족수 칼럼은 드랍하도록 한다
features_drop = ['Ticket', 'SibSp', 'Parch']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId'], axis=1) #인덱스 필요없음 
train.head(5)
#train.to_csv('train_dropnulll.csv', index=False)
train_data = train.drop('Survived', axis=1)
target = train['Survived']

train_data.shape, target.shape #survived를 때서 target값으로 준다 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import numpy as np
train.info()
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import xgboost as xgb
#use 10 fold cross validation
k_fold = KFold(n_splits=10, shuffle=True, random_state=0) #10개의 fold로 나눈다

RFC = RandomForestClassifier()


## Search grid for optimal parameters
rf_param_grid = {"max_depth": [None],
              "max_features": [3, 8, 8],
              "min_samples_split": [2, 3, 8],
              "min_samples_leaf": [1, 3, 8],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}


gsRFC = GridSearchCV(RFC, rf_param_grid, cv=k_fold, scoring="accuracy",  verbose = 1)
#print(score)

gsRFC.fit(train_data,target)

RFC_best = gsRFC.best_estimator_

# Best score
gsRFC.best_score_
 

GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] 
              }

gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=k_fold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsGBC.fit(train_data,target)

GBC_best = gsGBC.best_estimator_

# Best score
gsGBC.best_score_
### SVC classifier
SVMC = SVC(probability=True)
svc_param_grid = {'kernel': ['rbf'], 
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [1, 10, 50, 100,200,300, 1000]}

gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=k_fold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsSVMC.fit(train_data,target)

SVMC_best = gsSVMC.best_estimator_

# Best score
gsSVMC.best_score_
gsSVMC.best_estimator_
XGBC = XGBClassifier()
xgb_param_grid = {'max_depth':[3,5,7],
                  'min_child_weight':[3,5,6],
                  'gamma': [ 0, 0.001, 0.01, 0.1, 1],
                  'learning_rate':[0.1, 0.05, 0.01]}

gsXGBC = GridSearchCV(XGBC,param_grid = xgb_param_grid, cv=k_fold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsXGBC.fit(train_data,target)

XGBC_best = gsXGBC.best_estimator_

# Best score
gsXGBC.best_score_



votingC = VotingClassifier(estimators=[('rfc', RFC_best), 
('svc', SVMC_best),('gbc',GBC_best), ('xgb', XGBC_best)], voting='hard', n_jobs=4)

votingC = votingC.fit(train_data, target)
votingC.predict
test_data = test.drop("PassengerId", axis=1).copy()
prediction = votingC.predict(test_data) 
#케글에 제출할 csv파일 저장
#submission = pd.DataFrame({
#        "PassengerId": test["PassengerId"],
#        "Survived": prediction
#    })

#submission.to_csv('submission.csv', index=False)
