# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()

# Modeling을 하기 위해 필요한 library import

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score
# Data preprocessing



X_train = pd.DataFrame(train_data,columns = train_data.columns)

X_test = pd.DataFrame(test_data,columns = test_data.columns)



y_train = train_data['Survived']



drop_features = ['PassengerId','Cabin','Ticket'] # 안 쓸 데이터



X_train = X_train.drop(drop_features+['Survived'],axis = 1)

X_test = X_test.drop(drop_features,axis = 1)



train_test = [X_train,X_test]



# Sex data와 Embarked데이터의 문자열을 숫자열로 바꾸어준다.

for dataset in train_test:

    sex_mapping = {'male':0, 'female':1}

    dataset['Sex'] = dataset['Sex'].map(sex_mapping).astype(int)



for dataset in train_test:

    embarked_mapping = {'S':0,'C':1,'Q':2}

    dataset['Embarked'].fillna('S', inplace = True) # 비어있는 데이터를 최대값인 S로 넣는다.

    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping).astype(int)

# Age data의 분포

sns.distplot(X_train['Age'])
# Fare data의 분포

sns.distplot(X_test['Fare'])
# 결측값이 있는 age데이터와 fare 데이터를 보고 결측값을 판단한다.



# 비어있는 데이터를 채운다. age에는 평균값을 넣어주고

# fare에는 데이터의 최소값을 넣어준다

for dataset in train_test:

    dataset['Age'].fillna(dataset['Age'].mean(), inplace = True)



X_test['Fare'].fillna(dataset['Fare'].min(), inplace = True)



# 결측치 확인

print(X_train.isnull().sum())

print(X_test.isnull().sum())
# Data binning



# Name data

for dataset in train_test:

    dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]+)\.',expand = False)

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

    dataset['Title_binning'] = dataset['Title'].astype('category').cat.codes

    

# Age data

# 0 ~ 10세: 0, 10 ~ 20세: 1, 20 ~ 30세: 2, 30 ~ 40세: 3, 40 ~ 50세: 4, 50 ~ 60세: 5, 60세 이상 : 6

for dataset in train_test:

    dataset.loc[dataset['Age'] <= 10, 'Age_binning'] = 0

    dataset.loc[(dataset['Age'] >10) & (dataset['Age']<= 20) , 'Age_binning'] = 1

    dataset.loc[(dataset['Age'] >20) & (dataset['Age']<= 30) , 'Age_binning'] = 2

    dataset.loc[(dataset['Age'] >30) & (dataset['Age']<= 40) , 'Age_binning'] = 3

    dataset.loc[(dataset['Age'] >40) & (dataset['Age']<= 50) , 'Age_binning'] = 4

    dataset.loc[(dataset['Age'] >50) & (dataset['Age']<= 60) , 'Age_binning'] = 5

    dataset.loc[(dataset['Age'] >60), 'Age_binning'] = 6

    
drop_features_2 = ['Age','Name','Title'] # binning 처리가 된 데이터를 drop한다.



X_train = X_train.drop(drop_features_2, axis =1)

X_test = X_test.drop(drop_features_2,axis = 1)

print(X_train,X_test)
train_x,test_x,train_y,test_y = train_test_split(X_train,y_train,test_size = 0.3)
# 적당한 model 찾기



# SVM

svc = SVC(kernel = 'linear', gamma = 'auto',C=5)

svc.fit(train_x,train_y)

y_pred_svc = svc.predict(test_x)



# Decision tree

tree = DecisionTreeClassifier(max_depth = 4)

tree.fit(train_x,train_y)

y_pred_tree = tree.predict(test_x)



# Randomforest

forest = RandomForestClassifier(n_estimators = 500, max_depth = 5,random_state = 1)

forest.fit(train_x,train_y)

y_pred_forest = forest.predict(test_x)



# Xgboost

xgb = XGBClassifier(n_estimators=500,max_depth = 5,random_state=1)

xgb.fit(train_x,train_y)

y_pred_xgb = xgb.predict(test_x)



# Adaboost

ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators = 500,

                        algorithm="SAMME.R",learning_rate=0.5)

ada.fit(train_x,train_y)

y_pred_ada = ada.predict(test_x)
print(accuracy_score(y_pred_svc,test_y))

print(accuracy_score(y_pred_tree,test_y))

print(accuracy_score(y_pred_forest,test_y))

print(accuracy_score(y_pred_xgb,test_y))

print(accuracy_score(y_pred_ada,test_y))
model_RF = RandomForestClassifier(n_estimators = 500, max_depth = 4,random_state = 1)

model_RF.fit(X_train,y_train)

y_pred = model_RF.predict(X_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived':y_pred})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")