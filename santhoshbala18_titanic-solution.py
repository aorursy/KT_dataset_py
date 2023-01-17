#b This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
from keras import utils
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
df['Sex'] = df['Sex'].astype('category')
df['Sex'] = df['Sex'].cat.codes
df.head(5)
if df.PassengerId.nunique() == df.shape[0]:
    print("unique ids")
else:
    print("not unique")
if df.count().min() == df.shape[0]:
    print("no nan")
else:
    print("nan")
print(df.isnull().sum())
print(df.dtypes)
print(df[['Pclass','Survived']].groupby(['Pclass']).mean())
print(df[['Sex','Survived']].groupby(['Sex']).mean().sort_values(by='Sex',ascending=False))
print(df[['SibSp','Survived']].groupby(['SibSp'],as_index=False).mean().sort_values(by='Survived',ascending=False))
print(df[['Parch','Survived']].groupby(['Parch'],as_index=False).mean().sort_values(by='Survived',ascending=False))
train_random_ages = np.random.randint(df['Age'].mean()-df['Age'].std(),df['Age'].mean()+df['Age'].std(),size=df['Age'].isnull().sum())
test_random_ages = np.random.randint(test['Age'].mean()-test['Age'].std(),test['Age'].mean()+test['Age'].std(),size=test['Age'].isnull().sum())
df['Age'][np.isnan(df['Age'])] = train_random_ages
test['Age'][np.isnan(test['Age'])] = test_random_ages
df['Age'] = df['Age'].astype('int')
test['Age'] = test['Age'].astype('int')
df['Embarked'].fillna('S',inplace=True)
test['Embarked'].fillna('S',inplace=True)
df['Port'] = df['Embarked'].map({'S':0,'C':1,'Q':2}).astype('int')
test['Port'] = test['Embarked'].map({'S':0,'C':1,'Q':2}).astype('int')
del df['Embarked']
del test['Embarked']                                                               
if test['Fare'].isnull().sum():
    print("null present")
else:
    print("no null")

test['Fare'].fillna(test['Fare'].median(),inplace=True)
df['Has_Cabin'] = df['Cabin'].apply(lambda x:0 if type(x)==float else 1)
test['Has_Cabin'] = test['Cabin'].apply(lambda x:0 if type(x)==float else 1)
print(df['Has_Cabin'].head(5))
del df['Cabin']
del test['Cabin']
df['FamilySize'] = df['SibSp']+df['Parch']+1
test['FamilySize'] = test['SibSp']+test['Parch']+1
df['IsAlone'] = 0
test['IsAlone'] = 0
df.loc[df['FamilySize']==1,'IsAlone']=1
test.loc[test['FamilySize']==1,'IsAlone']=1
print(df['IsAlone'].head(5))
full_dataset = [df,test]
df['Title'] = df.Name.str.extract('([a-zA-z]+)\.',expand=False)
test['Title'] = test.Name.str.extract('([a-zA-z]+)\.',expand=False)
for each in full_dataset:
    each['Title'] = each['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    each['Title'] = each['Title'].replace('Mlle', 'Miss')
    each['Title'] = each['Title'].replace('Ms', 'Miss')
    each['Title'] = each['Title'].replace('Mme', 'Mrs')
print(df['Title'])
for each in full_dataset:
    each['FamilySizeGroup'] = 'Small'
    each.loc[each['FamilySize']==1,'FamilySizeGroup'] = 'Alone'
    each.loc[each['FamilySize']>=5,'FamilySizeGroup'] = 'Big'
    
df[['FamilySizeGroup','Survived']].groupby(['FamilySizeGroup'],as_index=False).mean()
for dataset in full_dataset:    
    dataset.loc[ dataset['Age'] <= 14, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 14) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
df[['Age','Survived']].groupby(['Age'],as_index=False).mean()

for dataset in full_dataset:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
df['Title'].unique()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
family_mapping = {"Small": 0, "Alone": 1, "Big": 2}
for dataset in full_dataset:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['FamilySizeGroup'] = dataset['FamilySizeGroup'].map(family_mapping)

for dataset in full_dataset:
    dataset['IsChildandRich'] = 0
    dataset.loc[(dataset['Age'] <= 0) & (dataset['Pclass'] == 1 ),'IsChildandRich'] = 1  
    dataset.loc[(dataset['Age'] <= 0) & (dataset['Pclass'] == 2 ),'IsChildandRich'] = 1  

del df['Ticket']
del test['Ticket']

del df['Port']
del test['Port']

del df['Name']
del test['Name']

test['Sex'] = test['Sex'].astype('category')
test['Sex'] = test['Sex'].cat.codes
df.head(5)
del df['PassengerId']

X_train = df.drop(['Survived'],axis=1)
y_train = df['Survived']

X_test = test.drop(['PassengerId'],axis=1).copy()
del X_train['SibSp']
del X_train['Parch']

del X_test['SibSp']
del X_test['Parch']
del X_test['FamilySize']
(X_train.head(5))
X_test.head(5)
del X_train['FamilySize']
from sklearn.model_selection import cross_val_score,learning_curve,ShuffleSplit
from sklearn.linear_model import LogisticRegression

# Logistic Regression
logreg = LogisticRegression() #(C=0.1, penalty='l1', tol=1e-6)
logreg.fit(X_train, y_train)
Y_pred = logreg.predict(X_test)

result_train = logreg.score(X_train, y_train)
result_val = cross_val_score(logreg,X_train, y_train, cv=5).mean()
print('training score = %s , while validation score = %s' %(result_train , result_val))

from sklearn.svm import SVC
svc = SVC(C = 0.1, gamma=0.1)
svc.fit(X_train, y_train)
Y_pred = svc.predict(X_test)

result_train = svc.score(X_train, y_train)
result_val = cross_val_score(svc,X_train, y_train, cv=5).mean()
print('training score = %s , while validation score = %s' %(result_train , result_val))
from sklearn.ensemble import RandomForestClassifier
seed=2
random_forest =RandomForestClassifier(n_estimators=1000, criterion='entropy', max_depth=5, min_samples_split=2,
                           min_samples_leaf=1, max_features='auto',    bootstrap=False, oob_score=False, 
                           n_jobs=1, random_state=seed,verbose=0)

random_forest.fit(X_train, y_train)
Y_pred = random_forest.predict(X_test)

result_train = random_forest.score(X_train, y_train)
result_val = cross_val_score(random_forest,X_train, y_train, cv=5).mean()

print('training score = %s , while validation score = %s' %(result_train , result_val))
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('titanic.csv', index=False)
print('Submitted')