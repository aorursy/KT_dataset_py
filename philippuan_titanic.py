import pandas as pd
train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")
train.head()
test.head()
y_train = train.pop('Survived')
print(train['SibSp'].isnull().sum())
print(train['Parch'].isnull().sum())
print(train['Cabin'].isnull().sum())
print(train['Fare'].isnull().sum())
print(train['Embarked'].isnull().sum())
check = train['Name'].str.split(',',expand = True)[1].str.split('.',expand = True)[0]
check.head()
check = train['Name'].str.split(',',expand = True)[1].str.split('.',expand = True)[0]
test['title'] = test['Name'].str.split(',',expand = True)[1].str.split('.',expand = True)[0]
test.head()
train.head()

train['Sex'].unique()
sex_map = {'male':1,'female':0}
train.replace({'Sex':sex_map},inplace = True)
test.replace({'Sex':sex_map},inplace = True)
train.head()
train['title'].unique()
title_map = {' Mr': 1, ' Mrs': 2, ' Dona':2, ' Miss':3, ' Master':4, ' Don':5, ' Rev':6, ' Dr':7, ' Mme':2,
           ' Ms':9, ' Major':10, ' Lady':11, ' Sir':12, ' Mlle':3, ' Col':14, ' Capt':15,
           ' the Countess':11, ' Jonkheer':11}
train.replace({'title':title_map},inplace = True)
test.replace({'title':title_map},inplace = True)
train.head()
train['Embarked'].unique()

X = train[['Pclass','Sex','Age','SibSp','Parch','Fare','title']]
X = X.fillna(0)
y = y_train
y
X_test = test[['Pclass','Sex','Age','SibSp','Parch','Fare','title']]
X_test = X_test.fillna(0)
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X,y)
rfc_predict = rfc.predict(X_test)
rfc_predict
ids = test['PassengerId']
submit = pd.DataFrame()
submit['PassengerId'] = ids
submit['Survived'] = rfc_predict
submit.to_csv('./submission2.csv',index=False)