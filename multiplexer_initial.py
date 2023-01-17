import numpy as np 
import pandas as pd 
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# reading data
train = pd.read_csv('../input/'+os.listdir("../input")[0])
test = pd.read_csv('../input/'+os.listdir("../input")[1])
train.head()
train_x = train
train_x = train_x.drop('Survived',axis=1)
train_x = train_x.drop('Ticket',axis=1)
train_x = train_x.drop('Name',axis=1) 
train_x = train_x.drop('PassengerId',axis=1) 
train_x = train_x.drop('Embarked',axis=1) 
train_y = train['Survived'] # target
# dealing with empty,null data
train_x['Age'] = train_x['Age'].fillna(train_x['Age'].mean())
train_x['Fare'] = train_x['Fare'].fillna(train_x['Fare'].mean())
train_x['Cabin'] = train_x['Cabin'].fillna(train_x['Cabin'].astype(str).min())
# train_x['Embarked'] = train_x['Embarked'].fillna(train_x['Embarked'].astype(str).max())
# dealing with string objects
le = LabelEncoder()
le.fit(train_x['Sex'])
train_x['Sex'] = le.transform(train_x['Sex'])
# le.fit(train_x['Embarked'])
# train_x['Embarked'] = le.transform(train_x['Embarked'])
le.fit(train_x['Cabin'])
train_x['Cabin'] = le.transform(train_x['Cabin'])
train_x.head()
# train classifier
clf = RandomForestClassifier()
clf.fit(train_x,train_y)
# from test set
test_x = test
test_x = test_x.drop('Ticket',axis=1)
test_x = test_x.drop('PassengerId',axis=1)
test_x = test_x.drop('Name',axis=1)
test_x = test_x.drop('Embarked',axis=1)
test_x.head()
# fill empty,null
test_x['Age'] = test_x['Age'].fillna(test_x['Age'].mean())
test_x['Fare'] = test_x['Fare'].fillna(test_x['Fare'].mean())
test_x['Cabin'] = test_x['Cabin'].fillna(test_x['Cabin'].astype(str).min())
test_x.head()
# deal with string objects
le = LabelEncoder()
le.fit(test_x['Sex'])
test_x['Sex'] = le.transform(test_x['Sex'])
le.fit(test_x['Cabin'])
test_x['Cabin'] = le.transform(test_x['Cabin'])
test_x.head()
# predict survived
predicted = clf.predict(test_x)
# score
print(clf.score(train_x,train_y))
'Score: ',clf.score(test_x, predicted) # 1 ?
# write to csv
new = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': predicted})
new.to_csv('kaggle_submission_titanic.csv',index=False)
# survived by PassengerClass
train.Pclass[train['Survived'] == 1].describe(),train.Pclass[train['Survived'] == 0].describe()
# survived by gender
train.Sex[train['Survived']==1].describe(),train.Sex[train['Survived']==0].describe()