import numpy as np

import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")

train.head()
train.shape
train.isnull().sum()
test.isnull().sum()
train.drop(['Cabin'],axis=1,inplace=True)

test.drop(['Cabin'],axis=1,inplace=True)
train.drop(['Ticket'],axis=1,inplace=True)

test.drop(['Ticket'],axis=1,inplace=True)
train.drop(['Name'],axis=1,inplace=True)

test.drop(['Name'],axis=1,inplace=True)
print("Mean age:", np.mean(train['Age'].dropna()))

print("Mean age:", np.mean(test['Age'].dropna()))
from collections import Counter

from sklearn import preprocessing



train['Age'].fillna(30, inplace=True)

test['Age'].fillna(30, inplace=True)



ctr = Counter(train['Embarked'])

print("Embarked feature's most common 2 data points:", ctr.most_common(2))

train['Embarked'].fillna('S', inplace=True)

test.isnull().sum()
test['Fare'].fillna(np.mean(test['Fare']), inplace=True)

test['Age'] = test.Age.astype(int)



test.isnull().sum()

test.info()
import copy



encoder = preprocessing.LabelEncoder()



embarkedEncoder = copy.copy(encoder.fit(train['Embarked']))

train['Embarked'] = embarkedEncoder.transform(train['Embarked'])



sexEncoder = copy.copy(encoder.fit(train['Sex']))

train['Sex'] = sexEncoder.transform(train['Sex'])



train['Fare'] = train['Fare'].astype(int)

train.loc[train.Fare<=7.91,'Fare']=0

train.loc[(train.Fare>7.91) &(train.Fare<=14.454),'Fare']=1

train.loc[(train.Fare>14.454)&(train.Fare<=31),'Fare']=2

train.loc[(train.Fare>31),'Fare']=3



train['Age']=train['Age'].astype(int)

train.loc[ train['Age'] <= 16, 'Age']= 0

train.loc[(train['Age'] > 16) & (train['Age'] <= 32), 'Age'] = 1

train.loc[(train['Age'] > 32) & (train['Age'] <= 48), 'Age'] = 2

train.loc[(train['Age'] > 48) & (train['Age'] <= 64), 'Age'] = 3

train.loc[train['Age'] > 64, 'Age'] = 4

import copy



encoder = preprocessing.LabelEncoder()



embarkedEncoder = copy.copy(encoder.fit(test['Embarked']))

test['Embarked'] = embarkedEncoder.transform(test['Embarked'])



sexEncoder = copy.copy(encoder.fit(test['Sex']))

test['Sex'] = sexEncoder.transform(test['Sex'])

train.head()
test.head()
X=train.drop('Survived',axis=1)

y=train['Survived'].astype(int)



train['Fare'] = train['Fare'].astype(int)

train['Age'] = train['Age'].astype(int)

test['Fare'] = test['Fare'].astype(int)



from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.svm import SVC

from sklearn.preprocessing import Imputer





from sklearn.model_selection import train_test_split



sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=0)

acc_dict = {}



for train_index, test_index in sss.split(X, y):

    

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    



classifier=SVC()

xtrain=train.iloc[:,1:]

ytrain=train.iloc[:,:1]

ytrain=ytrain.values.ravel()

classifier.fit(xtrain,ytrain)



testIm=Imputer(missing_values='NaN',strategy='most_frequent',axis=1)

Age1=testIm.fit_transform(test.Age.values.reshape(1,-1))

Fare2=testIm.fit_transform(test.Fare.values.reshape(1,-1))

test['Age']=Age.T

test['Fare']=Fare.T

test.set_index('PassengerId',inplace=True)



test['Fare'] = test['Fare'].astype(int)

test.loc[test.Fare<=7.91,'Fare']=0

test.loc[(test.Fare>7.91) &(test.Fare<=14.454),'Fare']=1

test.loc[(test.Fare>14.454)&(test.Fare<=31),'Fare']=2

test.loc[(test.Fare>31),'Fare']=3



test['Age']=test['Age'].astype(int)

test.loc[ test['Age'] <= 16, 'Age']= 0

test.loc[(test['Age'] > 16) & (test['Age'] <= 32), 'Age'] = 1

test.loc[(test['Age'] > 32) & (test['Age'] <= 48), 'Age'] = 2

test.loc[(test['Age'] > 48) & (test['Age'] <= 64), 'Age'] = 3

test.loc[test['Age'] > 64, 'Age'] = 4



Result=classifier.predict(test)

print(Result)

print(len(Result))