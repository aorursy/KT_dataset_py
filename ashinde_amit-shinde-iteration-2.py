import pandas as pd

import numpy as np

import seaborn as sns

from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
combine = train.append(test)

combine.head()
train.shape,test.shape,combine.shape
combine['Age']=combine['Age'].fillna(np.mean(combine['Age']))



combine['Fare']= combine['Fare'].round()



combine['Age']= combine['Age'].round()



combine['Cabin']= combine['Cabin'].str[0]



combine['Cabin']=combine['Cabin'].fillna('Z')



combine['Embarked']=combine['Embarked'].fillna('S')



combine.head()
combine = combine.drop(['Name','Ticket'], axis=1)

combine = combine.drop(['PassengerId'], axis=1)


X_train = combine.iloc[:891,:-1]

Y_train = combine.iloc[:891,-1]



X_test = combine.iloc[891:,:-1]



X_train.shape,X_test.shape,Y_train.shape,combine.shape
#X_train['Embarked']= X_train['Embarked'].astype('category')

#X_test['Embarked']= X_test['Embarked'].astype('category')



var = ['Cabin','SibSp','Parch','Sex','Pclass','Embarked','Survived']

var1 = ['Cabin','SibSp','Parch','Sex','Pclass','Embarked']



le = preprocessing.LabelEncoder()



X_train['Cabin'] = le.fit_transform(X_train['Cabin'])

X_train['Sex'] = le.fit_transform(X_train['Sex'])

X_train['SibSp'] = le.fit_transform(X_train['SibSp'])

X_train['Parch'] = le.fit_transform(X_train['Parch'])

X_train['Pclass'] = le.fit_transform(X_train['Pclass'])

X_train['Embarked'] = le.fit_transform(X_train['Embarked'])



Y_train = le.fit_transform(Y_train)



X_test['Cabin'] = le.fit_transform(X_test['Cabin'])

X_test['Sex'] = le.fit_transform(X_test['Sex'])

X_test['SibSp'] = le.fit_transform(X_test['SibSp'])

X_test['Parch'] = le.fit_transform(X_test['Parch'])

X_test['Pclass'] = le.fit_transform(X_test['Pclass'])

X_test['Embarked'] = le.fit_transform(X_test['Embarked'])
X_test = X_test.fillna(0)
np.any(np.isnan(X_test)),np.all(np.isfinite(X_test))
from sklearn import ensemble

clf = ensemble.RandomForestClassifier(max_features='sqrt',n_estimators=50)



clf = clf.fit(X_train,Y_train)

Y_pred = clf.predict(X_test)
clf1 = LogisticRegression()

clf1 = clf1.fit(X_train,Y_train)

Y_pred = clf1.predict(X_test)
submission = pd.DataFrame({'PassengerID':test.PassengerId,

                          'Survived':Y_pred})



submission.to_csv('submission_Logistic.csv',index=False)