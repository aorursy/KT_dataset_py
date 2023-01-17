import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC, VotingClassifier as VS
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.impute import SimpleImputer as SI;imp = SI()
from sklearn.preprocessing import MinMaxScaler as MMS;sca = MMS()

train = pd.read_csv('../input/titanic/train.csv')
train.drop(['Name', 'Ticket', 'Cabin', 'PassengerId','Age'], axis=1, inplace=True)
train['Sex'] = pd.get_dummies(train['Sex'], drop_first=True)
train['Fare'] = sca.fit_transform(np.array(train['Fare']).reshape(-1,1))
train.dropna(inplace=True)
X = train.drop(['Survived', 'SibSp', 'Parch','Embarked'], axis=1); y = train['Survived']
model = VS(estimators=[('dtc', DTC()),('rfc', RFC()),('lr', LR())], voting='hard', n_jobs=-1)

test = pd.read_csv('../input/titanic/test.csv')  
test.drop(['Name', 'Ticket', 'Cabin', 'Age'], axis=1, inplace=True)
test['Sex'] = pd.get_dummies(test['Sex'], drop_first=True)
test['Fare'] = imp.fit_transform(np.array(test['Fare']).reshape(-1,1))
test['Fare'] = sca.fit_transform(np.array(test['Fare']).reshape(-1,1))
X_valid = test.drop(['PassengerId', 'Parch', 'SibSp', 'Embarked'], axis=1)

model.fit(X, y)
pred = model.predict(X_valid)
sub = pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':pred})
sub.to_csv('trial.csv', index=False)