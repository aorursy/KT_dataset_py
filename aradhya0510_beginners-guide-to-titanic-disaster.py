import pandas as pd

import sklearn as sk
data_train = pd.read_csv('../input/train.csv') 
data_test = pd.read_csv('../input/test.csv')
data_train.head()
data_train.head()
data_train['Sex'] = data_train['Sex'].map({'female': 1, 'male':0})

data_train['Embarked'] = data_train['Embarked'].map({'S': 0, 'Q': 1, 'C': 2})
data_train.isnull().sum()
data_train.Age.mean()
data_train.Age.fillna(value=30,inplace=True)
data_train.Embarked.fillna(value=0,inplace=True)
del data_train['Name']

del data_train['Ticket']

del data_train['Cabin']
data_test.head()
data_test.isnull().sum()
data_test['Sex'] = data_test['Sex'].map({'female': 1, 'male':0})

data_test['Embarked'] = data_test['Embarked'].map({'S': 0, 'Q': 1, 'C': 2})
data_test.Age.mean()
data_test.Age.fillna(value=30,inplace=True)
data_test.Embarked.fillna(value=0,inplace=True)
data_test.Fare.mean()
data_test.Fare.fillna(value=35.6271884892086,inplace=True)
del data_test['Name']

del data_test['Ticket']

del data_test['Cabin']
data_train.head()
data_test.head()
train_cols = data_train.iloc[:,2:]

label_col = data_train.iloc[:,1]



X_train = train_cols

y_train = label_col



X_test = data_test.iloc[:,1:]
id_list = data_test.PassengerId
from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier(n_estimators=100)



train_features = X_train

train_target = y_train



clf = clf.fit(train_features, train_target)

pred = clf.predict(X_test)
Titanic_pred1 = pd.DataFrame({'PassengerId': id_list, 'Survived': pred})
Titanic_pred1.head()
from sklearn.ensemble import GradientBoostingClassifier



model = GradientBoostingClassifier()

model = model.fit(X_train,y_train)





pred2 = model.predict(X_test)
Titanic_pred2 = pd.DataFrame({'PassengerId': id_list, 'Survived': pred})
Titanic_pred2.head()
from sklearn.ensemble import BaggingClassifier



clf = BaggingClassifier(n_estimators=1000)



train_features = X_train

train_target = y_train



clf = clf.fit(train_features, train_target)

pred3 = clf.predict(X_test)
Titanic_pred3 = pd.DataFrame({'PassengerId': id_list, 'Survived': pred3})
Titanic_pred3.head()
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

pred4 = gnb.fit(X_train, y_train).predict(X_test)
Titanic_pred4 = pd.DataFrame({'PassengerId': id_list, 'Survived': pred4})
Titanic_pred4.head()