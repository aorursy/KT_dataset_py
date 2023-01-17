import numpy as np 
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
train.head()
train.drop(['Name','Ticket','Cabin','Embarked'],axis=1,inplace=True)
train.head()
train.isnull().sum()
train['Age']=train['Age'].fillna(train['Age'].mean())
train.isnull().sum()
label = LabelEncoder()
dicts = {}

label.fit(train.Sex.drop_duplicates()) 
dicts['Sex'] = list(label.classes_)
train.Sex = label.transform(train.Sex)

train
test.drop(['Name','Ticket','Cabin','Embarked'],axis=1,inplace=True)
test.head()
test.isnull().sum()
test['Age']=train['Age'].fillna(test['Age'].mean())
test['Fare']=train['Fare'].fillna(test['Fare'].mean())
test.isnull().sum()
label_test = LabelEncoder()
dicts_test = {}

label_test.fit(test.Sex.drop_duplicates()) 
dicts_test['Sex'] = list(label_test.classes_)
test.Sex = label_test.transform(test.Sex)

test.head()
#tr_rows, tr_cols = train.shape
#X_train = train.iloc[:, 1:tr_cols]
#y_train = train['Survived']
#x_test = test.iloc[:,0:]
#logreg = LogisticRegression()
#logreg.fit(X_train , y_train)
#predictions = logreg.predict(x_test)
#output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
#output
y = train["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch", "Age" ,"Parch" ,"Fare"]
X = pd.get_dummies(train[features])
X_test = pd.get_dummies(test[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
output
output.to_csv('my_submission.csv', index=False)