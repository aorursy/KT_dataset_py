import pandas as pd

from sklearn import preprocessing
train_data = pd.read_csv("../input/titanic/train.csv")

test_data = pd.read_csv("../input/titanic/test.csv")

y_test = pd.read_csv("../input/titanic/gender_submission.csv")

test_data = test_data.join(y_test.set_index('PassengerId')[['Survived']], on='PassengerId')
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1

train_data = train_data.drop(['Name', 'Cabin', 'Embarked', 'Ticket', 'SibSp', 'Parch'], axis=1)

test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1

test_data = test_data.drop(['Name', 'Cabin', 'Embarked', 'Ticket', 'SibSp', 'Parch'], axis=1)
train_data = train_data.fillna(train_data.mean())

test_data = test_data.fillna(test_data.mean())
train_data.head()
test_data.head()
train_data.loc[train_data["Sex"] == "male", "Sex"] = 0

train_data.loc[train_data["Sex"] == "female", "Sex"] = 1

train_data["Sex"] = train_data["Sex"].astype(int)

test_data.loc[test_data["Sex"] == "male", "Sex"] = 0

test_data.loc[test_data["Sex"] == "female", "Sex"] = 1

test_data["Sex"] = test_data["Sex"].astype(int)
train_data.head()
test_data.head()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = train_data[['Pclass',

                      'Sex',

                      'Age',

                      'Fare',

                      'FamilySize']]

y_train = train_data[['Survived']]

X_test = test_data[['Pclass',

                      'Sex',

                      'Age',

                      'Fare',

                      'FamilySize']]

y_test = test_data[['Survived']]
X_train[X_train.columns] = scaler.fit_transform(X_train[X_train.columns])

X_test[X_test.columns] = scaler.fit_transform(X_test[X_test.columns])

X_test
from sklearn.naive_bayes import GaussianNB
GNB = GaussianNB()

GNB.fit(X_train, y_train)
pred = GNB.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, pred)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': pred})
output
output.to_csv('titanic_results.csv',index=False, encoding='utf-8')