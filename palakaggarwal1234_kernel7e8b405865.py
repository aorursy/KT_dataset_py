import numpy as np
import pandas as pd
import random as rnd

import matplotlib.pyplot as plt




from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

print(train_data.columns.values)
train_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_data[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_data[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print(train_data.head(5))
print(test_data.head(5))

train_data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
train_data.head()

test_data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test_data.head()
train_data['Age']=train_data['Age'].fillna((train_data['Age']).mean())
test_data['Age']=test_data['Age'].fillna((test_data['Age']).mean())
test_data['Fare']=test_data['Fare'].fillna((test_data['Fare']).mean())
train_data['Embarked']=train_data['Embarked'].fillna('S')
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
train_data['Sex'] = lb.fit_transform(train_data['Sex'])
test_data['Sex'] = lb.fit_transform(test_data['Sex'])
train_data['Embarked'] = lb.fit_transform(train_data['Embarked'])
test_data['Embarked'] = lb.fit_transform(test_data['Embarked'])
print(train_data.head())
print(test_data.head())
X_train = train_data.drop("Survived", axis=1)
Y_train = train_data["Survived"]
X_test  = test_data
X_train.shape, Y_train.shape, X_test.shape
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
y_pred = random_forest.predict(X_test)

print(y_pred)
my_predictions = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": y_pred
    })
print(my_predictions)