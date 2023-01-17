import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

titanic = pd.read_csv('../input/titanic/train.csv')

titanic.head(3)
print("Unique Value: ", titanic.nunique())

print ("\nMissing values :  ", titanic.isnull().sum())
titanic = titanic.drop('Name', axis=1)

titanic = titanic.drop('Ticket', axis=1)

titanic = titanic.drop('Cabin', axis=1)

titanic['Fare'] = titanic['Fare'].fillna(np.mean(titanic['Fare']))

titanic['Age'] = titanic['Age'].fillna(np.mean(titanic['Age']))

titanic.Parch = titanic.Parch.apply(lambda x: 5 if x>=5 else x)

titanic.head(1)
titanic['Age'] = pd.cut(titanic.Age,

                     bins=[0, 10, 45,100],

                     labels=["young", "adult", "elderly"])
titanic['Fare'] = pd.cut(titanic.Fare,

                     bins=[0, 10, 50,10000],

                     labels=["low", "medium", "high"])

titanic.head(10)
titanic = titanic.dropna()

y = titanic['Survived']

X = titanic.drop('Survived', axis=1)

X = X.drop('PassengerId', axis=1)



X.nunique()
from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



X = X.apply(lambda col: le.fit_transform(col))

X.nunique()
onehotencoder = OneHotEncoder(categories = 'auto')

X = onehotencoder.fit_transform(X).toarray()

X_train = pd.DataFrame(X)

X_train.head(3)
from keras.models import Sequential

from keras.layers import Dense





model = Sequential()

model.add(Dense(64, input_dim=27, activation='relu'))

model.add(Dense(32, activation='relu'))

model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y, epochs=60, batch_size=5)
test = pd.read_csv('../input/titanic/test.csv')

test_id = test['PassengerId']

test = test.drop('PassengerId', axis=1)

test = test.drop('Name', axis=1)

test = test.drop('Ticket', axis=1)

test = test.drop('Cabin', axis=1)

test.Parch = test.Parch.apply(lambda x: 5 if x>=5 else x)

test['Fare'] = test['Fare'].fillna(np.mean(test['Fare']))

test['Age'] = test['Age'].fillna(np.mean(test['Age']))

test['Age'] = pd.cut(test.Age,

                     bins=[0, 10, 45,100],

                     labels=["young", "adult", "elderly"])

test['Fare'] = pd.cut(test.Fare,

                     bins=[0, 10, 50,10000],

                     labels=["low", "medium", "high"])

test.Fare = test.Fare.fillna("low")

test = test.apply(lambda col: le.fit_transform(col))

print(test.nunique())

len(test)
test = onehotencoder.fit_transform(test).toarray()

test = pd.DataFrame(test)

test.head(3)

pred = model.predict(test)

y1 = (pred > 0.5).astype(int).reshape(test.shape[0])

y2 = (pred > 0.3).astype(int).reshape(test.shape[0])

y3 = (pred > 0.9).astype(int).reshape(test.shape[0])



np.mean(y3)

output1 = pd.DataFrame({'PassengerId': test_id, 'Survived': y1})

output2 = pd.DataFrame({'PassengerId': test_id, 'Survived': y2})

output3 = pd.DataFrame({'PassengerId': test_id, 'Survived': y3})



output1.head(5)
output1.to_csv('Prediction 1.csv', index=False)

output2.to_csv('Prediction 2.csv', index=False)

output3.to_csv('Prediction 3.csv', index=False)
