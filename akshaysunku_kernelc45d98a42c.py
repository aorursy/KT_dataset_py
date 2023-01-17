import pandas as pd

import keras

from keras import models

from keras import layers

from keras import optimizers

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import numpy as np

import re
train_data = pd.read_csv("../input/train.csv") 

test_data = pd.read_csv("../input/test.csv")

combined_data = [train_data, test_data]

Y = train_data['Survived']
train_data.head()
train_data.isnull().sum().sort_values(ascending=False)
test_data.isnull().sum().sort_values(ascending=False)
for dataset in combined_data:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1



    dataset['Singleton'] = dataset['FamilySize'].map(lambda s: 1 if s == 1 else 0)

    dataset['SmallFamily'] = dataset['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)

    dataset['LargeFamily'] = dataset['FamilySize'].map(lambda s: 1 if 5 <= s else 0)
train_data['CategoricalFare'] = pd.qcut(train_data['Fare'], 4)

train_data[['CategoricalFare', 'Survived']].groupby(['CategoricalFare']).mean()
test_data['Fare'].fillna(test_data['Fare'].mode()[0], inplace = True)
for dataset in combined_data:

    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)
print(train_data[['Sex', 'Survived']].groupby(['Sex'], as_index = False).mean())

for dataset in combined_data:

    dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)
for dataset in combined_data:

    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)

    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
for dataset in combined_data:

    age_avg = dataset['Age'].mean()

    age_std = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()



    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)
for dataset in combined_data:

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
for dataset in combined_data:

    dataset['Title'] = dataset.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())



normalized_titles = {

    "Capt":       "Officer",

    "Col":        "Officer",

    "Major":      "Officer",

    "Jonkheer":   "Royalty",

    "Don":        "Royalty",

    "Sir" :       "Royalty",

    "Dr":         "Officer",

    "Rev":        "Officer",

    "the Countess":"Royalty",

    "Dona":       "Royalty",

    "Mme":        "Mrs",

    "Mlle":       "Miss",

    "Ms":         "Mrs",

    "Mr" :        "Mr",

    "Mrs" :       "Mrs",

    "Miss" :      "Miss",

    "Master" :    "Master",

    "Lady" :      "Royalty"

}

train_data.Title = train_data.Title.map(normalized_titles)

test_data.Title = test_data.Title.map(normalized_titles)



title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Officer": 5, 'Royalty': 6}

for dataset in combined_data:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)
X = train_data.drop(['Survived', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'FamilySize', 'CategoricalFare'], axis=1)

test = test_data.drop(['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'FamilySize'], axis=1)

# X = pd.get_dummies(X)

X.shape, test.shape
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.8, test_size = 0.2, random_state = 0)
X_train.drop('PassengerId', axis=1, inplace=True)

X_test.drop('PassengerId', axis=1, inplace=True)
X_train.head(10)
model = models.Sequential()

model.add(layers.Dense(32, activation='relu', input_shape=(X_train.shape[1], )))

model.add(layers.Dense(32, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))



model.compile(optimizer=optimizers.RMSprop(lr=0.01), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=25, batch_size=8, validation_data=(X_test, Y_test))
PassengerId = test['PassengerId']

X = X.drop('PassengerId', axis=1)

test = test.drop('PassengerId', axis=1)
final_model = models.Sequential()

final_model.add(layers.Dense(32, activation='relu', input_shape=(X.shape[1], )))

final_model.add(layers.Dense(32, activation='relu'))

final_model.add(layers.Dense(1, activation='sigmoid'))



final_model.compile(optimizer=optimizers.RMSprop(lr=0.01), loss='binary_crossentropy', metrics=['accuracy'])
final_model.fit(X, Y, epochs=10, batch_size=8)
test.head()
predictions = final_model.predict(test)
preds= []

for i in predictions:

    if i[0] >= 0.5:

        preds.append(1)

    else:

        preds.append(0)
output = pd.DataFrame({'PassengerId': PassengerId, 'Survived': preds})
output.head(10)
output.to_csv('submission.csv', index=False)