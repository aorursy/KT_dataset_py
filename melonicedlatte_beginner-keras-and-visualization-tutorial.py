import pandas as pd



train_x = pd.read_csv("../input/titanic/train.csv", sep=',')

test_x = pd.read_csv("../input/titanic/test.csv", sep=',')
train_x.head(5)
train_x.isnull().sum()
import seaborn as sns

import matplotlib.pyplot as plt



sns.barplot(x="Pclass", y="Survived", data=train_x);
sns.barplot(x="Sex", y="Survived", data=train_x);
sns.barplot(x="SibSp", y="Survived", data=train_x);
sns.barplot(x="Parch", y="Survived", data=train_x);
sns.barplot(x="Age", y="Survived", data=train_x);
import numpy as np



def make_bins(d, col, factor=2):

    rounding = lambda x: np.around(x / factor)

    d[col] = d[col].apply(rounding)

    return d



t = make_bins(train_x.copy(True), 'Age', 7.5)

sns.barplot(x="Age", y="Survived", data=t);
t = make_bins(train_x.copy(True), 'Age', 5)

sns.barplot(x="Age", y="Survived", data=t);
g = sns.FacetGrid(train_x, col='Survived')

g.map(plt.hist, 'Age', bins=20);
t = make_bins(train_x, 'Fare', 10)

sns.barplot(x="Fare", y="Survived", data=t);
sns.barplot(x="Pclass", y="Fare", data=t);
sns.barplot(x="Embarked", y="Survived", data=train_x);
sns.barplot(x="Embarked", y="Fare", data=train_x);
train_x.head(3)
train_x['Title'] = train_x.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
train_x['Title'].value_counts()
train_x['Title'] = train_x['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don',\

                                             'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],\

                                             'Rare')

train_x['Title'] = train_x['Title'].replace('Mlle', 'Miss')

train_x['Title'] = train_x['Title'].replace('Ms', 'Miss')

train_x['Title'] = train_x['Title'].replace('Mme', 'Mrs')
sns.barplot(x="Title", y="Survived", data=train_x);
_, train_x['Title'] = np.unique(train_x['Title'], return_inverse=True)
train_x['Title'].head(10)
train_x.drop(['SibSp', 'Parch', 'Ticket', 'Embarked', 'Name',\

        'Cabin', 'PassengerId', 'Fare', 'Age'], inplace=True, axis=1)
train_x.dropna(inplace=True)
_, train_x['Sex'] = np.unique(train_x['Sex'], return_inverse=True)
train_y = np.ravel(train_x.Survived) # Make 1D

train_x.drop(['Survived'], inplace=True, axis=1)
from keras.models import Sequential

from keras.layers import Dense



model = Sequential()



model.add(Dense(16, activation='relu', input_shape=(3,)))

model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
model.fit(train_x, train_y, epochs=50, batch_size=1, verbose=1)
to_test = test_x.copy(True)



# Add Title

to_test['Title'] = to_test.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

to_test['Title'] = to_test['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don',\

                                                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],\

                                                'Rare')

to_test['Title'] = to_test['Title'].replace('Mlle', 'Miss')

to_test['Title'] = to_test['Title'].replace('Ms', 'Miss')

to_test['Title'] = to_test['Title'].replace('Mme', 'Mrs')



_, to_test['Title'] = np.unique(to_test['Title'], return_inverse=True)



# Clean Data

to_test = to_test.drop(['SibSp', 'Parch', 'Ticket', 'Embarked', 'Name', 'Cabin',\

                        'PassengerId', 'Fare', 'Age'], axis=1)

_, to_test['Sex'] = np.unique(test_x['Sex'], return_inverse=True)
predictions = model.predict_classes(to_test).flatten()

predictions[:5]
submission = pd.DataFrame({

    "PassengerId": test_x["PassengerId"],

    "Survived": predictions

})

submission.to_csv('submission.csv', index=False)