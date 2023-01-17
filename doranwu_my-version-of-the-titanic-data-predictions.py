from speedml import Speedml

sml = Speedml('../input/train.csv', 

              '../input/test.csv', 

              target = 'Survived',

              uid = 'PassengerId')

sml.shape()
#This code was copied off of a data science tutorial from Manav Seghal

sml.feature.density('Age')

sml.train[['Age', 'Age_density']].head()

sml.feature.density('Ticket')

sml.train[['Ticket', 'Ticket_density']].head()

sml.feature.drop(['Ticket'])

sml.feature.fillna(a='Cabin', new='Z')

sml.feature.extract(new='Deck', a='Cabin', regex='([A-Z]){1}')

sml.feature.mapping('Sex', {'male': 0, 'female': 1})

sml.feature.sum(new='FamilySize', a='Parch', b='SibSp')

sml.feature.add('FamilySize', 1)

sml.feature.impute()

sml.feature.extract(new='Title', a='Name', regex=' ([A-Za-z]+)\.')

sml.feature.replace(a='Title', match=['Lady', 'Countess','Capt', 'Col',\

'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], new='Rare')

sml.feature.replace('Title', 'Mlle', 'Miss')

sml.feature.replace('Title', 'Ms', 'Miss')

sml.feature.replace('Title', 'Mme', 'Mrs')

sml.train[['Name', 'Title']].head()

sml.feature.drop(['Name'])

sml.feature.labels(['Title', 'Embarked', 'Cabin'])

sml.train.head()

sml.plot.importance()

#Most of this code is from Raman Sah and Ahmed BesBes

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

data = pd.read_csv('../input/train.csv')

from sklearn.preprocessing import LabelEncoder

data['Sex'] = LabelEncoder().fit_transform(data['Sex'])

data['Name'] = data['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())

titles = data['Name'].unique()

titles

data['Age'].fillna(-1, inplace=True)

medians = dict()

for title in titles:

    median = data.Age[(data["Age"] != -1) & (data['Name'] == title)].median()

    medians[title] = median

for index, row in data.iterrows():

    if row['Age'] == -1:

        data.loc[index, 'Age'] = medians[row['Name']]

fig = plt.figure(figsize=(15,6))

replacement = {

    'Don': 0,

    'Rev': 0,

    'Jonkheer': 0,

    'Capt': 0,

    'Mr': 1,

    'Dr': 2,

    'Col': 3,

    'Major': 3,

    'Master': 4,

    'Miss': 5,

    'Mrs': 6,

    'Mme': 7,

    'Ms': 7,

    'Mlle': 7,

    'Sir': 7,

    'Lady': 7,

    'the Countess': 7

}

data['Name'] = data['Name'].apply(lambda x: replacement.get(x))

from sklearn.preprocessing import StandardScaler

data['Name'] = StandardScaler().fit_transform(data['Name'].values.reshape(-1, 1))

data['Age'] = StandardScaler().fit_transform(data['Age'].values.reshape(-1, 1))

data['Fare'].fillna(-1, inplace=True)

medians = dict()

#banding Passenger Class and Fare

for pclass in data['Pclass'].unique():

    median = data.Fare[(data["Fare"] != -1) & (data['Pclass'] == pclass)].median()

    medians[pclass] = median

for index, row in data.iterrows():

    if row['Fare'] == -1:

        data.loc[index, 'Fare'] = medians[row['Pclass']]

data['Fare'] = StandardScaler().fit_transform(data['Fare'].values.reshape(-1, 1))

fig = plt.figure(figsize=(15,4))

data['Pclass'] = StandardScaler().fit_transform(data['Pclass'].values.reshape(-1, 1))

data.drop('Parch', axis=1, inplace = True)

data.drop('Ticket', axis=1, inplace=True)

data.drop('Embarked', axis=1, inplace=True)

data.drop('SibSp', axis=1, inplace=True)

#This is not cabin, it is deck

#replace it with common letters for deck

data['Cabin'].fillna('U', inplace=True)

data['Cabin'] = data['Cabin'].apply(lambda x: x[0])

data['Cabin'].unique()

replacement = {

    'T': 0,

    'U': 1,

    'A': 2,

    'G': 3,

    'C': 4,

    'F': 5,

    'B': 6,

    'E': 7,

    'D': 8

}

data['Cabin'] = data['Cabin'].apply(lambda x: replacement.get(x))

data['Cabin'] = StandardScaler().fit_transform(data['Cabin'].values.reshape(-1, 1))

data.head()['Cabin']

from sklearn.model_selection import train_test_split

survived = data['Survived']

data.drop('Survived', axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(data, survived, test_size=0.2, random_state=42)

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

models = [

    MLPClassifier(),

    SVC(),

    GaussianProcessClassifier(),

    DecisionTreeClassifier(),

    RandomForestClassifier(n_estimators=100),

    GaussianNB(),

    QuadraticDiscriminantAnalysis(),

    KNeighborsClassifier()

]

allscores= []

for model in models:

    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)

    allscores.append(score)    

finalscore = max(allscores)

print(allscores)

correctone = round(finalscore*418,0)

print("The highest accuracy I got was",finalscore,",with",correctone,"people out of 418 people by the RandomForestClassifier Algorithm")