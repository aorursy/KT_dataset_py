%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sn

import re as re

from random import randint

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score

from sklearn.svm import SVC
train = pd.read_csv('../input/titanic/train.csv')

train.head(3)
test = pd.read_csv('../input/titanic/test.csv')

test.head(3)
ntrain = train.shape[0]

ntest = test.shape[0]



y_train = train['Survived'].values

passengerId = test['PassengerId']



dataset = pd.concat((train, test))
dataset.info()
dataset.describe(include='all')
dataset[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
pd.crosstab(dataset.Pclass, dataset.Survived, margins=True)
dataset[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean()
dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

dataset[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean()
dataset['IsAlone'] = 0

dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

dataset[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())

dataset['Fgroup'] = pd.qcut(dataset['Fare'], 10, labels=range(10))

dataset[['Fgroup', 'Survived']].groupby(['Fgroup'], as_index=False).mean()
def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    if title_search:

        return title_search.group(1)

    return ''





dataset['Title'] = dataset['Name'].apply(get_title)

pd.crosstab(dataset['Title'], dataset['Sex'])
dataset[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
def get_initial_name(name):

    initial_search = re.search('([A-Za-z]+)', name)

    if initial_search:

        return initial_search.group(1)

    return ''





dataset['LastName'] = dataset['Name'].apply(get_initial_name)

dataset['NumName'] = dataset['LastName'].factorize()[0]

    

dataset[['NumName', 'Survived']].groupby(['NumName'], as_index=False).mean()
print('Oldest Passenger was', dataset['Age'].max(), 'Years')

print('Youngest Passenger was', dataset['Age'].min(), 'Years')

print('Average Age on the ship was', int(dataset['Age'].mean()), 'Years')
dataset.groupby('Title').agg({'Age': ['mean', 'count']})
dataset = dataset.reset_index(drop=True)

dataset['Age'] = dataset.groupby('Title')['Age'].apply(lambda x: x.fillna(x.mean()))
dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Countess', 'Don', 'Dona' , 'Dr', 'Jonkheer', 'Lady', 

                                             'Major', 'Master',  'Miss'  ,'Mlle', 'Mme', 'Mr', 'Mrs', 'Ms', 'Rev', 'Sir'], 

                                            ['Sacrificed', 'Respected', 'Nobles', 'Mr', 'Mrs', 'Respected', 'Mr', 'Nobles', 

                                             'Respected', 'Kids', 'Miss', 'Nobles', 'Nobles', 'Mr', 'Mrs', 'Nobles', 'Sacrificed', 'Nobles'])

dataset['Title'] = dataset['Title'].replace(['Kids', 'Miss', 'Mr', 'Mrs', 'Nobles', 'Respected', 'Sacrificed'], [4, 4, 2, 5, 6, 3, 1])
dataset['TempAgroup'] = pd.qcut(dataset['Age'], 10)



dataset[['TempAgroup', 'Survived']].groupby(['TempAgroup'], as_index=False).mean()
dataset['Agroup'] = pd.qcut(dataset['Age'], 10, labels=range(10))

dataset[['Agroup', 'Survived']].groupby(['Agroup'], as_index=False).mean()
pd.crosstab(dataset.Pclass, dataset.Agroup, margins=True)
dataset['Gclass'] = 0

dataset.loc[((dataset['Sex'] == 'male') & (dataset['Pclass'] == 1)), 'Gclass'] = 1

dataset.loc[((dataset['Sex'] == 'male') & (dataset['Pclass'] == 2)), 'Gclass'] = 2

dataset.loc[((dataset['Sex'] == 'male') & (dataset['Pclass'] == 3)), 'Gclass'] = 2

dataset.loc[((dataset['Sex'] == 'female') & (dataset['Pclass'] == 1)), 'Gclass'] = 3

dataset.loc[((dataset['Sex'] == 'female') & (dataset['Pclass'] == 2)), 'Gclass'] = 4

dataset.loc[((dataset['Sex'] == 'female') & (dataset['Pclass'] == 3)), 'Gclass'] = 5

dataset.loc[(dataset['Age'] < 1), 'Gclass'] = 6
dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
dataset['Priority'] = 0

dataset.loc[(dataset['Title'] == 6), 'Priority'] = 1

dataset.loc[(dataset['Gclass'] == 3), 'Priority'] = 2

dataset.loc[(dataset['Gclass'] == 6), 'Priority'] = 3

dataset.loc[(dataset['Pclass'] == 1) & (dataset['Age'] <= 17), 'Priority'] = 4

dataset.loc[(dataset['Pclass'] == 2) & (dataset['Age'] <= 17), 'Priority'] = 5

dataset.loc[(dataset['Pclass'] == 3) & (dataset['Sex'] == 1), 'Priority'] = 6

dataset.loc[(dataset['Fgroup'] == 9), 'Priority'] = 7
dataset['FH'] = 0

dataset.loc[(dataset['Gclass'] == 1), 'FH'] = 0

dataset.loc[(dataset['Gclass'] == 2), 'FH'] = 0

dataset.loc[(dataset['Gclass'] == 3), 'FH'] = 1

dataset.loc[(dataset['Gclass'] == 4) & (dataset['FamilySize'] == 2), 'FH'] = 2

dataset.loc[(dataset['Gclass'] == 4) & (dataset['FamilySize'] == 3), 'FH'] = 3

dataset.loc[(dataset['Gclass'] == 4) & (dataset['FamilySize'] == 4), 'FH'] = 4

dataset.loc[(dataset['Gclass'] == 4) & (dataset['FamilySize'] == 1) & (dataset['Pclass'] == 1), 'FH'] = 5

dataset.loc[(dataset['Gclass'] == 4) & (dataset['FamilySize'] == 1) & (dataset['Pclass'] == 2), 'FH'] = 6

dataset.loc[(dataset['Gclass'] == 4) & (dataset['Fgroup'] == 3), 'FH'] = 7

dataset.loc[(dataset['Gclass'] == 4) & (dataset['Fgroup'] >= 5), 'FH'] = 8
dataset['MH'] = 0

dataset.loc[(dataset['Sex'] == 1), 'MH'] = 0

dataset.loc[(dataset['Gclass'] == 1), 'MH'] = 1

dataset.loc[(dataset['Gclass'] == 1) & (dataset['FamilySize'] == 2), 'MH'] = 2

dataset.loc[(dataset['Gclass'] == 1) & (dataset['FamilySize'] == 3), 'MH'] = 3

dataset.loc[(dataset['Gclass'] == 1) & (dataset['FamilySize'] == 4), 'MH'] = 4

dataset.loc[(dataset['Gclass'] == 1) & (dataset['FamilySize'] == 1) & (dataset['Pclass'] == 1), 'MH'] = 5

dataset.loc[(dataset['Gclass'] == 1) & (dataset['FamilySize'] == 1) & (dataset['Pclass'] == 2), 'MH'] = 6

dataset.loc[(dataset['Gclass'] == 1) & (dataset['Fgroup'] == 3), 'MH'] = 7

dataset.loc[(dataset['Gclass'] == 1) & (dataset['Fgroup'] >= 5), 'MH'] = 8
dataset['FL'] = 0

dataset.loc[(dataset['Gclass'] != 5), 'FL'] = 0

dataset.loc[(dataset['Gclass'] == 5) & (dataset['Fgroup'] < 5), 'FL'] = 1

dataset.loc[(dataset['Gclass'] == 5) & (dataset['Fgroup'] != 3), 'FL'] = 2

dataset.loc[(dataset['Gclass'] == 5) & (dataset['FH'] == 1), 'FL'] = 3

dataset.loc[(dataset['Gclass'] == 5) & (dataset['FamilySize'] < 2), 'FL'] = 4

dataset.loc[(dataset['Gclass'] == 5) & (dataset['FamilySize'] > 4), 'FL'] = 5

dataset.loc[(dataset['Gclass'] == 5) & (dataset['FamilySize'] == 1) & (dataset['Pclass'] == 3), 'FL'] = 6
dataset['ML'] = 0

dataset.loc[(dataset['Gclass'] == 2) & (dataset['Fgroup'] < 5), 'ML'] = 1

dataset.loc[(dataset['Gclass'] == 2) & (dataset['Fgroup'] != 3), 'ML'] = 2

dataset.loc[(dataset['Gclass'] == 2) & (dataset['MH'] < 7), 'ML'] = 3

dataset.loc[(dataset['Gclass'] == 2) & (dataset['FamilySize'] < 2), 'ML'] = 4

dataset.loc[(dataset['Gclass'] == 2) & (dataset['FamilySize'] > 4), 'ML'] = 5

dataset.loc[(dataset['Gclass'] == 2) & (dataset['FamilySize'] == 1) & (dataset['Pclass'] == 3), 'ML'] = 6

dataset.loc[(dataset['Gclass'] == 3) & (dataset['Fgroup'] < 5), 'ML'] = 1

dataset.loc[(dataset['Gclass'] == 3) & (dataset['Fgroup'] != 3), 'ML'] = 2

dataset.loc[(dataset['Gclass'] == 3) & (dataset['MH'] < 7), 'ML'] = 3

dataset.loc[(dataset['Gclass'] == 3) & (dataset['FamilySize'] < 2), 'ML'] = 4

dataset.loc[(dataset['Gclass'] == 3) & (dataset['FamilySize'] > 4), 'ML'] = 5

dataset.loc[(dataset['Gclass'] == 3) & (dataset['FamilySize'] == 1) & (dataset['Pclass'] == 3), 'ML'] = 6
dataset.columns
dfl = pd.DataFrame()

good_columns = ['Priority', 'Gclass', 'Title','NumName', 'FL','IsAlone','ML', 'FH', 'MH', 'Fgroup', 'FamilySize']

dfl[good_columns] = dataset[good_columns]
corrMatrix = pd.concat([dfl[:ntrain], train['Survived']], axis=1).corr()

fig, ax = plt.subplots(figsize=(10,10))     

sn.heatmap(corrMatrix, annot=True, linewidths=.5, ax=ax)

plt.show()
dfl = pd.DataFrame()

good_columns = ['Priority', 'Gclass', 'Title','NumName', 'FL','IsAlone','ML', 'FH', 'MH', 'Fgroup', 'FamilySize']

dfl[good_columns] = dataset[good_columns]
corrMatrix = pd.concat([dfl[:ntrain], train['Survived']], axis=1).corr()

fig, ax = plt.subplots(figsize=(10,10))     

sn.heatmap(corrMatrix, annot=True, linewidths=.5, ax=ax)

plt.show()
dfh = dfl.copy()

dfl_enc = dfl.apply(LabelEncoder().fit_transform)

one_hot_cols = dfh.columns.tolist()

dfh_enc = pd.get_dummies(dfh, columns=one_hot_cols)

X_train = dfh_enc[:ntrain]

X_test = dfh_enc[ntrain:]
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
model = SVC(probability=True, gamma=0.001, C=10)

scores = cross_val_score(model, X_train, y_train, cv=7)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
model = SVC(probability=True, gamma=0.001, C=10)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
output = pd.DataFrame({'PassengerId': passengerId, 'Survived': predictions})

output
output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")