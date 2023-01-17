import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import KBinsDiscretizer  



sns.set()
raw_data = pd.read_csv('../input/titanic/train.csv')

raw_data.describe(include='all')
survived_df = raw_data.query('Survived == 1')

not_survived_df = raw_data.query('Survived == 0')

bins = 5



f, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True)



sns.distplot(survived_df["Age"] , color="green", label="Survived", bins=bins, ax=axes[0])

sns.distplot(not_survived_df["Age"] , color="red", label="Not survived", bins=bins, ax=axes[1])
sns.distplot(raw_data["Fare"])
raw_data['Cabin'].describe(include='all')
data = raw_data.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1)
data['Age'].fillna(raw_data["Age"].median(skipna=True), inplace=True)

data['Embarked'].fillna(raw_data['Embarked'].value_counts().idxmax(), inplace=True)
data.describe(include='all')
def get_title(name):

    title_dictionary = {

        'Capt': 'Dr/Clergy/Mil',

        'Col': 'Dr/Clergy/Mil',

        'Major': 'Dr/Clergy/Mil',

        'Jonkheer': 'Honorific',

        'Don': 'Honorific',

        'Dona': 'Honorific',

        'Sir': 'Honorific',

        'Dr': 'Dr/Clergy/Mil',

        'Rev': 'Dr/Clergy/Mil',

        'the Countess': 'Honorific',

        'Mme': 'Mrs',

        'Mlle': 'Miss',

        'Ms': 'Mrs',

        'Mr': 'Mr',

        'Mrs': 'Mrs',

        'Miss': 'Miss',

        'Master': 'Master',

        'Lady': 'Honorific'

    }



    key = name.split(',')[1].split('.')[0].strip()

    

    return title_dictionary[key]



data['Title'] = raw_data['Name'].map(get_title)



data.head()
data['FamilySize'] = data['Parch'] + data['SibSp'] + 1



data.head()
data['CabinType'] = raw_data['Cabin'].map(lambda x: x[0] if not pd.isna(x) else 'unknown')
est_age = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')

data['AgeClass'] = est_age.fit_transform(data[['Age']])



est_fare = KBinsDiscretizer(n_bins=8, encode='ordinal', strategy='uniform')

data['FareClass'] = est_fare.fit_transform(data[['Fare']])



data.head()
category_cols = ['AgeClass', 'FareClass', 'Title', 'Sex', 'Pclass', 'Embarked', 'CabinType']



data_with_dummies = pd.get_dummies(data, columns=category_cols)

data_with_dummies.head()
data_with_dummies.columns
targets = data_with_dummies['Survived']

inputs = data_with_dummies.drop(['Survived'], axis=1)
reg = LogisticRegression(random_state=42, solver='liblinear')

reg.fit(inputs, targets)

reg.score(inputs, targets)
raw_test_data = pd.read_csv('../input/titanic/test.csv')



test_data = raw_test_data.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1)



test_data['Age'].fillna(raw_data["Age"].median(skipna=True), inplace=True)

test_data['Embarked'].fillna(raw_data['Embarked'].value_counts().idxmax(), inplace=True)

test_data['Fare'].fillna(raw_data['Fare'].median(skipna=True), inplace=True)



test_data['Title'] = raw_test_data['Name'].map(get_title)



test_data['FamilySize'] = test_data['Parch'] + test_data['SibSp'] + 1



test_data['AgeClass'] = est_age.transform(test_data[['Age']])

test_data['FareClass'] = est_fare.transform(test_data[['Fare']])



test_data['CabinType'] = raw_test_data['Cabin'].map(lambda x: x[0] if not pd.isna(x) else 'unknown')



test_data_with_dummies = pd.get_dummies(test_data, columns=category_cols)



test_data_with_dummies['CabinType_T'] = 0



test_data_with_dummies.describe(include='all')
df = pd.DataFrame()



df['PassengerId'] = raw_test_data['PassengerId']

df['Survived'] = reg.predict(test_data_with_dummies)



df.to_csv('titanic_submission_4.csv',index=False)