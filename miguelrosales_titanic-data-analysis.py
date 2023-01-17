import pandas as pd
import numpy as np

import re as re #regular expressions
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
%xmode Verbose
traindata = pd.read_csv ('/kaggle/input/titanic/train.csv')
traindata
testdata = pd.read_csv('/kaggle/input/titanic/test.csv')
testdata
genderdata = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
genderdata
combine = (traindata, testdata)
combine
import matplotlib.pyplot as plt

import seaborn as sns
corrmat = traindata.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
print (traindata[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())

print (traindata[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean())
print (traindata[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean())
print (traindata[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean())
for dataset in combine:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1 #←Got to know the logic of this code

print (traindata[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())
for dataset in combine:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

print (traindata[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

print (traindata[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())
for dataset in combine:

    dataset['Fare'] = dataset['Fare'].fillna(traindata['Fare'].median())

traindata['CategoricalFare'] = pd.qcut(traindata['Fare'], 4)

print (traindata[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean())
for dataset in combine:

    age_avg 	   = dataset['Age'].mean()

    age_std 	   = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)

    

traindata['CategoricalAge'] = pd.cut(traindata['Age'], 5)



print (traindata[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())
def get_title(name):

	title_search = re.search(' ([A-Za-z]+)\.', name)

	# If the title exists, extract and return it.

	if title_search:

		return title_search.group(1)

	return ""



for dataset in combine:

    dataset['Title'] = dataset['Name'].apply(get_title)



print(pd.crosstab(traindata['Title'], traindata['Sex']))
for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



print (traindata[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())
for dataset in combine:

    # Mapping Sex

    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    

    # Mapping titles

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

    

    # Mapping Embarked

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    

    # Mapping Fare

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

    

    # Mapping Age

    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']                           = 4



# Feature Selection

drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp',\

                 'Parch', 'FamilySize']

train = traindata.drop(drop_elements, axis = 1)

train = traindata.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)



test  = testdata.drop(drop_elements, axis = 1)



print (traindata.head(10))



train = traindata.values

test  = testdata.values
