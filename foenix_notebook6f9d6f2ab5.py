# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
train = pd.read_csv('../input/train.csv', header=0, dtype={'Age': np.float64})

test = pd.read_csv('../input/test.csv', header=0, dtype={'Age': np.float64})

full_data = [train, test]

print(train.info())

print('\n')

print(test.info())

print('\n')

print(train.columns)

print(test.columns)
print(train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())
print(train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean())
for df in full_data:

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

print(train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())
for df in full_data:

    df['IsAlone'] = 0

    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1

print(train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())
for dataset in full_data:

    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

train['CategoricalFare'] = pd.qcut(train['Fare'], 4)

print(train[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean())
for dataset in full_data:

    age_avg 	   = dataset['Age'].mean()

    age_std 	   = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)

    

train['CategoricalAge'] = pd.cut(train['Age'], 5)

print(train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())  
import re

def get_title(name):

	title_search = re.search(' ([A-Za-z]+)\.', name)

	# If the title exists, extract and return it.

	if title_search:

		return title_search.group(1)

	return ""



for dataset in full_data:

    dataset['Title'] = dataset['Name'].apply(get_title)



print(pd.crosstab(train['Title'], train['Sex']))
for dataset in full_data:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



print (train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())
for dataset in full_data:

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

train = train.drop(drop_elements, axis = 1)

train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)



test  = test.drop(drop_elements, axis = 1)



print(train.head(10))
df.head(2)
median_ages = np.zeros((2,3))

median_ages
for i in range(0, 2):

    for j in range(0, 3):

        median_ages[i,j] = df[(df['Gender'] == i) & (df['Pclass'] == j+1)]['Age'].dropna().median()

median_ages
df['AgeFill'] = df['Age']
df[ df['Age'].isnull() ][['Gender','Pclass','Age','AgeFill']].head(10)
for i in range(0, 2):

    for j in range(0, 3):

        df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),\

                'AgeFill'] = median_ages[i,j]
df['AgeIsNull'] = pd.isnull(df.Age).astype(int)
df.describe()
df['FamilySize'] = df['SibSp'] + df['Parch']
df['Age*Class'] = df.AgeFill * df.Pclass
df.head()
df.dtypes[df.dtypes.map(lambda x: x=='object')]
df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age'], axis=1) 
df = df.dropna()
df.dtypes
train_datadf = df.drop(['PassengerId'], axis=1)
train_datadf.dtypes
train_data = train_datadf.values
# Import the random forest package

from sklearn.ensemble import RandomForestClassifier 



# Create the random forest object which will include all the parameters

# for the fit

forest = RandomForestClassifier(n_estimators = 100)



# Fit the training data to the Survived labels and create the decision trees

forest = forest.fit(train_data[0::,1::],train_data[0::,0])
test_df = pd.read_csv('../input/test.csv', index_col=[0])
test_df['Gender'] = test_df.Sex.map( {'female': 0, 'male': 1} ).astype(int)
for i in range(0, 2):

    for j in range(0, 3):

        test_df.loc[ (test_df.Age.isnull()) & (test_df.Gender == i) & (test_df.Pclass == j+1),\

                'AgeFill'] = median_ages[i,j]
test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch']

test_df['Age*Class'] = test_df.AgeFill * test_df.Pclass
test_df.head()
test_df.dtypes
test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age'], axis=1)
test_df.dtypes
test_data = test_df.values
# Take the same decision trees and run it on the test data

output = forest.predict(test_data)