# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
train.info()
test.info()
print (train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean())
print (train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())
full = [train,test]

for data in full:

    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

    

print (train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())
for data in full:

    data['Alone'] = 0

    data.loc[data['FamilySize'] == 1, 'Alone'] = 1

print (train[['Alone', 'Survived']].groupby(['Alone'], as_index=False).mean())
print (train['Embarked'].value_counts())
for dataset in full:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

print (train['Embarked'].value_counts())

print (train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())
train.head()
import re as re

def get_title(name):

	title_search = re.search(' ([A-Za-z]+)\.', name)

	# If the title exists, extract and return it.

	if title_search:

		return title_search.group(1)

	return ""



for dataset in full:

    dataset['Title'] = dataset['Name'].apply(get_title)



print(pd.crosstab(train['Title'], train['Sex']))
for dataset in full:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



print(train[['Title','Survived']].groupby(['Title'], as_index=False).mean())

print(pd.crosstab(train['Title'], train['Age'].isnull()))
for data in full:

    mst_mean = data.loc[(data.Title=='Master')]['Age'].mean()

    mst_std = data.loc[(data.Title=='Master')]['Age'].std()



    ms_mean = data.loc[(data.Title=='Miss')]['Age'].mean()

    ms_std = data.loc[(data.Title=='Miss')]['Age'].std()



    mr_mean = data.loc[(data.Title=='Mr')]['Age'].mean()

    mr_std = data.loc[(data.Title=='Mr')]['Age'].std()



    mrs_mean = data.loc[(data.Title=='Mrs')]['Age'].mean()

    mrs_std = data.loc[(data.Title=='Mrs')]['Age'].std()



    r_mean = data.loc[(data.Title=='Rare')]['Age'].mean()

    r_std = data.loc[(data.Title=='Rare')]['Age'].std()



#     np.random.seed(5)

    mst_null_random = np.random.randint(mst_mean - mst_std, mst_mean + mst_std)

    ms_null_random = np.random.randint(ms_mean - ms_std, ms_mean + ms_std)

    mr_null_random = np.random.randint(mr_mean - mr_std, mr_mean + mr_std)

    mrs_null_random = np.random.randint(mrs_mean - mrs_std, mrs_mean + mrs_std)

    r_null_random = np.random.randint(r_mean - r_std, r_mean + r_std)

    

    

    data.loc[data['Title']=='Master',['Age']] = data['Age'].fillna(mst_null_random)

    data.loc[data['Title']=='Miss',['Age']] = data['Age'].fillna(ms_null_random)

    data.loc[data['Title']=='Mr',['Age']] = data['Age'].fillna(mr_null_random)

    data.loc[data['Title']=='Mrs',['Age']] = data['Age'].fillna(mrs_null_random)

    data.loc[data['Title']=='Rare',['Age']] = data['Age'].fillna(r_null_random)
train['AgeCategory'] = pd.cut(train['Age'], 6)

print ((train['Age'].isnull()).value_counts())

print (train[['AgeCategory', 'Survived']].groupby(['AgeCategory'], as_index=False).mean())
for dataset in full:

    

    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    

    

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    dataset['Title'] = dataset['Title'].map(title_mapping)

    #dataset['Title'] = dataset['Title'].fillna(0)

    

    

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    

    

    dataset.loc[ dataset['Age'] <= 12, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 12) & (dataset['Age'] <= 40), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 60), 'Age'] = 2

    dataset.loc[ dataset['Age'] > 60, 'Age'] = 3

    dataset['Age'] = dataset['Age'].astype(int)



    

drop_elements = ['Name', 'Ticket', 'Cabin', 'SibSp','FamilySize','Parch','Fare']

train = train.drop(drop_elements, axis = 1)

test  = test.drop(drop_elements, axis = 1)

train = train.drop(['PassengerId', 'AgeCategory'], axis = 1)



print (train.head(10))

from sklearn.model_selection import train_test_split



from sklearn.linear_model import LogisticRegression



from sklearn.metrics import accuracy_score

from sklearn.metrics import mean_absolute_error
predictors = train.drop(['Survived'], axis=1)

target = train["Survived"]

x_train, x_val, y_train, y_val = train_test_split(predictors, target, random_state = 0)
classifier = LogisticRegression()

classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_val)

acc = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc)
ids = test['PassengerId']

predictions = classifier.predict(test.drop('PassengerId', axis=1))



#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('submission.csv', index=False)