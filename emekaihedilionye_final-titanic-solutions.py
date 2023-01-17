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
# linear algebra

import numpy as np 



# data processing

import pandas as pd 



# data visualization

import seaborn as sns

%matplotlib inline

from matplotlib import pyplot as plt

from matplotlib import style



# machine learning

from sklearn.ensemble import RandomForestClassifier
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

train_data.info()
train_data.describe()
train_data.head()
total = train_data.isnull().sum().sort_values(ascending=False)

percent_1 = train_data.isnull().sum()/train_data.isnull().count()*100

percent_2 = (round(percent_1, 1)).sort_values(ascending=False)

missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])

missing_data.head(5)
train_data.columns.values
survived = 'survived'

not_survived = 'not survived'

fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))

women = train_data[train_data['Sex']=='female']

men = train_data[train_data['Sex']=='male']

ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)

ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)

ax.legend()

ax.set_title('Female')

ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)

ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)

ax.legend()

_ = ax.set_title('Male')
FacetGrid = sns.FacetGrid(train_data, row='Embarked', height=4.5, aspect=1.6)

FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=None,  order=None, hue_order=None )

FacetGrid.add_legend()
sns.barplot(x='Pclass', y='Survived', data=train_data)
grid = sns.FacetGrid(train_data, col='Survived', row='Pclass', height=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
data = [train_data, test_data]

for dataset in data:

    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']

    dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0

    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1

    dataset['not_alone'] = dataset['not_alone'].astype(int)

train_data['not_alone'].value_counts()
axes = sns.factorplot('relatives','Survived', data=train_data, aspect = 2.5,)

train_data = train_data.drop(['PassengerId'], axis=1)
import re

deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}

data = [train_data, test_data]



for dataset in data:

    dataset['Cabin'] = dataset['Cabin'].fillna("U0")

    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())

    dataset['Deck'] = dataset['Deck'].map(deck)

    dataset['Deck'] = dataset['Deck'].fillna(0)

    dataset['Deck'] = dataset['Deck'].astype(int)

    

# we can now drop the cabin feature

train_data = train_data.drop(['Cabin'], axis=1)

test_data = test_data.drop(['Cabin'], axis=1)

data = [train_data, test_data]



for dataset in data:

    mean = train_data["Age"].mean()

    std = test_data["Age"].std()

    is_null = dataset["Age"].isnull().sum()

    

    # compute random numbers between the mean, std and is_null

    rand_age = np.random.randint(mean - std, mean + std, size = is_null)

    

    # fill NaN values in Age column with random values generated

    age_slice = dataset["Age"].copy()

    age_slice[np.isnan(age_slice)] = rand_age

    dataset["Age"] = age_slice

    dataset["Age"] = train_data["Age"].astype(int)

train_data["Age"].isnull().sum()
train_data['Embarked'].describe()
common_value = 'S'

data = [train_data, test_data]



for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].fillna(common_value)

train_data.info()
data = [train_data, test_data]



for dataset in data:

    dataset['Fare'] = dataset['Fare'].fillna(0)

    dataset['Fare'] = dataset['Fare'].astype(int)
data = [train_data, test_data]

titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}



for dataset in data:

    # extract titles

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    # replace titles with a more common title or as Rare

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\

                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    # convert titles into numbers

    dataset['Title'] = dataset['Title'].map(titles)

    # filling NaN with 0, to get safe

    dataset['Title'] = dataset['Title'].fillna(0)

train_data = train_data.drop(['Name'], axis=1)

test_data = test_data.drop(['Name'], axis=1)

genders = {"male": 0, "female": 1}

data = [train_data, test_data]



for dataset in data:

    dataset['Sex'] = dataset['Sex'].map(genders)
train_data['Ticket'].describe()
train_data = train_data.drop(['Ticket'], axis=1)

test_data = test_data.drop(['Ticket'], axis=1)
ports = {"S": 0, "C": 1, "Q": 2}

data = [train_data, test_data]



for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].map(ports)
data = [train_data, test_data]

for dataset in data:

    dataset['Age'] = dataset['Age'].astype(int)

    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3

    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4

    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5

    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6

    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6



# let's see how it's distributed train_data['Age'].value_counts()
train_data.head(10)
data = [train_data, test_data]



for dataset in data:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3

    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4

    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5

    dataset['Fare'] = dataset['Fare'].astype(int)
data = [train_data, test_data]

for dataset in data:

    dataset['Age_Class']= dataset['Age']* dataset['Pclass']
for dataset in data:

    dataset['Fare_Per_Person'] = dataset['Fare']/(dataset['relatives']+1)

    dataset['Fare_Per_Person'] = dataset['Fare_Per_Person'].astype(int)

# Let's take a last look at the training set, before we start training the models.

train_data.head(10)
X_train = train_data.drop("Survived", axis=1)

y_train = train_data["Survived"]

X_test  = test_data.drop("PassengerId", axis=1).copy()
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, y_train)



y_prediction = random_forest.predict(X_test)



random_forest.score(X_train, y_train)

acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)

print(round(acc_random_forest,2,), "%")
from sklearn.model_selection import cross_val_score

rf = RandomForestClassifier(n_estimators=100)

scores = cross_val_score(rf, X_train, y_train, cv=10, scoring = "accuracy")

print("Scores:", scores)

print("Mean:", scores.mean())

print("Standard Deviation:", scores.std())
submission = pd.DataFrame({"PassengerId": test_data["PassengerId"],"Survived": y_prediction})

submission.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")