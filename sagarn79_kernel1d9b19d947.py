# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

%matplotlib inline

import matplotlib.pyplot as plt

from matplotlib import style



from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
test_df = pd.read_csv("/kaggle/input/titanic/test.csv")

train_df = pd.read_csv("/kaggle/input/titanic/train.csv")
train_df.describe()
train_df.info()
train_df.head(15)
total = train_df.isnull().sum().sort_values(ascending=False)

percent1 = train_df.isnull().sum()/train_df.isnull().count()*100

percent2 = (round(percent1,1)).sort_values(ascending=False)

missing_data = pd.concat([total, percent2], axis=1, keys=['Total', '%'])

missing_data.head(5)
# Embarked has 2 missing values, Age has 177 and cabin has 687 missing values.

train_df.columns.values
survived = 'survived'

not_survived = 'not survived'

fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))

women = train_df[train_df['Sex']=='female']

men = train_df[train_df['Sex']=='male']

ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)

ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)

ax.legend()

ax.set_title('Female')

ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)

ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)

ax.legend()

_ = ax.set_title('Male')
FacetGrid = sns.FacetGrid(train_df, row='Embarked', size=4.5, aspect=1.6)

FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=None,  order=None, hue_order=None )

FacetGrid.add_legend()
sns.barplot(x='Pclass', y='Survived', data=train_df)
train_df = train_df.drop(['PassengerId'], axis=1)
import re

deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}

data = [train_df, test_df]



for dataset in data:

    dataset['Cabin'] = dataset['Cabin'].fillna("U0")

    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())

    dataset['Deck'] = dataset['Deck'].map(deck)

    dataset['Deck'] = dataset['Deck'].fillna(0)

    dataset['Deck'] = dataset['Deck'].astype(int) 
# I will drop the cabin feature

train_df = train_df.drop(['Cabin'], axis =1)

test_df = test_df.drop(['Cabin'], axis=1)
data = [train_df, test_df]



for dataset in data:

    mean = train_df['Age'].mean()

    std = test_df['Age'].std()

    is_null = dataset["Age"].isnull().sum()

    # compute random numbers between the mean, std and is_null

    rand_age = np.random.randint(mean - std, mean + std, size = is_null)

    # fill NaN values in Age column with random values generated

    age_slice = dataset["Age"].copy()

    age_slice[np.isnan(age_slice)] = rand_age

    dataset["Age"] = age_slice

    dataset["Age"] = train_df["Age"].astype(int)
train_df["Age"].isnull().sum()
train_df['Embarked'].describe()
common_value = 'S'

data = [train_df, test_df]



for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].fillna(common_value)
train_df.info()
data = [train_df, test_df]



for dataset in data:

    dataset['Fare'] = dataset['Fare'].fillna(0)

    dataset['Fare'] = dataset['Fare'].astype(int)
data = [train_df, test_df]

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
train_df = train_df.drop(['Name'], axis =1)

test_df = test_df.drop(['Name'], axis =1)
genders = {"male": 0, "female": 1}

data = [train_df, test_df]



for dataset in data:

    dataset['Sex'] = dataset['Sex'].map(genders)
train_df['Ticket'].describe()
train_df = train_df.drop(['Ticket'], axis=1)

test_df = test_df.drop(['Ticket'], axis=1)
ports = {"S": 0, "C": 1, "Q": 2}

data = [train_df, test_df]



for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].map(ports)
data = [train_df, test_df]

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
train_df['Age'].value_counts()
train_df.head(10)
data = [train_df, test_df]



for dataset in data:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3

    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4

    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5

    dataset['Fare'] = dataset['Fare'].astype(int)
data = [train_df, test_df]

for dataset in data:

    dataset['Age_Class']= dataset['Age']* dataset['Pclass']
data = [train_df, test_df]

for dataset in data:

    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']

    dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0

    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1

    dataset['not_alone'] = dataset['not_alone'].astype(int)
for dataset in data:

    dataset['Fare_Per_Person'] = dataset['Fare']/(dataset['relatives']+1)

    dataset['Fare_Per_Person'] = dataset['Fare_Per_Person'].astype(int)
train_df.head(20)
X_train = train_df.drop("Survived", axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop("PassengerId", axis=1).copy()
m = RandomForestClassifier(n_estimators= 100)

m.fit(X_train, Y_train)



Y_prediction = m.predict(X_test)



m.score(X_train, Y_train)

acc_random_forest = round(m.score(X_train, Y_train) * 100, 2)

print(round(acc_random_forest,2,), "%")
LReg = LogisticRegression()

LReg.fit(X_train, Y_train)



Y_pred = LReg.predict(X_test)



acc_log = round(LReg.score(X_train, Y_train) * 100, 2)

print(round(acc_log,2,), "%")
from sklearn.ensemble import RandomForestRegressor 

m = RandomForestClassifier(n_jobs= -1) 

m.fit(X_train, Y_train)



Y_prediction = m.predict(X_test)



m.score(X_train, Y_train) 

acc_random_forest = round(m.score(X_train, Y_train) * 100, 2) 

print(round(acc_random_forest,2,), "%")