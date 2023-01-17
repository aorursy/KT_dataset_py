# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import random as rnd



from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns



from sklearn.ensemble import RandomForestClassifier
df_train = pd.read_csv("../input/titanic/train.csv")

df_test = pd.read_csv("../input/titanic/test.csv")

combine = [df_train, df_test]
df_train.head()
df_train.columns
df_train.describe()
df_train.describe(include= ['O'])
df_train.info()

print('_'*45)

df_test.info()
df_train.Embarked.unique()
df_train[['Pclass','Survived']].groupby(['Pclass'], as_index=False).mean()
df_train[['Sex','Survived']].groupby(['Sex'], as_index=False).mean()
df_train[['SibSp','Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending= False)
df_train[['Parch','Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
bar = sns.FacetGrid(df_train, col='Survived')

bar.map(plt.hist, 'Age', bins=20)
grid = sns.FacetGrid(df_train, col='Survived',row='Pclass', height=2.2, aspect=1.6)

grid.map(plt.hist, 'Age')
grid = sns.FacetGrid(df_train, row='Embarked', height=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()
grid = sns.FacetGrid(df_train, row='Embarked', col='Survived', height=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

grid.add_legend()
df_train = df_train.drop(['Ticket', 'Cabin'], axis=1)

df_test = df_test.drop(['Ticket', 'Cabin'], axis=1)

combine = [df_train, df_test]
for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(df_train['Title'], df_train['Sex'])
for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Capt','Col','Countess','Don','Dr','Jonkheer','Lady',

                                                'Major','Rev','Sir'],'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

df_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
map_title = {'Master':1,'Miss':2,'Mr':3,'Mrs':4,'Rare':5}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(map_title)

    dataset['Title'] = dataset['Title'].fillna(0)

    

df_train.head(6)
df_train = df_train.drop(['Name', 'PassengerId'], axis=1)

df_test = df_test.drop(['Name'], axis=1)

combine = [df_train, df_test]

df_train.shape, df_test.shape
for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)



df_train.head()
grid = sns.FacetGrid(df_train, row='Pclass', col='Sex', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()
guess_ages = np.zeros((2,3))



for dataset in combine:

    for i in range(0, 2):

        for j in range(0, 3):

            guess_df = dataset[(dataset['Sex'] == i) & \

                                  (dataset['Pclass'] == j+1)]['Age'].dropna()



            # age_mean = guess_df.mean()

            # age_std = guess_df.std()

            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)



            age_guess = guess_df.median()



            # Convert random age float to nearest .5 age

            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

            

    for i in range(0, 2):

        for j in range(0, 3):

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\

                    'Age'] = guess_ages[i,j]



    dataset['Age'] = dataset['Age'].astype(int)



df_train.head()
df_train['AgeBand'] = pd.cut(df_train['Age'], 5)

df_train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
for dataset in combine:    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']

df_train.head()
df_train = df_train.drop(['AgeBand'], axis=1)

combine = [df_train, df_test]

df_train.head()
for dataset in combine:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1



df_train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1



df_train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
df_train = df_train.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

df_test = df_test.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

combine = [df_train, df_test]



df_train.head()
for dataset in combine:

    dataset['Age*Class'] = dataset.Age * dataset.Pclass



df_train.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)
freq_port = df_train.Embarked.dropna().mode()[0]

freq_port
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    

df_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} )



df_train.head()
df_test['Fare'].fillna(df_test['Fare'].dropna().median(), inplace=True)

df_test.head()
df_train['FareBand'] = pd.qcut(df_train['Fare'], 4)

df_train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
for dataset in combine:

    dataset.loc[dataset['Fare']<=7.91,'Fare'] = 0

    dataset.loc[(dataset['Fare']>7.91) & (dataset['Fare']<=14.454),'Fare'] = 1

    dataset.loc[(dataset['Fare']>14.454) & (dataset['Fare']<=31.0),'Fare'] = 2

    dataset.loc[(dataset['Fare']>31.0) & (dataset['Fare']<=512.329),'Fare'] = 3

df_train.head()
df_train = df_train.drop(['FareBand'],axis=1)

combine = [df_train,df_test]

df_train.head()
X_train = df_train.drop("Survived", axis=1)

Y_train = df_train["Survived"]

X_test  = df_test.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
submission = pd.DataFrame({

        "PassengerId": df_test["PassengerId"],

        "Survived": Y_pred

    })
submission.to_csv('submission.csv', index=False)