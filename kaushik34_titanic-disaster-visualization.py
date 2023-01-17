import pandas as pd

from pandas import Series, DataFrame

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

sns.set_style('whitegrid')

import warnings

warnings.filterwarnings("ignore")
titanic = pd.read_csv('../input/train.csv')

titanic.head()
titanic.describe()
titanic.info()
titanic.describe(include=['O'])
sns.countplot(titanic['Sex'])
sns.countplot(titanic['Pclass'],hue=titanic['Sex'])
def child(passenger):

    age,sex = passenger

    if age<16:

        return 'child'

    else:

        return sex

titanic['person'] = titanic[['Age', 'Sex']].apply(child,axis=1)
titanic[0:5]
sns.countplot(titanic['Pclass'],hue=titanic['person'])
titanic['person'].value_counts()
deck = titanic["Cabin"].dropna()
deck.head(10)
levels_list = []

for level in deck:

    levels_list.append(level[0])

cabin = DataFrame(levels_list)

cabin.columns = ['Cabin']

sns.countplot('Cabin',data=cabin,palette='winter')
sns.countplot('Embarked',data=titanic,hue='Pclass')
titanic["Alone"] = titanic.Parch + titanic.SibSp

titanic["Alone"].head()
titanic["Alone"].loc[titanic["Alone"] != 0] = 'With Family'

titanic["Alone"].loc[titanic["Alone"] == 0] = 'Alone'
titanic.head()
sns.countplot('Alone',data=titanic)
titanic["Survived"] = titanic.Survived.map({0: "no", 1: "yes"})

sns.countplot('Survived',data=titanic)
titanic['Survived'].value_counts(normalize=True) * 100
titanic[['Survived','Age']][0:10]
survived_yes = pd.crosstab(titanic.Sex,titanic.Survived)

survived_yes['yes'].plot.barh()

plt.xticks(size = 20)

plt.yticks(size = 20)
Survived_all = pd.crosstab(titanic.Sex,titanic.Survived)

Survived_all.plot.barh()

plt.xticks(size = 20)

plt.yticks(size = 20)
g = sns.FacetGrid(titanic, col='Survived')

g.map(plt.hist, 'Age', bins=20)
grid = sns.FacetGrid(titanic, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
relation = pd.crosstab( titanic.Pclass, titanic.Embarked )

relation.plot.barh(figsize=(10,10))

plt.xticks(size = 20)

plt.yticks(size = 20)

plt.title('Relation Between Pclass and Embarked',size=20)
dummy = relation.div(relation.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)

dummy = plt.xlabel('Pclass')

dummy = plt.ylabel('Embarked')
grid = sns.FacetGrid(titanic, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
tab = pd.crosstab(titanic['Pclass'], titanic['Survived'])

dummy = tab['yes'].div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)

dummy = plt.xlabel('Pclass')

dummy = plt.ylabel('Percentage')
tab = pd.crosstab(titanic['Pclass'], titanic['Survived'])

print(tab)

dummy = tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)

dummy = plt.xlabel('Pclass')

dummy = plt.ylabel('Percentage')
embarked_sur = pd.crosstab(titanic.Embarked,titanic.Survived)

embarked_sur.plot.bar()
print(titanic.isnull().sum())
age_avg = titanic['Age'].mean()

age_std = titanic['Age'].std()

age_null_count = titanic['Age'].isnull().sum()

age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

titanic['Age'][np.isnan(titanic['Age'])] = age_null_random_list

titanic['Age'] = titanic['Age'].astype(int)
titanic['Age'].head()
titanic.drop('Cabin',axis=1)
lol = pd.crosstab(titanic.Sex,titanic.Survived)

lol.plot.barh()

plt.xticks(size = 20)

plt.yticks(size = 20)

some = pd.crosstab( titanic.Pclass, titanic.Embarked )

some.plot.barh(figsize=(10,10))





dummy = some.div(some.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)

dummy = plt.xlabel('Pclass')

dummy = plt.ylabel('Embarked')
grid = sns.FacetGrid(titanic, row='Pclass', col='Sex', size=2.0, aspect=2.4)

grid.map(plt.hist, 'Age', bins=20)

grid.add_legend()
tab = pd.crosstab(titanic['Pclass'], titanic['Survived'])

print(tab)

dummy = tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)

dummy = plt.xlabel('Pclass')

dummy = plt.ylabel('Percentage')
duffers = pd.crosstab(titanic.Embarked,titanic.Survived)

duffers.plot.bar()
ax = sns.boxplot(x="Survived", y="Age", 

                data=titanic)

ax = sns.stripplot(x="Survived", y="Age",

                   data=titanic, jitter=True,

                   edgecolor="gray")

sns.plt.title("Survival by Age",fontsize=12);
corr=titanic.corr()#["Survived"]

plt.figure(figsize=(10, 10))



sns.heatmap(corr, vmax=.8, linewidths=0.01,

            square=True,annot=True,cmap='YlGnBu',linecolor="white")

plt.title('Correlation between features');
titanic['person'].value_counts().plot.bar()
## Thank You