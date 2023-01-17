%matplotlib inline

import numpy as np

import pandas as pd

from scipy import stats

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
df = pd.read_csv('../input/train.csv')

df.head()
df.shape
df.info()
df['Embarked'].value_counts()
df['Embarked'] = df['Embarked'].fillna('S')
sns.distplot(df[df['Age'].notnull()]['Age'])
df['Age'].describe()
df['Age'] = df['Age'].interpolate()
df['Age'].describe()
sns.distplot(df['Age'])
len(df['Cabin'].drop_duplicates())
df = df.drop('Cabin', axis=1)
df.head()
df = df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)

df.head()
sns.countplot(x='Pclass', data=df)
sns.set(style="whitegrid")

g = sns.PairGrid(data=df, x_vars=['Pclass'], y_vars='Survived', size=5)

g.map(sns.pointplot, color=sns.xkcd_rgb["plum"])

g.set(ylim=(0, 1))
h = sns.PairGrid(data=df, x_vars=['Sex'], y_vars='Survived', size=5)

h.map(sns.pointplot)

h.set(ylim=(0, 1))
df['is_child'] = df['Age'].apply(lambda x: 1 if x <= 15 else 0)
i = sns.PairGrid(data=df, x_vars=['is_child'], y_vars='Survived', size=5)

i.map(sns.pointplot)

i.set(ylim=(0, 1))
df['family'] = df['SibSp'] + df['Parch']

df = df.drop(['SibSp', 'Parch'], axis=1)
df['is_alone'] = df['family'].apply(lambda x: 1 if x == 0 else 0)
df['is_female'] = df['Sex'].apply(lambda x: 1 if x == 'female' else 0)

df = df.drop('Sex', axis=1)

df.head()
df = pd.get_dummies(df, prefix=['is'])
df.head()
X, y = df.drop(['Survived'], axis=1), df['Survived']
train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.75, test_size=0.25, stratify=y)
y.value_counts()
def build_classifier(model):

    classifier = model()

    classifier.fit(train_X, train_y)

    print(classifier.score(test_X, test_y))

    return classifier
decision_tree = build_classifier(GradientBoostingClassifier)
knn = build_classifier(KNeighborsClassifier)