# Import modules

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import tree

from sklearn.metrics import accuracy_score



# Figures inline and set visualization style

%matplotlib inline

sns.set()
import os

print(os.listdir('../input/titanic'))
# Import test and train datasets

df_train = pd.read_csv('../input/titanic/train.csv')

df_test = pd.read_csv('../input/titanic/test.csv')



# View first lines of training data

df_train.head(n=4)
# View first lines of test data

df_test.head()
df_train.info()
df_train.describe()
sns.countplot(x='Survived', data=df_train);
df_test['Survived'] = 0

df_test[['PassengerId', 'Survived']].to_csv('survivors.csv', index=False)
sns.countplot(x='Sex', data=df_train);
sns.factorplot(x='Survived', col='Sex', kind='count', data=df_train);
df_train.groupby(['Sex']).Survived.sum()
print(df_train[df_train.Sex == 'female'].Survived.sum()/df_train[df_train.Sex == 'female'].Survived.count())

print(df_train[df_train.Sex == 'male'].Survived.sum()/df_train[df_train.Sex == 'male'].Survived.count())
df_test['Survived'] = df_test.Sex == 'female'

df_test['Survived'] = df_test.Survived.apply(lambda x: int(x))

df_test.head()
df_test[['PassengerId', 'Survived']].to_csv('women_survive.csv', index=False)
sns.factorplot(x='Survived', col='Pclass', kind='count', data=df_train);
sns.factorplot(x='Survived', col='Embarked', kind='count', data=df_train);
sns.distplot(df_train.Fare, kde=False);
df_train.groupby('Survived').Fare.hist(alpha=0.6);
df_train_drop = df_train.dropna()

sns.distplot(df_train_drop.Age, kde=False);

sns.stripplot(x='Survived', y='Fare', data=df_train, alpha=0.3, jitter=True);
sns.swarmplot(x='Survived', y='Fare', data=df_train);
df_train.groupby('Survived').Fare.describe()
sns.lmplot(x='Age', y='Fare', hue='Survived', data=df_train, fit_reg=False, scatter_kws={'alpha':0.5});
sns.pairplot(df_train_drop, hue='Survived');