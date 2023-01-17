import pandas as pd

import seaborn as sns

%matplotlib inline
train = pd.read_csv('../input/titanic/train.csv')

train.head()
sns.countplot(x='Sex', data=train)
sns.barplot(x='Sex', y='Survived', data=train)
sns.boxplot(x='Sex', y='Age', data=train)
sns.swarmplot(x='Pclass', y='Fare', data=train)
sns.swarmplot(x='Pclass', y='Fare', hue='Sex', data=train)
iris = pd.read_csv('../input/iris/Iris.csv')

iris.head()
sns.heatmap(iris.corr())
sns.heatmap(iris.corr(), cmap='coolwarm', annot=True)