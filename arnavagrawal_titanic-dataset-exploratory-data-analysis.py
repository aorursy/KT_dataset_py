import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
plt.rcParams['figure.figsize'] = [6, 3]
plt.rcParams['figure.dpi'] = 80
titanic = sns.load_dataset('titanic')
titanic.head()
cols = titanic.columns
cols
titanic.describe()
titanic.info()
plt.style.use('ggplot')
titanic.isnull().sum()
sns.heatmap(titanic.isnull(), cmap = 'viridis', cbar = True)
corrmat = titanic.corr()
corrmat
sns.heatmap(corrmat)
print(list(cols))
fig, ax = plt.subplots(3, 3, figsize = (16, 16))

sns.countplot('survived', data = titanic, ax = ax[0][0])
sns.countplot('pclass', data = titanic, ax = ax[0][1])
sns.countplot('sex', data = titanic, ax = ax[0][2])
sns.countplot('sibsp', data = titanic, ax = ax[1][0])
sns.countplot('parch', data = titanic, ax = ax[1][1])
sns.countplot('embarked', data = titanic, ax = ax[1][2])
sns.countplot('alone', data = titanic, ax = ax[2][0])

sns.distplot(titanic['fare'], kde = True, ax = ax[2][1])
sns.distplot(titanic['age'], kde = True, ax = ax[2][2])

plt.tight_layout()
titanic['survived'].value_counts()
sns.countplot('survived', data = titanic)
plt.title('Titanic Survival Plot')
plt.show()
titanic['survived'].plot.hist()
titanic['survived'].value_counts().plot.pie()
titanic['survived'].value_counts().plot.pie(autopct = '%1.2f%%')
explode = [0, 0.1]
titanic['survived'].value_counts().plot.pie(explode = explode, autopct = '%1.2f%%')
titanic['pclass'].value_counts()
titanic.groupby(['pclass', 'survived'])['survived'].count()
sns.countplot('pclass', data = titanic)
sns.countplot('pclass', data = titanic, hue = 'survived')
titanic['pclass'].value_counts().plot.pie(autopct = "%1.1f%%")
sns.catplot(x = 'pclass', y = 'survived', kind = 'bar', data = titanic)
sns.catplot(x = 'pclass', y = 'survived', kind = 'point', data = titanic)
sns.catplot(x = 'pclass', y = 'survived', kind = 'violin', data= titanic)
titanic['sex'].value_counts()
titanic.groupby(['sex', 'survived'])['survived'].count()
sns.countplot('sex', data = titanic)
sns.countplot('sex', data = titanic, hue = 'survived')
titanic['sex'].value_counts().plot.pie(autopct = '%1.1f%%')
sns.catplot(x = 'sex', y = 'survived', kind = 'bar', data = titanic)
sns.catplot(x = 'sex', y = 'survived', kind = 'bar', data = titanic, hue = 'pclass')
sns.catplot(x = 'sex', y = 'survived', kind = 'bar', data = titanic, col = 'pclass')
sns.catplot(x = 'sex', y = 'survived', kind = 'bar', data = titanic, row = 'pclass')
sns.catplot(x = 'pclass', y = 'survived', kind = 'bar', data = titanic, col = 'sex')
sns.catplot(x = 'sex', y = 'survived', kind = 'point', data = titanic)
sns.catplot(x = 'sex', y = 'survived', kind = 'point', data = titanic, hue = 'pclass')
sns.catplot(x = 'pclass', y = 'survived', kind = 'point', data = titanic, hue = 'sex')
sns.catplot(x = 'sex', y = 'survived', kind = 'violin', data = titanic)
sns.catplot(x = 'sex', y = 'survived', kind = 'violin', data = titanic, hue = 'pclass')
sns.catplot(x = 'sex', y = 'survived', kind = 'violin', data = titanic, col = 'pclass')
titanic['age'].hist(bins = 30, density = True, color = 'orange', figsize = (10, 5))
plt.xlabel('Age')
plt.show()
sns.distplot(titanic['age'])
sns.distplot(titanic['age'], hist = False)
sns.kdeplot(titanic['age'], shade = True)
sns.catplot(x = 'sex', y = 'age', data = titanic, kind = 'box')
sns.catplot(x = 'sex', y = 'age', data = titanic, kind = 'box', hue = 'pclass')
sns.catplot(x = 'sex', y = 'age', data = titanic, kind = 'box', col = 'pclass')
sns.catplot(x = 'pclass', y = 'age', data = titanic, kind = 'violin')
sns.catplot(x = 'pclass', y = 'age', data = titanic, kind = 'violin', hue = 'sex')
sns.catplot(x = 'pclass', y = 'age', data = titanic, kind = 'violin', hue = 'sex', split = True)
sns.catplot(x = 'pclass', y = 'age', data = titanic, kind = 'violin', col = 'sex')
sns.catplot(x = 'pclass', y = 'age', kind = 'swarm', data = titanic)
sns.catplot(x = 'pclass', y = 'age', kind = 'swarm', data = titanic, col = 'sex')
sns.catplot(x = 'survived', y = 'age', data = titanic, kind = 'swarm', col = 'sex')
sns.catplot(x = 'survived', y = 'age', data = titanic, kind = 'swarm', row = 'sex', col = 'pclass')
titanic['fare'].hist(bins = 40, color = 'orange')
sns.distplot(titanic['fare'])
plt.xlabel('Fare')
plt.show()
sns.kdeplot(titanic['fare'], shade = True)
sns.catplot(x = 'sex', y = 'fare', data = titanic, kind = 'box')
sns.catplot(x = 'sex', y = 'fare', data = titanic, kind = 'box', hue = 'pclass')
sns.catplot(x = 'sex', y = 'fare', data = titanic, kind = 'box', col = 'pclass')
sns.catplot(x = 'sex', y = 'fare', data = titanic, kind = 'boxen', col = 'pclass')
sns.catplot(x = 'pclass', y = 'fare', data = titanic, kind = 'swarm', col = 'sex')
sns.catplot(x = 'survived', y = 'fare', data = titanic, kind = 'swarm', col = 'sex')
sns.catplot(x = 'survived', y = 'fare', data = titanic, kind = 'swarm', col = 'pclass')
sns.jointplot(x = 'age', y = 'fare', data = titanic)
sns.jointplot(x = 'age', y = 'fare', data = titanic, kind = 'kde')
sns.relplot(x = 'age', y = 'fare', data = titanic, row = 'sex', col = 'pclass')
titanic['sibsp'].value_counts()
sns.countplot('sibsp', data = titanic)
sns.countplot('sibsp', data = titanic, hue = 'survived')
sns.catplot(x = 'sibsp', y = 'survived', kind = 'bar', data = titanic)
sns.catplot(x = 'sibsp', y = 'survived', kind = 'bar', data = titanic, hue = 'sex')
sns.catplot(x = 'sibsp', y = 'survived', kind = 'bar', data = titanic, col = 'sex')
sns.catplot(x = 'sibsp', y = 'survived', kind = 'bar', data = titanic, col = 'pclass')
sns.catplot(x = 'sibsp', y = 'survived', kind = 'point', data = titanic)
sns.catplot(x = 'sibsp', y = 'survived', kind = 'point', data = titanic, hue = 'sex')
sns.catplot(x = 'sibsp', y = 'survived', kind = 'point', data = titanic, col = 'pclass')
sns.catplot(x = 'sibsp', y = 'fare', data = titanic, kind = 'swarm', col = 'sex')
sns.catplot(x = 'sibsp', y = 'fare', data = titanic, kind = 'swarm', col = 'pclass')
sns.catplot(x = 'sibsp', y = 'fare', data = titanic, kind = 'swarm', col = 'pclass', row = 'sex')
titanic['parch'].value_counts()
sns.countplot('parch', data = titanic)
sns.countplot('parch', data = titanic, hue = 'sex')
sns.catplot(x = 'parch', y = 'survived', data = titanic, kind = 'bar')
sns.catplot(x = 'parch', y = 'survived', data = titanic, kind = 'bar', hue = 'sex')
sns.catplot(x = 'parch', y = 'fare', data = titanic, kind = 'swarm')
sns.catplot(x = 'parch', y = 'fare', data = titanic, kind = 'swarm', col = 'sex')
sns.catplot(x = 'parch', y = 'fare', data = titanic, kind = 'swarm', col = 'pclass')
sns.catplot(x = 'parch', y = 'fare', data = titanic, kind = 'swarm', col = 'pclass', row = 'sex')
titanic['embarked'].value_counts()
sns.countplot('embarked', data = titanic)
sns.countplot('embarked', data = titanic, hue = 'survived')
sns.catplot(x = 'embarked', y = 'survived', kind = 'bar', data = titanic)
sns.catplot(x = 'embarked', y = 'survived', kind = 'bar', data = titanic, hue = 'sex')
sns.catplot(x = 'embarked', y = 'survived', kind = 'bar', data = titanic, col = 'sex')
titanic['who'].value_counts()
sns.countplot('who', data = titanic)
sns.countplot('who', data = titanic, hue = 'survived')
sns.catplot(x = 'who', y = 'survived', kind = 'bar', data = titanic)
sns.catplot(x = 'who', y = 'survived', kind = 'bar', data = titanic, hue = 'pclass')
sns.catplot(x = 'who', y = 'survived', kind = 'bar', data = titanic, col = 'parch')