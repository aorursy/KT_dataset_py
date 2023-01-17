import pandas as pd

import seaborn as sns

import numpy as np



# import dataset 

titanic = pd.read_csv("../input/titanic/train.csv")

titanic.head(10)
# number of people who survivied vs. not survivied

titanic['Survived'].value_counts().plot.bar()
# percentage of people survivied vs. not survived instead of count

(titanic['Survived'].value_counts() / len(titanic)).plot.bar()
# select variables

data = (titanic.loc[:, ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch']])

data.Sex[data.Sex == 'male'] = 0

data.Sex[data.Sex == 'female'] = 1

data['Sex'] = data['Sex'].astype(np.int64)

data.head()
# Exam the correlation between each pair of variables

f = data.corr()

sns.heatmap(f, annot=True)
p = sns.catplot(x='Pclass', y='Survived', hue='Sex', data=titanic,kind="bar", ci=None)

p.set_ylabels("Survival Rate").set_xlabels("Passengers Class").despine(left=True)