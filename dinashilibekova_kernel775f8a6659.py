import pandas as pd

import numpy as np
train_df = pd.read_csv('../input/titanic/train.csv')

train_df.tail()
train_df.describe()
train_df.describe(include=['O'])
s_total = train_df.Survived.sum()

s_female = train_df.Sex[(train_df.Survived==1) & (train_df.Sex=='female')].count()

s_male = train_df.Sex[(train_df.Survived==1) & (train_df.Sex=='male')].count()

print("Total survived: {}, survived female: {}, survived male: {}".format(s_total, s_female, s_male))
train_df[['Sex', 'Survived']].groupby(['Sex']).mean().sort_values(by='Survived', ascending=False)
train_df[['Pclass', 'Survived']].groupby(['Pclass']).mean().sort_values(by='Survived')
train_df[["SibSp", "Survived"]].groupby(['SibSp']).mean().sort_values(by='Survived')
train_df[["Parch", "Survived"]].groupby(['Parch']).mean().sort_values(by='Survived', ascending=False)
import seaborn as sns

import matplotlib.pyplot as plt
g = sns.FacetGrid(train_df, col='Survived', size=5)

g.map(plt.hist, 'Age', bins=20)
test_df = pd.read_csv('../input/titanic/test.csv')
combine = [train_df, test_df]

print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)



train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)

test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)

combine = [train_df, test_df]



"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape
data = [train_df, test_df]



for sett in data:

    sett['FamilySize'] = sett['SibSp']+sett['Parch']

    sett['Sex'] = sett['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    sett['Embarked'] = sett['Embarked'].fillna('S')

    sett['Age'] = sett['Age'].fillna(sett['Age'].mean())

    sett['Fare'] = sett['Fare'].fillna(train_df['Fare'].median())

    sett['Embarked'] = sett['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
for sett in data:

    sett['isAlone'] = 1

    sett.loc[sett.FamilySize>=1, 'isAlone']=0
data
colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

train_df = train_df.drop(['PassengerId', 'Name'], axis = 1)

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(train_df.astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)
from sklearn.model_selection import cross_validate

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.svm import SVC

rf = RandomForestClassifier()

et = ExtraTreesClassifier()

ada = AdaBoostClassifier()

gb = GradientBoostingClassifier()

svc = SVC()

y_train = train_df['Survived'].ravel()

train_df = train_df.drop(['Survived'], axis=1)

model = cross_validate(rf, train_df, y_train, cv=10)
model = cross_validate(gb, train_df, y_train, cv=10)

model
test_df
model = GradientBoostingClassifier()

PassengerId = test_df.PassengerId

test_df = test_df.drop(['PassengerId', 'Name'], axis = 1)

model.fit(train_df, y_train)

predictions = model.predict(test_df)
StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerId,

                            'Survived': predictions })

StackingSubmission.to_csv("StackingSubmission.csv", index=False)