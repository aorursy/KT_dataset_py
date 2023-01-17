import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
train_df = pd.read_csv('../input/titanic/train.csv')

test_df = pd.read_csv('../input/titanic/test.csv')

df = [train_df, test_df]
print('train_shape:',train_df.shape)

train_df.head()
print('test_shape:', test_df.shape)

test_df.head()
train_df.info()
train_df.isnull().sum()
test_df.isnull().sum()
sns.heatmap(train_df.isnull(), cbar=False, yticklabels=False)
train_df.Embarked.value_counts()
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])

df = [train_df, test_df]
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()
grid = sns.FacetGrid(train_df, col='Survived')

grid.map(plt.hist, 'Embarked')
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)

grid.map(sns.barplot, 'Pclass', 'Survived', 'Sex', ci=None, palette='deep')

grid.add_legend()
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', ci=None, palette='deep')

grid.add_legend()
for dataset in df:

    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

train_df.head()
train_df.Parch.value_counts()
grid = sns.FacetGrid(train_df, col='Survived')

grid.map(plt.hist, 'Parch')

grid.add_legend()
train_df.SibSp.value_counts()
grid = sns.FacetGrid(train_df, col='Survived')

grid.map(plt.hist, 'SibSp')

grid.add_legend()
train_df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in df:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in df:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df.head()
train_df.Sex.value_counts()
train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()
for dataset in df:

    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)
def func(a):

    try:

        return a.split(' ')[-1]

    except:

        return a
for dataset in df:

    dataset['Ticket_num'] = dataset['Ticket'].apply(lambda x: func(x))

    dataset['Ticket_num'] = dataset['Ticket_num'].replace('LINE', 0)
train_df.head()
train_df.Ticket_num.unique().shape
for dataset in df:

    dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])
train_df[['Title', 'Age']].groupby(['Title'], as_index=False).mean()
for dataset in df:

    dataset['Age_fill'] = 0

    dataset.loc[dataset['Title'] == 'Capt', 'Age_fill'] = 70

    dataset.loc[dataset['Title'] == 'Col', 'Age_fill'] = 58

    dataset.loc[dataset['Title'] == 'Countess', 'Age_fill'] = 33   

    dataset.loc[dataset['Title'] == 'Don', 'Age_fill'] = 40    

    dataset.loc[dataset['Title'] == 'Dr', 'Age_fill'] = 42   

    dataset.loc[dataset['Title'] == 'jonkheer', 'Age_fill'] = 38    

    dataset.loc[dataset['Title'] == 'Lady', 'Age_fill'] = 48    

    dataset.loc[dataset['Title'] == 'Major', 'Age_fill'] = 48.5    

    dataset.loc[dataset['Title'] == 'Master', 'Age_fill'] = 4.6    

    dataset.loc[dataset['Title'] == 'Miss', 'Age_fill'] = 21.8

    dataset.loc[dataset['Title'] == 'Mlle', 'Age_fill'] = 24

    dataset.loc[dataset['Title'] == 'Mme', 'Age_fill'] = 24

    dataset.loc[dataset['Title'] == 'Mr', 'Age_fill'] = 32.4

    dataset.loc[dataset['Title'] == 'Mrs', 'Age_fill'] = 35.9

    dataset.loc[dataset['Title'] == 'Ms', 'Age_fill'] = 28

    dataset.loc[dataset['Title'] == 'Rev', 'Age_fill'] = 43.2

    dataset.loc[dataset['Title'] == 'Sir', 'Age_fill'] = 49
train_df.head()
for dataset in df:

    dataset['Age'] = dataset['Age'].fillna(dataset['Age_fill'])
train_df.isnull().sum()
test_df.isnull().sum()
for dataset in df:

    dataset.drop(['Age_fill', 'Cabin', 'Ticket_num', 'Ticket', 'FamilySize', 'SibSp', 'Parch', 'Name'], axis=1, inplace=True)
train_df.head()
print('Before conversion')

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in df:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

print('After conversion')

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean().sort_values(by='Survived', ascending=False)
mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}

for dataset in df:

    dataset['Title'] = dataset['Title'].map(mapping)

train_df.head()
print('Introducing statistical parameters for all of the features:-')

train_df.describe()
train_df.info()
for dataset in df:

    dataset['Age'] = dataset['Age'].astype(int)
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
for dataset in df:

    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[dataset['Age'] > 64, 'Age'] = 4

train_df.head()
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].mode()[0])
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
for dataset in df:

    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2

    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)



df = [train_df, test_df]

train_df.head()
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
for dataset in df:

    dataset['Age*Class'] = dataset.Age * dataset.Pclass

train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head()
train_df.head()
train_df.drop(['PassengerId', 'AgeBand', 'FareBand'], axis=1, inplace=True)
X_train = train_df.drop(['Survived'], axis=1)

y_train = train_df['Survived']

X_test = test_df.drop(['PassengerId'], axis=1)
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

from sklearn.model_selection import cross_val_score

classifier = LogisticRegression()

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
coeff_df = pd.DataFrame(train_df.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(classifier.coef_[0])



coeff_df.sort_values(by='Correlation', ascending=False)
acc_Logistic = cross_val_score(classifier, X_train, y_train, cv=10, scoring='accuracy').mean()

acc_Logistic
classifier = KNeighborsClassifier(n_neighbors=5)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
acc_KNN = cross_val_score(classifier, X_train, y_train, cv=10, scoring='accuracy').mean()

acc_KNN
classifier = SVC()

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
acc_SVC = cross_val_score(classifier, X_train, y_train, cv=10, scoring='accuracy').mean()

acc_SVC
classifier = DecisionTreeClassifier()

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
acc_Tree = cross_val_score(classifier, X_train, y_train, cv=10, scoring='accuracy').mean()

acc_Tree
datasets = pd.DataFrame({

    'PassengerId': test_df['PassengerId'],

    'Survived': y_pred

})

datasets.to_csv('gender_submission3.csv', index=False)