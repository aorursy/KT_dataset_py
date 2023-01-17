# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



sns.set(rc={'figure.figsize': (10, 8)})



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv', index_col='PassengerId')

test_df = pd.read_csv('../input/test.csv', index_col='PassengerId')

train_df.head()
print('Train dataset:')

print(train_df.isna().sum()[train_df.isna().any()])



print('\nTest dataset:')

print(test_df.isna().sum()[test_df.isna().any()])
train_df.describe()
cat_feat = ['Sex', 'Embarked']



for cf in cat_feat:

    # add newline

    if cf != cat_feat[0]:

        print()

        

    print(train_df[cf].value_counts() / train_df[cf].count())
fig, ax = plt.subplots(2, 3, figsize=(24, 14))

cats = ['Sex', 'Pclass', 'Embarked']



for c, a in zip(cats, ax.T):

    a[0].set_ylim(0, 1)

    sns.barplot(x=c, y='Survived', data=train_df, ax=a[0])

    a[1].set_ylim(0, 700)

    sns.countplot(train_df[c], ax=a[1])
fig, ax = plt.subplots(1, 2, figsize=(16, 6))



sns.barplot(x='Pclass', y='Survived', hue='Sex', data=train_df, ax=ax[0])

sns.countplot(x='Pclass', hue='Sex', data=train_df, ax=ax[1])
g = sns.FacetGrid(train_df, col='Embarked')

g.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep', order=[1, 2, 3], hue_order=['male', 'female'])

g.add_legend()
fig, ax = plt.subplots(2, 2, figsize=(16, 12))



sns.violinplot(x='Survived', y='Age', data=train_df, ax=ax[0, 0])

sns.violinplot(x='Survived', y='Fare', data=train_df[train_df.Fare < 100], ax=ax[0, 1])

sns.distplot(train_df.Age.dropna(), ax=ax[1, 0])

sns.distplot(train_df.Fare, ax=ax[1, 1])
fig, ax = plt.subplots(2, 2, figsize=(16, 12))



sns.barplot(x='Parch', y='Survived', data=train_df, ax=ax[0, 0])

sns.barplot(x='SibSp', y='Survived', data=train_df, ax=ax[0, 1])

sns.countplot(train_df.Parch, ax=ax[1, 0])

sns.countplot(train_df.SibSp, ax=ax[1, 1])
fig, ax = plt.subplots(1, 2, figsize=(16, 6))



sns.violinplot(x='Pclass', y='Fare', data=train_df[train_df.Fare < 100], ax=ax[0])

sns.violinplot(x='Pclass', y='Fare', hue='Survived', split=True, data=train_df[train_df.Fare < 100], ax=ax[1])
fig, ax = plt.subplots(1, 2, figsize=(16, 6))



sns.violinplot(x='Embarked', y='Fare', data=train_df[train_df.Fare < 100], ax=ax[0])

sns.violinplot(x='Embarked', y='Fare', hue='Survived', split=True, data=train_df[train_df.Fare < 100], ax=ax[1])
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)

test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
extract_title = lambda df: df.Name.str.extract(r'([A-Za-z]+)\.', expand=False)



train_df['Title'] = extract_title(train_df)

test_df['Title'] = extract_title(test_df)



train_df.Title.value_counts()
def replace_titles(df): 

    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    df['Title'] = df['Title'].replace('Mlle', 'Miss')

    df['Title'] = df['Title'].replace('Ms', 'Miss')

    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    return df

    

train_df = replace_titles(train_df)

test_df = replace_titles(test_df)



train_df = train_df.drop('Name', axis=1)

test_df = test_df.drop('Name', axis=1)
fig, ax = plt.subplots(1, 3, figsize=(24, 6))



sns.countplot(train_df.Title, ax=ax[0])

sns.barplot(x='Title', y='Survived', data=train_df, ax=ax[1])

sns.violinplot(x='Title', y='Age', data=train_df, ax=ax[2])
def add_family(df):

    df['Family'] = df.SibSp + df.Parch

    return df



train_df = add_family(train_df)

test_df = add_family(test_df)
sns.barplot(x='Family', y='Survived', hue='Sex', ci=False, data=train_df)
family_feat = ['SibSp', 'Parch']

train_df = train_df.drop(family_feat, axis=1)

test_df = test_df.drop(family_feat, axis=1)
def fill_ages(df):

    df.loc[(df.Age.isnull()) & (df.Title == 'Master'), 'Age'] = df[df.Title == 'Master'].Age.median()

    df.loc[(df.Age.isnull()) & (df.Title != 'Master'), 'Age'] = df[df.Title != 'Master'].Age.median()

    return df



def fill_embarked(df):

    df['Embarked'] = df.Embarked.fillna('S')

    return df



train_df = fill_ages(fill_embarked(train_df))

test_df = fill_ages(fill_embarked(test_df))



test_df['Fare'] = test_df.Fare.fillna(test_df.Fare.median())
def add_bands(df):

    df['AgeBand'] = pd.cut(df.Age, bins=5)

    df['FareBand'] = pd.qcut(df.Fare, q=4)

    return df



train_df = add_bands(train_df)

test_df = add_bands(test_df)



train_df = train_df.drop(['Age', 'Fare'], axis=1)

test_df = test_df.drop(['Age', 'Fare'], axis=1)
fig, ax = plt.subplots(1, 2, figsize=(16, 6))



sns.barplot(x='AgeBand', y='Survived', data=train_df, ax=ax[0])

sns.barplot(x='FareBand', y='Survived', data=train_df, ax=ax[1])
def factorize(df):

    df['Sex'] = df.Sex.factorize()[0]

    df['AgeBand'] = df.AgeBand.factorize(sort=True)[0]

    df['FareBand'] = df.FareBand.factorize(sort=True)[0]

    return df

    

def one_hot_encode(df):

    return pd.get_dummies(df, columns=['Embarked', 'Title'])



train_df = one_hot_encode(factorize(train_df))

test_df = one_hot_encode(factorize(test_df))



train_df.head()
X = train_df.drop('Survived', axis=1)

y = train_df.Survived
from xgboost import XGBClassifier

from sklearn.model_selection import KFold, cross_val_score, GridSearchCV



param_grid = {

    'n_estimators': [90, 95, 100],

    'learning_rate': [0.009, 0.01],

    'max_depth': [3],

    'min_child_weight' :range(1, 3),

    'gamma': [0, 0.001, 0.005]

}



gsearch = GridSearchCV(cv=5, estimator=XGBClassifier(), param_grid=param_grid, n_jobs=-1)

gsearch.fit(X, y)

gsearch.best_params_, gsearch.best_score_



# kfold = KFold(n_splits=5, random_state=7)

# scores = cross_val_score(model, X, y, cv=kfold)

# print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
from sklearn.svm import SVC



param_grid = {

    'C': np.arange(0.99, 1.1, 0.01)

}



model = SVC(gamma='auto')



gsearch = GridSearchCV(cv=5, estimator=model, param_grid=param_grid, n_jobs=-1)

gsearch.fit(X, y)

print(f'Best params: {gsearch.best_params_}')

print(f'Best score: {gsearch.best_score_}')

# kfold = KFold(n_splits=5, random_state=7)

# scores = cross_val_score(model, X, y, cv=kfold)

# print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
from sklearn.ensemble import RandomForestClassifier



param_grid = {

    'n_estimators': range(200, 300, 50),

    'max_depth': range(2, 5),

    'min_samples_split': [2, 3, 4],

    'bootstrap': [True, False]

}



model = RandomForestClassifier()



gsearch = GridSearchCV(cv=5, estimator=model, param_grid=param_grid, n_jobs=-1)

gsearch.fit(X, y)



print(f'Best params: {gsearch.best_params_}')

print(f'Best score: {gsearch.best_score_}')
model = SVC(C = 1.26, gamma = 0.09)

model.fit(X, y)

predictions = model.predict(test_df)
submit_df = pd.DataFrame({

    'PassengerId': test_df.index.values,

    'Survived': predictions

})

submit_df.to_csv('submission.csv', index=False)