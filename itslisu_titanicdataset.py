# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# load data to dataframe
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")
test_df = pd.read_csv("/kaggle/input/titanic/test.csv")
train_df.head()
test_df.head()
test_df.columns
train_df.describe()
# fill missing age values with mean of age
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
# import visualisation modules

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
train_df['Died'] = 1 - train_df['Survived']
train_df.groupby('Sex').agg('sum')[['Survived', 'Died']].plot(kind='pie',subplots=True)#, stacked=True)
train_df.groupby('Sex').agg('sum')[['Survived', 'Died']].plot(kind='bar', stacked=True)
train_df.groupby('Sex').agg('mean')[['Survived', 'Died']].plot(kind='bar', stacked=True)
fig = plt.figure(figsize=(25, 7))
sns.violinplot(x='Sex', y='Age', hue='Survived', data=train_df, split=True)
figs = plt.figure(figsize=(25, 7))
plt.hist([train_df[train_df['Survived']==1]['Fare'], train_df[train_df['Survived']==0]['Fare']],
         stacked=True, bins=50, label=['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.legend()
figs = plt.figure(figsize=(25, 7))
plt.hist([train_df[train_df['Survived']==0]['Age'], train_df[train_df['Survived']==1]['Age']], stacked=True, bins=10)
plt.xlabel('Age')
plt.ylabel('Number Dead/Survived')
plt.legend()
plt.figure(figsize=(25, 7))

ax = plt.subplot()
ax.scatter(train_df[train_df['Survived']==1]['Age'], train_df[train_df['Survived']==1]['Fare'], s=train_df[train_df['Survived']==1]['Fare'], c='teal')
ax.scatter(train_df[train_df['Survived']==0]['Age'], train_df[train_df['Survived']==0]['Fare'], s=train_df[train_df['Survived']==0]['Fare'], c='black')

train_df.groupby('Pclass').mean()['Fare'].plot(kind='bar')
train_df.corr()
train_df.groupby('Embarked').mean()['Fare'].plot(kind='bar')
plt.figure(figsize=(25, 7))

sns.violinplot(x='Embarked', y='Fare', hue='Survived', data=train_df, split=True)
def status(feature):
    print('Processing', feature, ': ok')

train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")

targets = train.Survived

train.drop(['Survived'], 1, inplace=True)

combined_data = train.append(test)
combined_data.reset_index(inplace=True)
combined_data.drop(['index', 'PassengerId'], inplace=True, axis=1)
combined_data.head()
print(combined_data.shape)
combined_data['Title'] = combined_data['Name'].str.split(', ').str[1].str.split('.').str[0]
Title_Dictionary = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Dona": "Royalty",
    "Sir" : "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess":"Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Master",
    "Lady" : "Royalty"
}

combined_data['Title'] = combined_data.Title.map(Title_Dictionary)
combined_data[combined_data['Title'].isnull()]
print(combined_data.iloc[:891].Age.isnull().sum())
print(combined_data.iloc[891:].Age.isnull().sum())
grouped_train = combined_data.iloc[:891].groupby(['Sex', 'Pclass', 'Title'])
grouped_median_train = grouped_train.median()
grouped_median_train = grouped_median_train.reset_index()[['Sex', 'Pclass', 'Title', 'Age']]
grouped_median_train.tail()
def fill_age(row):
    condition =  ((grouped_median_train['Sex'] == row['Sex']) & (grouped_median_train['Pclass']==row['Pclass']) & (grouped_median_train['Title']==row['Title']))   
    return grouped_median_train[condition]['Age'].values[0]

combined_data['Age'] = combined_data.apply(lambda row: fill_age(row) if np.isnan(row['Age']) else row['Age'], axis=1)
combined_data.head()
title_dummie = pd.get_dummies(combined_data['Title'], prefix='Title')
combined_data = pd.concat([combined_data, title_dummie], axis=1)
combined_data.drop(['Title', 'Name'], inplace=True, axis=1)
combined_data.head()
combined_data.Fare.fillna(combined_data.iloc[:891].Fare.mean(), inplace=True)
combined_data[combined_data['Fare'].isnull()]
combined_data[combined_data['Embarked'].isnull()]
train_df.groupby('Embarked').count().plot(kind='bar')
combined_data.Embarked.fillna('S', inplace=True)
em_dm = pd.get_dummies(combined_data['Embarked'], prefix='Embarked')
combined_data = pd.concat([combined_data, em_dm], axis=1)
combined_data[combined_data['Embarked'].isnull()]
combined_data = combined_data.drop(['Embarked'], axis=1)
train_cabin, test_cabin = set(), set()

for c in combined_data.iloc[:891]['Cabin']:
    try:
        train_cabin.add(c[0])
    except:
        train_cabin.add('U')
      
for c in combined_data.iloc[891:]['Cabin']:
    try:
        test_cabin.add(c[0])
    except:
        test_cabin.add('U')
print(train_cabin)
print(test_cabin)
combined_data.Cabin.fillna('U', inplace=True)
combined_data['Cabin'] = combined_data['Cabin'].map(lambda c: c[0])

cabin_dummies = pd.get_dummies(combined_data['Cabin'], prefix='Cabin')
combined_data = pd.concat([combined_data, cabin_dummies], axis=1)

combined_data.drop(['Cabin'], inplace=True, axis=1)

combined_data.head()
combined_data['Sex'] = combined_data['Sex'].map({'male': 1, 'female': 0})
combined_data.head()
pclass_dumies = pd.get_dummies(combined_data['Pclass'], prefix='Pclass')
combined_data = pd.concat([combined_data, pclass_dumies], axis=1)

combined_data.drop(['Pclass'], axis=1, inplace=True)
combined_data.head()
def cleanTicket(ticket):
    ticket = ticket.replace('.', '')
    ticket = ticket.replace('/', '')
    ticket = ticket.split()
    ticket = map(lambda t : t.strip(), ticket)
    ticket = list(filter(lambda t : not t.isdigit(), ticket))
    if len(ticket) > 0:
        return ticket[0]
    else: 
        return 'XXX'
    
tickets = set()
for t in combined_data['Ticket']:
    tickets.add(cleanTicket(t))

tickets
combined_data['Ticket'] = combined_data['Ticket'].map(cleanTicket)
ticket_dummies = pd.get_dummies(combined_data['Ticket'], prefix='Ticket')
combined_data = pd.concat([combined_data, ticket_dummies], axis=1)
combined_data.drop('Ticket', inplace=True, axis=1)
combined_data.head()
combined_data['FamilySize'] = combined_data['Parch'] + combined_data['SibSp'] + 1
combined_data['Singleton'] = combined_data['FamilySize'].map(lambda s: 1 if s==1 else 0)
combined_data['SmallSize'] = combined_data['FamilySize'].map(lambda s: 1 if 2<=s<=4 else 0)
combined_data['LargeSize'] = combined_data['FamilySize'].map(lambda s: 1 if 5<=s else 0)
combined_data.head()
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
# Cross-validation is a resampling procedure used to evaluate machine learning models on a limited data sample.

def compute_score(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)
    return np.mean(xval)
target = pd.read_csv('/kaggle/input/titanic/train.csv', usecols=['Survived'])['Survived'].values
train = combined_data.iloc[:891]
test = combined_data.iloc[891:]
clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf = clf.fit(train, targets)
features = pd.DataFrame()
features['feature'] = train.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)

features.plot(kind='barh', figsize=(25, 25))

model = SelectFromModel(clf, prefit=True)
train_reduced = model.transform(train)
print(train_reduced.shape)
# (891L, 14L)

test_reduced = model.transform(test)
print(test_reduced.shape)
# (418L, 14L)
logreg = LogisticRegression()
logreg_cv = LogisticRegressionCV()
rf = RandomForestClassifier()
gboost = GradientBoostingClassifier()

models = [logreg, logreg_cv, rf, gboost]

for model in models:
    print('Cross-validation of : {0}'.format(model.__class__))
    score = compute_score(clf=model, X=train_reduced, y=targets, scoring='accuracy')
    print('CV score = {0}'.format(score))
    print('****')
parameter_grid = {
    'max_depth': [4,6,8],
    'n_estimators': [50, 10, 100],
    'max_features': ['sqrt', 'auto', 'log2'],
    'min_samples_split': [2,3,10],
    'min_samples_leaf': [1,3,10],
    'bootstrap':[True, False]
    }

forest = RandomForestClassifier()
cross_validation = StratifiedKFold(n_splits=5)
grid_search = GridSearchCV(forest, scoring='accuracy', param_grid=parameter_grid, cv=cross_validation,verbose=1)
grid_search.fit(train, targets)
model = grid_search
parameters = grid_search.best_params_


print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

parameters = {'bootstrap': False, 'min_samples_leaf': 3, 'n_estimators': 50, 'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 6}
model = RandomForestClassifier(**parameters)
model.fit(train, targets)
output = model.predict(test).astype(int)
df_output = pd.DataFrame()
aux = pd.read_csv('/kaggle/input/titanic/test.csv')
df_output['PassengerId'] = aux['PassengerId']
df_output['Survived'] = output
df_output[['PassengerId','Survived']].to_csv('gridsearch_rf1.csv', index=False)
parameters = {'bootstrap': False, 'min_samples_leaf': 3, 'n_estimators': 50, 'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 6}
model = RandomForestClassifier(**parameters)
model.fit(train_reduced, targets)
output = model.predict(test_reduced).astype(int)
df_output = pd.DataFrame()
aux = pd.read_csv('/kaggle/input/titanic/test.csv')
df_output['PassengerId'] = aux['PassengerId']
df_output['Survived'] = output
df_output[['PassengerId','Survived']].to_csv('gridsearch_rf000.csv', index=False)
parameters = {'bootstrap': False, 'min_samples_leaf': 3, 'n_estimators': 100, 'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 6}
model = RandomForestClassifier(**parameters)
model.fit(train, targets)
output = model.predict(test).astype(int)
df_output = pd.DataFrame()
aux = pd.read_csv('/kaggle/input/titanic/test.csv')
df_output['PassengerId'] = aux['PassengerId']
df_output['Survived'] = output
df_output[['PassengerId','Survived']].to_csv('gridsearch_rf3.csv', index=False)
parameters = {'bootstrap': True, 'max_depth': 4, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 10}
model = RandomForestClassifier(**parameters)
model.fit(train, targets)
output = model.predict(test).astype(int)
df_output = pd.DataFrame()
aux = pd.read_csv('/kaggle/input/titanic/test.csv')
df_output['PassengerId'] = aux['PassengerId']
df_output['Survived'] = output
df_output[['PassengerId','Survived']].to_csv('gridsearch_rf2.csv', index=False)
model = GradientBoostingClassifier()
model.fit(train_reduced, targets)
output = model.predict(test_reduced).astype(int)
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
df_o = pd.DataFrame()
df_o['PassengerId'] = test_data['PassengerId']
df_o['Survived'] = output
df_o[['PassengerId', 'Survived']].to_csv('gbc.csv')
model = GradientBoostingClassifier()
model.fit(train, targets)
output = model.predict(test).astype(int)
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
df_o = pd.DataFrame()
df_o['PassengerId'] = test_data['PassengerId']
df_o['Survived'] = output
df_o[['PassengerId', 'Survived']].to_csv('gbc.csv')
