import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

%matplotlib inline
init_notebook_mode(connected=True)
cf.go_offline()
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
train.info()
plt.figure(figsize=(18,10))
sns.heatmap(data=train.isnull(), cmap='viridis', yticklabels=False)
train.describe(include='all')
train[['Sex', 'Survived']].groupby('Sex').mean().iplot(kind='bar', colors='blue', title='Survival Rate based on Gender')
survived = train[train['Survived'] == 1]['Age']
not_survived = train[train['Survived'] == 0]['Age']

cf.subplots([survived.figure(kind='hist', colors='blue'), not_survived.figure(kind='hist', colors='blue')],
            shared_yaxes=True, subplot_titles=['Survived', 'Not Survived']).iplot()
train[['Pclass', 'Survived']].groupby('Pclass').mean().iplot(kind='bar', colors='blue', title='Survival Rate based on Passenger Class')
train[['Embarked', 'Survived']].groupby('Embarked').mean().iplot(kind='bar', colors='blue', title='Survival Rate based on Port of Embarkment')
sibSp = train[['SibSp', 'Survived']].groupby('SibSp').mean().iplot(kind='bar', colors='blue', title='Survival Rate based on Spouse or number of Siblings')
parCh = train[['Parch', 'Survived']].groupby('Parch').mean().iplot(kind='bar', colors='blue', title='Survival Rate based on presence of Parents or Children')
g = sns.FacetGrid(data=train, row='Pclass', col='Survived', size=2.2, aspect=3)
g.map(plt.hist, 'Age', bins=20, edgecolor='black')
g = sns.FacetGrid(data=train, row='Sex', col='Survived', size=2.5, aspect=3)
g.map(plt.hist, 'Age', bins=20, edgecolor='black')
combine = [train, test]
for ds in combine:
    ds.drop(labels=['Ticket', 'Fare', 'Cabin'], axis= 1, inplace= True) 

train.drop('PassengerId', axis=1, inplace=True)
train.head(2)
for ds in combine:
    ds['Title'] = ds['Name'].str.extract(pat= '([A-Za-z]+)\.', expand= False)

train['Title'].unique()
pd.crosstab(train['Sex'], train['Title'])
for ds in combine:
    ds['Title'] = ds['Title'].replace(['Capt', 'Col', 'Don', 'Dr', 'Jonkheer'], 'Rare')
    
    ds['Title'] = ds['Title'].replace(['Ms', 'Mme', 'Lady', 'Mlle'], 'Miss')
    ds['Title'] = ds['Title'].replace(['Countess'], 'Mrs')
    ds['Title'] = ds['Title'].replace(['Major', 'Rev', 'Sir'], 'Mr')
train[['Title', 'Survived']].groupby('Title').mean().iplot(kind='bar', colors='blue', title='Survival based on Titles')
def apply_median_age(values, df):
    for ds in combine:
        for title in values:
            median_age = ds.loc[ds['Title'] == title]['Age'].median()
            ds.loc[ds['Title'] == title, 'Age'] = ds.loc[ds['Title'] == title, 'Age'].fillna(median_age)

apply_median_age(train['Title'].unique(), combine)

train['Age'].isnull().sum()
train['Embarked'].isnull().sum()
most_embarked = train['Embarked'].mode()[0]

for ds in combine:
    ds['Embarked'].fillna(value= most_embarked, inplace= True)
train['Embarked'].isnull().sum()
plt.figure(figsize=(18,10))
sns.heatmap(train.isnull(), yticklabels= False, cmap= 'viridis')
for ds in combine:
    ds.drop(labels=['Name'], axis= 1, inplace= True)
train['Age'].iplot(kind='hist', colors='blue', title='Age distribution')
def age_categories(value):
    if value <= 16:
        return 'Child'
    elif 16 < value <=60:
        return 'Young and Adult'
    else:
        return 'Old'
for ds in combine:
    ds['AgeType'] = ds['Age'].apply(age_categories)
train[['AgeType', 'Survived']].groupby('AgeType').mean().iplot(kind='bar', colors='blue', title='Survival Rate based on Age Type')
for ds in combine:
    ds['FamilySize'] = ds['SibSp'] + ds['Parch'] + 1
for ds in combine:
    ds['IsAlone'] = 0
    ds.loc[ds['FamilySize'] == 1, 'IsAlone'] = 1
for ds in combine:
    ds['IsAlone'] = 0
    ds.loc[ds['FamilySize'] == 1, 'IsAlone'] = 1
train[['IsAlone', 'Survived']].groupby('IsAlone').mean().iplot(kind='bar', colors='blue', title="Survival Rate based on Family's presence")
for ds in combine:
    ds.drop(labels=['Age', 'SibSp', 'Parch', 'Title', 'FamilySize'], axis=1, inplace=True)
train.head()
train = pd.get_dummies(train)
train.head()
test = pd.get_dummies(test)
test.head()
X_train = train.drop('Survived', axis=1)
y_train = train['Survived']
X_test = test.drop('PassengerId', axis=1).copy()

X_train.shape, y_train.shape, X_test.shape
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
logreg_score = round(logreg.score(X_train, y_train) * 100, 2)
logreg_score
from sklearn.svm import SVC, LinearSVC
svc = SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)

svc_score = round(svc.score(X_train, y_train) * 100, 2)
svc_score
linsvc = LinearSVC()
linsvc.fit(X_train, y_train)
y_pred = linsvc.predict(X_test)

linsvc_score = round(linsvc.score(X_train, y_train) * 100, 2)
linsvc_score
from sklearn.tree import DecisionTreeClassifier

dectree = DecisionTreeClassifier()
dectree.fit(X_train, y_train)
y_pred = dectree.predict(X_test)

dectree_score = round(dectree.score(X_train, y_train) * 100, 2)
dectree_score
from sklearn.ensemble import RandomForestClassifier

randcls = RandomForestClassifier()
randcls.fit(X_train, y_train)
y_pred = randcls.predict(X_test)

randcls_score = round(randcls.score(X_train, y_train) * 100, 2)
randcls_score
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_pred
    })
# submission.to_csv('../output/submission.csv', index=False)