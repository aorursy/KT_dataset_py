from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from pandas import get_dummies
import matplotlib as mpl
import xgboost as xgb
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import warnings
import sklearn
import scipy
import numpy
import json
import sys
import csv
import os

# 데이터 불러오기
gender = pd.read_csv("../input/gender_submission.csv")
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
combine = [train, test]
train.info()
train.isnull().sum()
test.info()
test.isnull().sum()
train.describe()
train.describe(include=["O"])
corr = train.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
fig, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            linewidths=.5, cbar_kws={"shrink": .6});
train[['Pclass', 'Survived', 'Age']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train.pivot_table('Survived', index='Sex', columns='Pclass')
grid = sns.FacetGrid(train, size=5)
grid.map(sns.barplot, 'Pclass', 'Survived', palette='deep', order=[1,2,3]);
train[['Sex', 'Survived']].groupby('Sex', as_index=False).mean().sort_values(by='Sex', ascending=False)
grid = sns.FacetGrid(train, size=5)
grid.map(sns.barplot, 'Sex', 'Survived', order=['male','female'], palette='deep');
grid = sns.FacetGrid(train, hue='Survived', size=3.5)
grid.map(plt.hist, 'Age', alpha=.5)
grid.add_legend();
grid = sns.FacetGrid(train, col='SibSp', col_wrap=4, size = 3)
grid.map(sns.barplot, 'Sex', 'Survived', order=['male','female'], palette='deep');
grid = sns.FacetGrid(train, col='Parch', col_wrap=4, size = 3)
grid.map(sns.barplot, 'Sex', 'Survived', order=['male','female'], palette='deep');
grid = sns.FacetGrid(train, size = 5)
grid.map(sns.barplot, 'Ticket', 'Survived', palette='deep');
grid = sns.FacetGrid(train, hue='Survived', size=3.5)
grid.map(plt.hist, 'Fare', alpha=.5)
grid.add_legend();
grid = sns.FacetGrid(train, size = 5)
grid.map(sns.barplot, 'Cabin', 'Survived', palette='deep');
grid = sns.FacetGrid(train, col='Embarked', size = 3)
grid.map(sns.barplot, 'Sex', 'Survived', palette='deep', order=['female','male']);
grid = sns.FacetGrid(train, col='Embarked')
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', order=[1,2,3], hue_order=['female','male'], palette='deep')
grid.add_legend();
grid = sns.FacetGrid(train, row="Survived", col='Embarked', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', ci=None, palette='deep')
grid.add_legend()
grid = sns.FacetGrid(train, col="Pclass", hue="Survived", size=3)
grid.map(plt.hist, "Fare", alpha=.5)
plt.xlim(0,300)
grid.add_legend();
grid = sns.FacetGrid(train, col="Pclass", row="Embarked", hue="Survived", size=3)
grid.map(plt.hist, "Fare", alpha=.5)
plt.xlim(0,300)
plt.ylim(0,100)
grid.add_legend()
grid = sns.FacetGrid(train, col="Pclass", hue="Survived", size=2.5, aspect=1.3)
grid.map(plt.hist, "Age", bins=20, alpha=0.5)
grid.add_legend();
grid = sns.FacetGrid(train, col="Pclass", hue="Survived", size=3)
grid.map(plt.scatter, "Fare", "Age", alpha=.5)
plt.xlim(0,300)
grid.add_legend();
print('Ticket unique value  : {0}'.format(len(train['Ticket'].unique())))
print('Cabin unique value : {0}   Cabin null값 : {1}'
      .format(len(train['Cabin'].unique()), sum(train['Cabin'].isnull())))
train_df = train.drop(["Ticket", "Cabin"], axis=1)
test_df = test.drop(["Ticket", "Cabin"], axis=1)
combine = [train_df, test_df]

print("Before : ", train.shape, test.shape)
print("After  : ", train_df.shape, test_df.shape)
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]*)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])
rarelist = []
for a in set(train_df['Title']):
    if list(train_df['Title']).count(a)<10:
        rarelist.append(a)
rarelist
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace('Mlle','Miss')
    dataset['Title'] = dataset['Title'].replace('Ms','Miss')
    dataset['Title'] = dataset['Title'].replace('Mme','Mrs')
    dataset['Title'] = dataset['Title'].replace(rarelist,'Rare')

train_df[['Title','Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    dataset['Title'].astype(int)
train_df['Title'].head()
train_df = train_df.drop(['Name','PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df,test_df]
train_df.shape, test_df.shape
for dataset in combine:
    dataset.Sex = dataset.Sex.map({'female':1,'male':0}).astype(int)

train_df.head()
for dataset in combine:
    for i in range(2):
        for j in range(3):
            temp = (dataset.Sex == i) & (dataset['Pclass'] == j+1)
            dataset.loc[temp,'Age'] = dataset.loc[temp,'Age']. \
            where(pd.notnull(dataset[temp].Age),(dataset[temp].Age.median()/0.5 + 0.5) * 0.5)
    dataset.Age = dataset.Age.astype(int)

train_df.head()
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby('AgeBand', as_index=False).mean().sort_values(by='AgeBand', ascending=True)
for dataset in combine:
    dataset.loc[dataset.Age <= 16, 'Age'] = 0
    dataset.loc[(dataset.Age > 16) & (dataset.Age <= 32), 'Age'] = 1
    dataset.loc[(dataset.Age > 32) & (dataset.Age <= 48), 'Age'] = 2
    dataset.loc[(dataset.Age > 48) & (dataset.Age <= 64), 'Age'] = 3
    dataset.loc[dataset.Age > 64, 'Age'] = 4

train_df.head()
train_df = train_df.drop('AgeBand', axis=1)
combine = [train_df, test_df]
train_df.head()
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_df[['FamilySize','Survived']].groupby('FamilySize', as_index=False).mean().sort_values(by='Survived',ascending=False)
for dataset in combine:
    dataset['isAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'isAlone'] = 1

train_df[['isAlone', 'Survived']].groupby('isAlone', as_index = False).mean()
train_df = train_df.drop(['Parch', 'SibSp'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp'], axis=1)
combine = [train_df, test_df]

train_df.head()
for dataset in combine:
    dataset['Age*Pclass'] = dataset.Age * dataset.Pclass

train_df[['Age*Pclass', 'Survived']].groupby('Age*Pclass', as_index=False).mean().sort_values(by='Survived', ascending = False)
train_df[['Age*Pclass','Survived']].groupby('Age*Pclass').count()
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(train_df.Embarked.dropna().mode()[0])

train_df[['Embarked','Survived']].groupby('Embarked', as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)

train_df.head()
print(train_df.Fare.isnull().sum(), test_df.Fare.isnull().sum())
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
test_df.head()
train_df['Fareband'] = pd.qcut(train_df['Fare'], 4)
train_df[['Fareband','Survived']].groupby('Fareband').mean().sort_values(by='Survived', ascending=False)
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31.0), 'Fare'] = 2
    dataset.loc[ dataset['Fare'] > 31.0, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
train_df = train_df.drop('Fareband', axis=1)
combine = [train_df, test_df]

train_df.head(10)
test_df.isnull().sum()
train_df.head()
x_all = train_df.drop(['Survived'], axis=1)
y_all = train_df['Survived']
num_test = 0.3
X_train, X_test, y_train, y_test = train_test_split(x_all, y_all, test_size=num_test, random_state=100)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df['Correlation'] = pd.Series(logreg.coef_[0])
coeff_df.sort_values(by='Correlation', ascending=False)
logreg_prediction = logreg.predict(X_test)
logreg_score=accuracy_score(y_test, logreg_prediction)
print(logreg_score)
X_train = train_df.drop("Survived",axis=1)
y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId",axis=1)
xgboost = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(X_train, y_train)
Y_pred = xgboost.predict(X_test)
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('submission.csv', index=False)
