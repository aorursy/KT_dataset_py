def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
import pandas as pd
import os

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
train_df = pd.read_csv('../input/train.csv')
gender_sub = pd.read_csv('../input/gender_submission.csv')
test_df = pd.read_csv('../input/test.csv')
train_df.head()
train_df[['Sex', 'Survived']].groupby(['Sex']).mean()
train_df[['Pclass', 'Survived']].groupby(['Pclass']).mean()
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby('AgeBand').mean().sort_values(by='AgeBand', ascending=True)
del train_df['AgeBand']
combine = [train_df,test_df]
for dataset in combine:
    dataset["Age1"] = 0
    dataset["Age2"] = 0
    dataset["Age3"] = 0
    dataset["Age4"] = 0
    dataset["Age5"] = 0

for dataset in combine:
    dataset.loc[dataset['Age'] <= 16, 'Age1'] = 1
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age2'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age3'] = 1
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age4'] = 1
    dataset.loc[(dataset['Age'] > 64) & (dataset['Age'] <= 80), 'Age5'] = 1
for dataset in combine:
    dataset['Male'] = 0
    dataset['Female'] = 0
    
for dataset in combine:
    dataset.loc[dataset['Sex'] == 'male', 'Male'] = 1
    dataset.loc[dataset['Sex'] == 'female', 'Female'] = 1
for dataset in combine:
    dataset['Pclass1'] = 0
    dataset['Pclass2'] = 0
    dataset['Pclass3'] = 0
    
for dataset in combine:
    dataset.loc[dataset['Pclass'] == 1, 'Pclass1'] = 1
    dataset.loc[dataset['Pclass'] == 2, 'Pclass2'] = 1
    dataset.loc[dataset['Pclass'] == 3, 'Pclass3'] = 1
Test_Pass_Id = test_df['PassengerId']
train_df = train_df.drop(['PassengerId', 'Age', 'Pclass', 'Sex', 'Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis=1)
test_df = test_df.drop(['PassengerId', 'Age', 'Pclass', 'Sex', 'Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis=1)
train_X = train_df.drop(['Survived'], axis=1)
train_Y = train_df['Survived']

# With Cross Validation
lreg = LogisticRegression()
acc_logreg_cross_val = cross_val_score(lreg, train_X,train_Y, cv=5).mean()

# With Single Split
lreg = LogisticRegression()
train_X, test_X, train_Y, test_Y = train_test_split(train_X, train_Y, test_size=0.2, random_state=0)
lreg.fit(train_X, train_Y)
acc_logreg_single_split = lreg.score(test_X, test_Y)

print('Single Split:', acc_logreg_single_split, 'Cross Validation:', acc_logreg_cross_val)
train_X = train_df.drop(['Survived'], axis=1)
train_Y = train_df['Survived']

# With Cross Validation
svc = SVC(C = 0.1, gamma=0.1)
acc_svc_cross_val = cross_val_score(svc, train_X,train_Y, cv=5).mean()

# With Single Split
svc = SVC(C = 0.1, gamma=0.1)
train_X, test_X, train_Y, test_Y = train_test_split(train_X, train_Y, test_size=0.2, random_state=0)
svc.fit(train_X, train_Y)
acc_svc_single_split = svc.score(test_X, test_Y)

print('Single Split:', acc_svc_single_split, 'Cross Validation:', acc_svc_cross_val)
train_X = train_df.drop(['Survived'], axis=1)
train_Y = train_df['Survived']

# With Cross Validation
rand_forest = RandomForestClassifier()
acc_rf_cross_val = cross_val_score(rand_forest, train_X,train_Y, cv=5).mean()

# With Single Split
rand_forest = RandomForestClassifier()
train_X, test_X, train_Y, test_Y = train_test_split(train_X, train_Y, test_size=0.2, random_state=0)
rand_forest.fit(train_X, train_Y)
acc_rf_single_split = rand_forest.score(test_X, test_Y)

print('Single Split:', acc_rf_single_split, 'Cross Validation:', acc_rf_cross_val)
compare_models = pd.DataFrame({'Model': ['Logistic Regression', 'SVC', 'Random Tree'], 'Score' : [acc_logreg_cross_val, acc_svc_cross_val, acc_rf_cross_val]})
compare_models.sort_values(by='Score', ascending=False)
submission = pd.DataFrame({'PassengerId' : Test_Pass_Id, 'Survived' : rand_forest.predict(test_df)})
submission.to_csv('titanic_csv_submission.csv', index=False)