# This Python 3 environment comes with many helpful analytics libraries installed



# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

sns.set() # setting seaborn default for plots







%matplotlib inline
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_df.columns
train_df.head()
train_df.describe()
train_df.info()
test_df.info()
train_df.isnull().sum()
test_df.isnull().sum()
train_df['Survived'].value_counts()
def bar_chart(feature):

    survived = train_df[train_df['Survived']==1][feature].value_counts()

    dead = train_df[train_df['Survived']==0][feature].value_counts()

    df = pd.DataFrame([survived,dead])

    df.index = ['Survived','Dead']

    df.plot(kind='bar',stacked=True, figsize=(10,5))
bar_chart('Sex')
train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
bar_chart('Pclass')
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
g = sns.FacetGrid(train_df, col='Survived')

g.map(plt.hist, 'Age', bins=20)
train_test_data = [train_df, test_df] # combining train and test dataset



for dataset in train_test_data:

    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

train_df['Title'].value_counts()
test_df['Title'].value_counts()
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 

                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,

                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }

for dataset in train_test_data:

    dataset['Title'] = dataset['Title'].map(title_mapping)
bar_chart('Title')
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

train_df.columns
X_train_df = train_df.drop(columns=['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'])

X_test_df = test_df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
y_train_df = train_df['Survived']

y_test_df = test_df['PassengerId']
X_train_df.isnull().sum()
X_test_df.isnull().sum()
age_med = train_df.Age.median()

X_train_df.Age.fillna(age_med, inplace=True)

X_test_df.Age.fillna(age_med, inplace=True)
mod = X_train_df.Embarked.value_counts().argmax()

X_train_df.Embarked.fillna(mod, inplace=True)
fare_med = train_df.Fare.median()

X_test_df.Fare.fillna(fare_med, inplace=True)
X_train_df.isnull().sum()
X_test_df.isnull().sum()
X_train_df.columns
X_train_df.replace({"male": 0, "female": 1}, inplace=True)

X_test_df.replace({"male": 0, "female": 1}, inplace=True)

X_train_df.replace({"S": 0, "C": 1, "Q": 2}, inplace=True)

X_test_df.replace({"S": 0, "C": 1, "Q": 2}, inplace=True)
X_train_df.head()
X_train_df = pd.get_dummies(X_train_df, columns=['Pclass', 'Embarked'], drop_first=True)

X_test_df = pd.get_dummies(X_test_df, columns=['Pclass', 'Embarked'], drop_first=True)

X_train_df.head()
X_train_df.shape, X_test_df.shape
from sklearn.preprocessing import MinMaxScaler

sc_X = MinMaxScaler()

X_train_sc = sc_X.fit_transform(X_train_df)

X_test_sc = sc_X.transform(X_test_df)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier
logi_clf = LogisticRegression(random_state=0)

logi_parm = {"penalty": ['l1', 'l2'], "C": [0.1, 0.5, 1, 5, 10, 50]}



svm_clf = SVC(random_state=0)

svm_parm = {'kernel': ['rbf', 'poly'], 'C': [0.1, 0.5, 1, 5, 10, 50], 'degree': [3, 5, 7], 

            'gamma': ['auto', 'scale']}



dt_clf = DecisionTreeClassifier(random_state=0)

dt_parm = {'criterion':['gini', 'entropy']}



knn_clf = KNeighborsClassifier()

knn_parm = {'n_neighbors':[5, 10, 15, 20], 'weights':['uniform', 'distance'], 'p': [1,2]}



gnb_clf = GaussianNB()

gnb_parm = {'priors':['None']}



clfs = [logi_clf, svm_clf, dt_clf, knn_clf]

params = [logi_parm, svm_parm, dt_parm, knn_parm] 
clfs_opt = []

clfs_best_scores = []

clfs_best_param = []

for clf_, param in zip(clfs, params):

    clf = RandomizedSearchCV(clf_, param, cv=5)

    clf.fit(X_train_sc, y_train_df)

    clfs_opt.append(clf)

    clfs_best_scores.append(clf.best_score_)

    clfs_best_param.append(clf.best_params_)
max(clfs_best_scores)
arg = np.argmax(clfs_best_scores)

clfs_best_param[arg]
clf = clfs_opt[arg]
pred = clf.predict(X_test_sc)
cols = ['PassengerId', 'Survived']

submit_df = pd.DataFrame(np.hstack((y_test_df.values.reshape(-1,1),pred.reshape(-1,1))), 

                         columns=cols)
submit_df.to_csv('subition.csv', index=False)
submit_df.head()