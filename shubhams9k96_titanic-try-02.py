import matplotlib.pyplot as plt

import cufflinks as cf

import sklearn

from sklearn import svm, preprocessing 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random

import plotly.graph_objs as go

import plotly.plotly as py

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

import seaborn as sns

import plotly.figure_factory as ff

from sklearn.preprocessing import OneHotEncoder

%matplotlib inline

import warnings

warnings.filterwarnings("ignore")

import os

print(os.listdir("../input"))
train_df = pd.read_csv('../input/train.csv')

train_df.info()

test_df = pd.read_csv('../input/test.csv')
print(train_df.shape)

print(test_df.shape)
def rm_null_fare(colnames):

    fare = colnames[0]

    pclass = colnames[1]

    if pd.isnull(fare):

        if pclass == 1:

            return 84

        elif pclass == 2:

            return 21

        else:

            return 14

    else:

        return fare

def rm_null_age(colnames):

    age = colnames[0]

    pclass = colnames[1]

    if pd.isnull(age):

        if pclass == 1:

            return 38

        elif pclass == 2:

            return 29

        else:

            return 25

    else:

        return age



def univariate_barplots(data, col1, col2='Survived', top=False):

    # Count number of zeros in dataframe python: https://stackoverflow.com/a/51540521/4084039

    temp = pd.DataFrame(data.groupby(col1)[col2].agg(lambda x: x.eq(1).sum())).reset_index().sort_values(by = col1, ascending = True)



    # Pandas dataframe grouby count: https://stackoverflow.com/a/19385591/4084039

    temp['total'] = pd.DataFrame(data.groupby(col1)[col2].agg({'total':'count'})).reset_index()['total']

    temp['Avg Survival Rate']   = pd.DataFrame(data.groupby(col1)[col2].agg({'Avg':'mean'})).reset_index()['Avg']

    

    temp.sort_values(by=['Avg Survival Rate'],inplace=True, ascending=False)

    

    if top:

        temp = temp[0:top]

    

#     stack_plot(temp, xtick=col1, col2=col2, col3='total')

    print(temp)

#     print("="*50)

#     print(temp.tail(5))
print(train_df.columns)
train_df['Age'] = train_df[['Age', 'Pclass']].apply(rm_null_age, axis = 1)

test_df['Age']  = test_df[['Age', 'Pclass']].apply(rm_null_age, axis = 1)
train_df['Fare'] = train_df[['Fare', 'Pclass']].apply(rm_null_fare, axis = 1)

test_df['Fare']  = test_df[['Fare', 'Pclass']].apply(rm_null_fare, axis = 1)
train_df['Name_title'] = train_df['Name'].str.split('[,.]').str.get(1)

test_df['Name_title']  = test_df['Name'].str.split('[,.]').str.get(1)
def agg_names(name):

    name = name.replace(" ", '')



    if name in ['Mlle', 'Lady', 'Mme', 'Ms', 'theCountess' ]:

        return 'Lady'    

    if name in ['Rev', 'Jonkheer', 'Don','Capt' ]:

        return 'Crew'    

    if name in ['Col', 'Major']:

        return 'Major'    

    if name in ['Mrs', 'Miss']:

        return 'Mrs'

    if name in ['Sir', 'Master', 'Dr', 'Mr']:

        return name    

    else:

        return 'Others'
train_df['Name_title'] = train_df['Name_title'].apply(agg_names)

test_df['Name_title'] = test_df['Name_title'].apply(agg_names)
univariate_barplots(train_df, 'Name_title')
test_df['Name_title'].value_counts()
train_df.drop('Name', axis = 1, inplace = True)

test_df.drop('Name', axis = 1, inplace = True)
train_df['Cabin'].fillna('Not Available', inplace = True)

test_df['Cabin'].fillna('Not Available', inplace = True)
def fix_cabin(cabin):

    if cabin == 'Not Available':

        return cabin

    else:

        return cabin[0]
train_df['Cabin'] = train_df['Cabin'].apply(fix_cabin)

test_df['Cabin']  = test_df['Cabin'].apply(fix_cabin)
train_df['familynum'] = train_df['SibSp'] + train_df['Parch'] + 1

test_df['familynum'] = test_df['SibSp'] + test_df['Parch'] + 1
def countfamily(num):

    if num == 1:

        return 1

    else:

        return 0
train_df['Isalone'] = train_df['familynum'].apply(countfamily)

test_df['Isalone'] = test_df['familynum'].apply(countfamily)
univariate_barplots(train_df, 'familynum')
univariate_barplots(train_df, 'Isalone')
train_df.drop(['Ticket'], axis = 1, inplace = True)

test_df.drop(['Ticket'], axis = 1, inplace = True)
train_df['Embarked'].fillna('S', inplace=True)

test_df['Embarked'].fillna('S', inplace=True)
univariate_barplots(train_df, 'Embarked')
train_df.head()
X_train = train_df.drop(['PassengerId', 'Survived'], axis = 1)

y_train = train_df['Survived']

X_test  = test_df.drop(['PassengerId'], axis = 1)

print(X_train.shape)

print(y_train.shape)

print(X_test.shape)
X_train.head()
enc = OneHotEncoder(handle_unknown='ignore')

  
# Ref: https://medium.com/@vaibhavshukla182/how-to-solve-mismatch-in-train-and-test-set-after-categorical-encoding-8320ed03552f

X_train['train'] = 1

X_test['train'] = 0
combined = pd.concat([X_train, X_test])
combined.head()
dummies = pd.get_dummies(combined[['Pclass', 'Sex', 'Cabin', 'Embarked', 'Name_title']])
dummies.head()
combined = pd.concat([dummies, combined[['Age', 'SibSp', 'Parch', 'Fare', 'familynum', 'Isalone', 'train']],   ], axis =1)
combined.head()
X_train = combined[combined['train']== 1]

X_test  = combined[combined['train']== 0]

X_train.drop(['train'], axis = 1, inplace = True)

X_test.drop(['train'], axis = 1, inplace = True)
print(X_train.shape)

print(y_train.shape)

print(X_test.shape)
import time

from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.ensemble import (RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier, VotingClassifier)

from sklearn.metrics import accuracy_score



dict_clf = {}
paramgrid = {

    'n_estimators':      [100, 200, 500, 750, 1000],

    'criterion':         ['gini', 'entropy'],

    'max_features':      ['auto', 'log2'],

    'min_samples_leaf':  list(range(2, 7))

}

GS = GridSearchCV(RandomForestClassifier(random_state=77),

                  paramgrid,

                  cv=4)

t0 = time.time()

GS.fit(X_train, y_train)

t = time.time() - t0

best_clf = GS.best_estimator_

best_params = GS.best_params_

best_score = GS.best_score_

name = 'RF'

best_clf.fit(X_train, y_train)

acc_eval = accuracy_score(y_train, best_clf.predict(X_train))

dict_clf[name] = {

    'best_par': best_params,

    'best_clf': best_clf,

    'best_score': best_score,

    'score_eval': acc_eval,

    'fit_time': t,

}
# 2. GradientBoosting

paramgrid = {

    'n_estimators':      [100, 200, 500, 750, 1000],

    'max_features':      ['auto', 'log2'],

    'min_samples_leaf':  list(range(2, 7)),

    'loss' :             ['deviance', 'exponential'],

    'learning_rate':     [0.05, 0.1, 0.2],

}

GS = GridSearchCV(GradientBoostingClassifier(random_state=77),

                  paramgrid,

                  cv=4)

t0 = time.time()

GS.fit(X_train, y_train)

t = time.time() - t0

best_clf = GS.best_estimator_

best_params = GS.best_params_

best_score = GS.best_score_

name = 'GB'

best_clf.fit(X_train, y_train)

acc_eval = accuracy_score(y_train, best_clf.predict(X_train))

dict_clf[name] = {

    'best_par': best_params,

    'best_clf': best_clf,

    'best_score': best_score,

    'score_eval': acc_eval,

    'fit_time': t,

}
# 3. AdaBoost

paramgrid = {

    'n_estimators':  [100, 200, 500, 750, 1000],

    'learning_rate': [0.05, 0.1, 0.5, 1, 2]

}

GS = GridSearchCV(AdaBoostClassifier(random_state=77),

                  paramgrid,

                  cv=4)

t0 = time.time()

GS.fit(X_train, y_train)

t = time.time() - t0

best_clf = GS.best_estimator_

best_params = GS.best_params_

best_score = GS.best_score_

name = 'ADB'

best_clf.fit(X_train, y_train)

acc_eval = accuracy_score(y_train, best_clf.predict(X_train))

dict_clf[name] = {

    'best_par': best_params,

    'best_clf': best_clf,

    'best_score': best_score,

    'score_eval': acc_eval,

    'fit_time': t,

}
dict_clf
voting_clf = VotingClassifier(estimators = [ ('GB', dict_clf['GB']['best_clf']),

                                             ('RF', dict_clf['RF']['best_clf']),

                                             ('ADB',dict_clf['ADB']['best_clf'])], voting = 'soft', n_jobs = -1)

voting_clf.fit(X_train, y_train)
nb_pred = voting_clf.predict(X_test)

res = test_df[['PassengerId']]

res['Survived'] = nb_pred

res.to_csv('Prediction_gender.csv', index=False)