# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.cross_validation import train_test_split

from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import LinearSVC,SVC

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.cross_validation import cross_val_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(color_codes=True)

from scipy import stats

from scipy.stats import norm, skew #for some statistics

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )

test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
train.head()
train.isnull().sum(axis=0)
sns.distplot(train['Fare'])
test['Sex'].value_counts()
train['Survived'].plot(kind='hist')
test_id = test['PassengerId']

target = train['Survived']
train["Fare"].fillna(train["Fare"].median(), inplace=True)

test["Fare"].fillna(test["Fare"].median(), inplace=True)
train["Embarked"].fillna(train['Embarked'].mode()[0], inplace=True)

test["Embarked"].fillna(test['Embarked'].mode()[0], inplace=True)
train["Age"].fillna(train["Age"].median(), inplace=True)

test["Age"].fillna(test["Age"].median(), inplace=True)
# train['Age'] = train['Age'].astype(int)

# test['Age'] = test['Age'].astype(int)
import string

def substrings_in_string(big_string, substrings):

    for substring in substrings:

        if big_string.find(substring) != -1:

            return substring

    return np.nan
#replacing all titles with mr, mrs, miss, master

def replace_titles(x):

    title=x['Title']

    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:

        return 'Mr'

    elif title in ['Countess', 'Mme']:

        return 'Mrs'

    elif title in ['Mlle', 'Ms']:

        return 'Miss'

    elif title =='Dr':

        if x['Sex']=='Male':

            return 'Mr'

        else:

            return 'Mrs'

    else:

        return title
title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',

                    'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',

                    'Don', 'Jonkheer']
train['Title']=train['Name'].map(lambda x: substrings_in_string(x, title_list))

test['Title']=test['Name'].map(lambda x: substrings_in_string(x, title_list))
train['Title']=train.apply(replace_titles, axis=1)

test['Title']=test.apply(replace_titles, axis=1)
# train['Fare'] = train['Fare'].astype(int)

# test['Fare'] = test['Fare'].astype(int)

train['Family_Size']=train['SibSp']+train['Parch']

test['Family_Size']=test['SibSp']+test['Parch']
train['Age*Class']=train['Age']*train['Pclass']

test['Age*Class']=test['Age']*test['Pclass']
train['Fare_Per_Person']=train['Fare']/(train['Family_Size']+1)

test['Fare_Per_Person']=test['Fare']/(test['Family_Size']+1)
train["Embarked"].fillna(0, inplace=True)

test["Embarked"].fillna(0, inplace=True)
# cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']

# train['Deck']=train['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))

# test['Deck']=test['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))
# One-hot encoding

from sklearn.preprocessing import LabelEncoder



lb_make = LabelEncoder()

train["Embarked"] = lb_make.fit_transform(train["Embarked"])

train["Sex"] = lb_make.fit_transform(train["Sex"])

train["Title"] = lb_make.fit_transform(train["Title"])



test["Embarked"] = lb_make.fit_transform(test["Embarked"])

test["Sex"] = lb_make.fit_transform(test["Sex"])

test["Title"] = lb_make.fit_transform(test["Title"])
train = train.drop(['Name', 'PassengerId', 'Survived', 'Ticket', 'Cabin'], axis=1)

test = test.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1)
#correlation matrix

corrmat = train.corr()

f, ax = plt.subplots(figsize=(5, 5))

sns.heatmap(corrmat, vmax=.8, square=True);
train.head()
# from catboost import CatBoostClassifier



# model_catboost = CatBoostClassifier(rsm=1,eval_metric='Accuracy',learning_rate=0.3,verbose=True, iterations=60,depth=14)



# model_catboost.fit(train, target, cat_features=[0,  1,  3,  4])



# pred = model_catboost.predict(test)
import xgboost as xgb
#model = xgb.XGBClassifier(max_depth=7, n_estimators=500, learning_rate=0.01)
# Voting Classifier

vote_est = [

    ('ada', AdaBoostClassifier()),

    ('bc', BaggingClassifier()),

    ('etc',ExtraTreesClassifier()),

    ('gbc', GradientBoostingClassifier()),

    ('rfc', RandomForestClassifier()),



    ('gpc', GaussianProcessClassifier()),

    

    ('lr', LogisticRegressionCV()),

    

    ('bnb', BernoulliNB()),

    ('gnb', GaussianNB()),

    

    ('knn', KNeighborsClassifier()),

    

    ('svc', SVC(probability=True)),

    

    ('xgb', xgb.XGBClassifier())



]



model = VotingClassifier(estimators = vote_est , voting = 'soft')
from sklearn.cross_validation import cross_val_score

from sklearn.metrics import accuracy_score, make_scorer

print(cross_val_score(model, train, target, cv=5, scoring=make_scorer(accuracy_score)))
model.fit(train,target)

pred = model.predict(test)
submission = pd.DataFrame({

        "PassengerId": test_id,

        "Survived": pred

    })

submission.to_csv('titanic_result_pandas.csv', index=False)
# submission.head()