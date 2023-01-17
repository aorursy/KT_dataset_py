# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# classifier models

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV



# modules to handle data

import pandas as pd

import numpy as np



# visualization tools

import matplotlib.pyplot as plt

import seaborn as sns



# ignore warnings

import warnings

warnings.filterwarnings('ignore')
# load data

train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')



# save PassengerId for final submission

passengerId = test.PassengerId



# merge train and test

titanic = train.append(test, ignore_index=True)
# create indexes to separate data later on

train_idx = len(train)

test_idx = len(titanic) - len(test)
titanic.head()
# get info on features

titanic.info()
# PassengerId can be removed from data for now

titanic.drop('PassengerId', axis=1, inplace=True)
# create a new feature to extract title names from the Name column

titanic['Title'] = titanic.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())



# view the newly created feature

titanic.head()
# show count of titles

print("There are {} unique titles.".format(titanic.Title.nunique()))



# show unique titles

print("\n", titanic.Title.unique())
# normalize the titles

normalized_titles = {

    "Capt":       "Officer",

    "Col":        "Officer",

    "Major":      "Officer",

    "Jonkheer":   "Royalty",

    "Don":        "Royalty",

    "Sir" :       "Royalty",

    "Dr":         "Officer",

    "Rev":        "Officer",

    "the Countess":"Royalty",

    "Dona":       "Royalty",

    "Mme":        "Mrs",

    "Mlle":       "Miss",

    "Ms":         "Mrs",

    "Mr" :        "Mr",

    "Mrs" :       "Mrs",

    "Miss" :      "Miss",

    "Master" :    "Master",

    "Lady" :      "Royalty"

}
# map the normalized titles to the current titles

titanic.Title = titanic.Title.map(normalized_titles)



# view value counts for the normalized titles

print(titanic.Title.value_counts())
# group by Sex, Pclass, and Title

grouped = titanic.groupby(['Sex','Pclass', 'Title'])



# view the median Age by the grouped features

grouped.Age.median()
# apply the grouped median value on the Age NaN

titanic.Age = grouped.Age.apply(lambda x: x.fillna(x.median()))



# view changes

titanic.info()


# fill Cabin NaN with U for unknown

titanic.Cabin = titanic.Cabin.fillna('U')
# find most frequent Embarked value and store in variable

most_embarked = titanic.Embarked.value_counts().index[0]



# fill NaN with most_embarked value

titanic.Embarked = titanic.Embarked.fillna(most_embarked)


# fill NaN with median fare

titanic.Fare = titanic.Fare.fillna(titanic.Fare.median())



# view changes

titanic.info()
# view the percentage of those that survived vs. those that died in the Titanic

titanic.Survived.value_counts(normalize=True)
# group by sex

group_by_sex = titanic.groupby('Sex')



# survival rate by sex

group_by_sex.Survived.mean()
# group by passenger class and sex

group_class_sex = titanic.groupby(['Pclass', 'Sex'])



# survival rates by class and sex

group_class_sex.Survived.mean()
# plot by Survivded, Sex

_ = sns.factorplot(x='Sex', col='Survived', data=titanic, kind='count')


# plot by Pclass, Sex, Survived

_ = sns.factorplot(x='Pclass', hue='Sex', col='Survived', data=titanic, kind='count')
# get stats on all the features

titanic.describe()
# size of families (including the passenger)

titanic['FamilySize'] = titanic.Parch + titanic.SibSp + 1
# map first letter of cabin to itself

titanic.Cabin = titanic.Cabin.map(lambda x: x[0])



# view normalized count

titanic.Cabin.value_counts(normalize=True)
titanic.head()
# Convert the male and female groups to integer form

titanic.Sex = titanic.Sex.map({"male": 0, "female":1})
# create dummy variables for categorical features

pclass_dummies = pd.get_dummies(titanic.Pclass, prefix="Pclass")

title_dummies = pd.get_dummies(titanic.Title, prefix="Title")

cabin_dummies = pd.get_dummies(titanic.Cabin, prefix="Cabin")

embarked_dummies = pd.get_dummies(titanic.Embarked, prefix="Embarked")


# concatenate dummy columns with main dataset

titanic_dummies = pd.concat([titanic, pclass_dummies, title_dummies, cabin_dummies, embarked_dummies], axis=1)



# drop categorical fields

titanic_dummies.drop(['Pclass', 'Title', 'Cabin', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)



titanic_dummies.head()
# create train and test data

train = titanic_dummies[ :train_idx]

test = titanic_dummies[test_idx: ]



# convert Survived back to int

train.Survived = train.Survived.astype(int)
# create X and y for data and target values

X = train.drop('Survived', axis=1).values

y = train.Survived.values
# create array for test set

X_test = test.drop('Survived', axis=1).values

#X_test


# create param grid object

log_params = dict(

    C = np.logspace(-5, 8, 15),

    penalty = ['l1', 'l2']

)


# instantiate logistic regressor

log = LogisticRegression()



# load param grid and log model into GridSearcCV

logreg_cv = GridSearchCV(estimator=log, param_grid=log_params, cv=5)



# fit model

logreg_cv.fit(X, y)


# Print the tuned parameters and score

print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_)) 

print("Best score is {}".format(logreg_cv.best_score_))
# predict on test set for submission

log_pred = logreg_cv.predict(X_test)
# create param grid object

forrest_params = dict(

    max_depth = [n for n in range(9, 14)],

    min_samples_split = [n for n in range(4, 11)],

    min_samples_leaf = [n for n in range(2, 5)],

    n_estimators = [n for n in range(10, 60, 10)],

)
# instantiate Random Forest model

forrest = RandomForestClassifier()
# build and fit model

forest_cv = GridSearchCV(estimator=forrest, param_grid=forrest_params, cv=5)

forest_cv.fit(X, y)
# print("Best score: {}".format(forest_model.best_score_))

# print("Optimal params: {}".format(forest_model.best_estimator_))

print("Best score: {}".format(forest_cv.best_score_))

print("Optimal params: {}".format(forest_cv.best_estimator_))
# random forrest prediction on test set

forrest_pred = forest_cv.predict(X_test)

print(forrest_pred)
# dataframe with predictions

kaggle = pd.DataFrame( {'PassengerId': passengerId, 'Survived': forrest_pred} )
# ensure df is in the right format

kaggle.head(10)


# save to csv

filename = 'submit.csv'

kaggle.to_csv(filename, index=False)