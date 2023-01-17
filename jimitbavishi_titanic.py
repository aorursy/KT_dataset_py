# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.grid_search import GridSearchCV

from sklearn.cross_validation import cross_val_score

import xgboost as xgb
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
full_df = train_df.append(test_df, ignore_index = True)

print('Train: ',train_df.shape,'Test: ',test_df.shape,'Full: ',full_df.shape)
full_df.tail()
train_df.describe()
train_df.columns.values
train_df.describe(include = ['O'])
train_df.corr(method='pearson')
train_df[['Pclass','Survived']].groupby(['Pclass'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)
train_df[['Sex','Survived']].groupby('Sex').mean().sort_values(by = 'Survived', ascending = False)
train_df[['Parch','Survived']].groupby(['Parch'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)
train_df[['SibSp','Survived']].groupby(['SibSp'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)
g = sns.FacetGrid(train_df,col = 'Survived')

g.map(plt.hist,'Age',bins = 20)
full_df = full_df.drop(['PassengerId','Ticket','Cabin'],axis=1)
full_df.head()
full_df['Sex'] = full_df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
full_df.head()
embarked = pd.get_dummies( full_df.Embarked , prefix='Embarked')
embarked.head()
full_df.Age = full_df.Age.fillna(full_df.Age.mean())

full_df.Fare = full_df.Fare.fillna(full_df.Fare.mean())
full_df.head(10)
title = pd.DataFrame()

# we extract the title from each name

title[ 'Title' ] = full_df[ 'Name' ].map( lambda name: name.split( ',' )[1].split( '.' )[0].strip() )



# a map of more aggregated titles

Title_Dictionary = {

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



# we map each title

title[ 'Title' ] = title.Title.map( Title_Dictionary )

title = pd.get_dummies( title.Title )

#title = pd.concat( [ title , titles_dummies ] , axis = 1 )



title.head()
family = pd.DataFrame()



# introducing a new feature : the size of families (including the passenger)

family[ 'FamilySize' ] = full_df[ 'Parch' ] + full_df[ 'SibSp' ] + 1

family.head()
full_df = full_df.drop(['Name','Survived'],axis=1)

full_df.head()
created_features = pd.concat([embarked, title, family], axis = 1)

created_features.head()
final_df = pd.concat([full_df, created_features], axis = 1)

final_df = final_df.drop(['Embarked'],axis=1)

final_df.head(10)
# Create all datasets that are necessary to train, validate and test models

train_X = final_df[ 0:891 ]

train_y = train_df.Survived

test_X = final_df[ 891: ]

#train_X , valid_X , train_y , valid_y = train_test_split( train_valid_X , train_valid_y , train_size = .7 )



#print (final_df.shape , train_X.shape , valid_X.shape , train_y.shape , valid_y.shape , test_X.shape)

logreg = LogisticRegression()

cross_logreg_score = cross_val_score(logreg, train_X, train_y, cv=10, scoring='accuracy')

logreg_score = cross_logreg_score.mean()

print(logreg_score)
k_range = list(range(1,31))

weight_options = ['uniform','distance']

param_grid = dict(n_neighbors = k_range, weights = weight_options)

knn = KNeighborsClassifier()

knn_grid = GridSearchCV(knn, param_grid, cv = 10, scoring = 'accuracy')

knn_grid.fit(train_X, train_y)
knn_score = knn_grid.best_score_

print(knn_score)

print(knn_grid.best_params_)
svc = SVC()

cross_svc_score = cross_val_score(svc, train_X, train_y, cv=10, scoring='accuracy')

svc_score = cross_svc_score.mean()

print(svc_score)
decision_tree = DecisionTreeClassifier()

cross_decision_tree_score = cross_val_score(decision_tree, train_X, train_y, cv=10, scoring='accuracy')

decision_tree_score = cross_decision_tree_score.mean()

print(decision_tree_score)
estimator_range = list(range(1,21))

param_grid = dict(n_estimators = estimator_range)

random_forest = RandomForestClassifier()

random_forest_grid = GridSearchCV(random_forest, param_grid, cv = 10, scoring = 'accuracy')

random_forest_grid.fit(train_X, train_y)
random_forest_score = random_forest_grid.best_score_

print(random_forest_score)

print(random_forest_grid.best_params_)
gaussian = GaussianNB()

cross_gaussian_score = cross_val_score(gaussian, train_X, train_y, cv=10, scoring='accuracy')

gaussian_score = cross_gaussian_score.mean()

print(gaussian_score)
perceptron = Perceptron()

cross_perceptron_score = cross_val_score(perceptron, train_X, train_y, cv=10, scoring='accuracy')

perceptron_score = cross_perceptron_score.mean()

print(perceptron_score)
linear_svc = LinearSVC()

cross_linear_svc_score = cross_val_score(linear_svc, train_X, train_y, cv=10, scoring='accuracy')

linear_svc_score = cross_linear_svc_score.mean()

print(linear_svc_score)
sgd = SGDClassifier()

cross_sgd_score = cross_val_score(sgd, train_X, train_y, cv=10, scoring='accuracy')

sgd_score = cross_sgd_score.mean()

print(sgd_score)
depth_range = list(range(1,11))

estimator_range = list(range(1,41))

param_grid = dict(max_depth = depth_range, n_estimators = estimator_range)

gbm = xgb.XGBClassifier(learning_rate=0.05)

gbm_grid = GridSearchCV(gbm, param_grid, cv = 10, scoring = 'accuracy')

gbm_grid.fit(train_X, train_y)
gbm_score = gbm_grid.best_score_

print(gbm_score)

print(gbm_grid.best_params_)
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree', 'XGBoost'],

    'Score': [svc_score, knn_score, logreg_score, 

              random_forest_score, gaussian_score, perceptron_score, 

              sgd_score, linear_svc_score, decision_tree_score, gbm_score]})

models.sort_values(by='Score', ascending=False)
model = xgb.XGBClassifier(max_depth=3, n_estimators=36, learning_rate=0.05).fit(train_X, train_y)

test_y = model.predict(test_X)

passenger_id = test_df.PassengerId

predictions = pd.DataFrame({'PassengerId' : passenger_id, 'Survived' : test_y})

predictions.to_csv( 'revised_titanic_pred.csv' , index = False )
predictions.head()