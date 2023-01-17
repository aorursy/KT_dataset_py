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
# import modules

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import MinMaxScaler

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from matplotlib import pyplot as plt

from sklearn.svm import SVC

import seaborn as sns
# load dataset

trainset = pd.read_csv('/kaggle/input/titanic/train.csv')

testset = pd.read_csv('/kaggle/input/titanic/test.csv')

# store testset ids for submission

test_ids = testset['PassengerId']

# drop columns that may not be predictive

trainset.drop(['PassengerId','Name'], axis=1, inplace=True)

testset.drop(['PassengerId','Name'], axis=1, inplace=True)

# split features and targets

X_train = trainset.iloc[:,1:]

y_train = trainset.iloc[:,0]

X_test = testset

# store name of the variables

variable_names = X_train.columns

# quick view

X_train.head()
# percentage of missing values per variable 

pd.DataFrame(X_train.isnull().mean() * 100, columns=['Missing_percentage'])
# histograms of numerical variables

plt.figure(), X_train.hist(figsize=(10,10))

# histograms of categorical variables

plt.figure(figsize=(5,5)), X_train['Sex'].value_counts().plot(kind='bar'), plt.title('Sex')

plt.figure(figsize=(5,5)), X_train['Ticket'].value_counts().plot(kind='bar'), plt.title('Ticket')

plt.figure(figsize=(5,5)), X_train['Cabin'].value_counts().plot(kind='bar'), plt.title('Cabin')

plt.figure(figsize=(5,5)), X_train['Embarked'].value_counts().plot(kind='bar'), plt.title('Embarked')
# create list of variables for each inputation type

age_feature = ['Age']

cabin_feature = ['Cabin']

embarked_feature = ['Embarked']

# create pipelines to perform the inputation of missing data

age_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='mean'))])

cabin_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='constant', fill_value='missing'))])

embarked_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='constant', fill_value='S'))])

# ensemble the pipelines 

preprocessor = ColumnTransformer(transformers=[

    ('age_imputer', age_transformer, age_feature),

    ('cabin_imputer', cabin_transformer, cabin_feature),

    ('embarked_imputer', embarked_transformer, embarked_feature)

], remainder='passthrough')

# fit the preprocessor with training data

preprocessor.fit(X_train)

# apply imputation on the training set and test set

X_train = preprocessor.transform(X_train)

X_test = preprocessor.transform(X_test)

# check fit preprocessor

preprocessor.transformers_
# list with the indexes of the columns we did not preprocess

unprocessed_cols = [variable_names[i] for i in [0, 1, 3, 4, 5, 6]]

# capture the data to a dataframe

X_train = pd.DataFrame(X_train, columns=age_feature + cabin_feature + embarked_feature + unprocessed_cols)

X_test = pd.DataFrame(X_test, columns=age_feature + cabin_feature + embarked_feature + unprocessed_cols)

# we should now have no missing data

print('missing data in the training set:\n', X_train.isnull().mean() * 100)

# quick view

X_train.head()
# first, let us transform the type of the numerical features from "object" to "int"

X_train[['Age','Pclass','SibSp','Parch','Fare']] = X_train[['Age','Pclass','SibSp','Parch','Fare']].apply(pd.to_numeric)

X_test[['Age','Pclass','SibSp','Parch','Fare']] = X_test[['Age','Pclass','SibSp','Parch','Fare']].apply(pd.to_numeric)

# before applying one hot join train and test sets to create the same number of features

X = pd.concat([X_train, X_test], axis=0)

# get one hot encoding

X = pd.get_dummies(X)

# split training and test sets

X_train = X.iloc[:X_train.shape[0]]

X_test = X.iloc[X_train.shape[0]:]

# check new dimensions

print('Training set dimension: ', X_train.shape)

print('Test set dimension: ', X_test.shape)
# instance classifier

rf = RandomForestClassifier(criterion = "gini", 

                                       min_samples_leaf = 1, 

                                       min_samples_split = 10,   

                                       n_estimators=100, 

                                       max_features='auto', 

                                       oob_score=True, 

                                       random_state=1, 

                                       n_jobs=-1)

# perform cross validation

scores = cross_val_score(rf, X_train, y_train, cv=10, scoring = "accuracy")

# print cross validation scores

print("Scores:", scores)

print("Mean:", scores.mean())

print("Standard Deviation:", scores.std())
# check test missing data

X_test.isnull().mean()

# input missing data

X_test['Fare'] = X_test[['Fare']].fillna(X_test['Fare'].mean())
# fit classifier

rf.fit(X_train, y_train)

# get predictions for submission

file = pd.DataFrame({'PassengerId': test_ids, 'Survived': rf.predict(X_test)}).to_csv('my_submission.csv', index=False)