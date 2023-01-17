# Import neccesary libraries

# Numpy - linear algebra

# Pandas - Data manipulation

# os - operating system commands

# re - regular expressions

# sklearn - machine learning

import numpy as np

import pandas as pd

import seaborn as sns

import os

import re

from sklearn import preprocessing

from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier



train_data = pd.read_csv('/kaggle/input/train.csv')

test_data = pd.read_csv('/kaggle/input/test.csv')
# Support functions

def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""
full_data = [train_data, test_data]



for idx, dataset in enumerate(full_data):

    # Pclass variable - categorical / integer

    # We want to get dummy variables for the Pclass attribute and remove afterwards one created variable.

    # Why? Because we want to avoid basic mathematical issues. For instance in regression, ordinary least squares (OLS)

    # solves it mathematical issues by optimizing the model params to an overfitted system. For this,

    # the invers of the observation coefficients are needed. By having dummy variables we overcome the issue and ambiguity.

    # Another reason is the distribution of categorical data, several classifiers perform better on standard-

    # ized data and it is similar to the logical internal representation inside our computers.

    dataset = pd.concat([pd.get_dummies(dataset['Pclass'], prefix='Pclass'), dataset], axis=1)

    dataset = dataset.drop(['Pclass', 'Pclass_3'], axis=1)



    # Name variable - text / string

    # I've taken this example for the probably most famous titanic kernel from @Anisotropic:

    # It shows very well the idea of an example how to deal with names. Luckily he has dealt already with

    # all form of name prefixes, which might predict whether the person survived or not.

    # We look for rare name prefixes and replace all different spellings into a consistent view.

    dataset['Title'] = dataset['Name'].apply(get_title)

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

    # Here we transform the different titles to our found titles. In case the person does not have

    # a title, we give him the number 0. Finally we drop our name column.

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

    dataset = dataset.drop(['Name'], axis=1)



    # Sex - text / string

    # Similar to what we have done with our Pclass variable. First we transfrom the text into integers

    # by getting dummy variables for our observations. Then we remove the ambiguity of information

    # for a binary variable represented through two binary columns. Finally we drop the Sex information

    # as well.

    dataset = pd.concat([pd.get_dummies(dataset['Sex'], prefix='Sex'), dataset], axis=1)

    dataset = dataset.drop(['Sex', 'Sex_male'], axis=1)

    

    # Age - categorical / integer

    # Above we could see that age has approx. 100 missing values. We need to impute  missing values but

    # we do not want to falsify our system. A standard approach is to insert the mean but here we want to

    # use the median in order to avoid shifiting the age distribution to a certain direction only in order

    # to keep the mean stable. We have decided to go for 5 buckets.

    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)

    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5, labels=[1, 2, 3, 4, 5])

    dataset = dataset.drop(['Age'], axis=1)



    # Embarked - categorical / string

    # We replace our missing values with the mode of its values - this is 'S' in our case. We then again

    # get our dummy variables and transform remove a redundant column.

    #dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)

    #dataset = pd.concat([pd.get_dummies(dataset['Embarked'], prefix='Embarked'), dataset], axis=1)

    dataset = dataset.drop(['Embarked'], axis=1)

    

    # Parch - integer

    # SibSp - integer

    # From our siblings and parents information we create a combined variable and a 'is alone' flag.

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    dataset['IsAlone'] = 1

    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0

    dataset = dataset.drop(['SibSp', 'Parch'], axis=1)

    

    # Fare - continous / float

    # Similar to our ages we bucket our data into 4 bins but here we want to use quantile based bins

    # since the distribution of our fare data will have a lot of "normal" priced tickets and only a

    # few very high priced tickets.

    #dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)

    #dataset['FareBin'] = pd.qcut(dataset['Fare'], 4, labels=[1, 2, 3, 4])

    dataset = dataset.drop(['Fare'], axis=1)

    

    # Cabin - String

    # The titanic has been divided vertically into decks - this information we want to extract and check

    # whether this had influence on whether a person has survived or not. 

    #dataset['Deck'] = dataset['Cabin'].str.slice(0,1)

    #dataset['Deck'] = dataset['Deck'].fillna("N")

    #dataset['Deck'] = dataset['Deck'].astype('category')

    #dataset['Deck'] = dataset['Deck'].cat.codes

    dataset = dataset.drop(['Cabin'], axis=1)

    

    # Drop unnesscary features

    dataset = dataset.drop(['Ticket'], axis=1)

    

    if idx == 0:

        X_train, y_train = dataset.loc[:, dataset.columns != 'Survived'], dataset.loc[:, dataset.columns == 'Survived']



    else:

        submission_id = dataset['PassengerId']

        X_test = dataset 

# Change here through the classifiers you want to try.

kf = KFold(n_splits=5, random_state = 0)

clf = clf = RandomForestClassifier(n_estimators=400, max_depth=4, min_samples_split=4, random_state=0)

for train_index, test_index in kf.split(X_train):

    __kf_X_train, __kf_X_test = X_train.values[train_index], X_train.values[test_index]

    __kf_y_train, __kf_y_test = y_train.values[train_index].ravel(), y_train.values[test_index].ravel()

    clf.fit(__kf_X_train, __kf_y_train)

    print(accuracy_score(__kf_y_test, clf.predict(__kf_X_test)))
# Prepare the submission file by stacking passenger ids and predictions.

y_pred = pd.Series(clf.predict(X_test))

submission = pd.concat([submission_id, y_pred], axis=1)

submission = submission.rename(columns={0:'Survived'})

submission.to_csv('submisson.csv', index=False)