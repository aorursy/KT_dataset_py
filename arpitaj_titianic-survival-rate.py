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
# get the imports

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

%matplotlib inline
#loading the dataframes with input data

# training data

input_train = pd.read_csv('/kaggle/input/train.csv')

# test data

input_test = pd.read_csv('/kaggle/input/test.csv')
input_test.head()

input_train.head()
input_train.dtypes

input_test.dtypes
input_train.isnull().sum()

input_test.isnull().sum()
# Replacing NaN values with N, when cabin is unavailable

input_train.Cabin.fillna("N", inplace=True)

input_test.Cabin.fillna("N", inplace=True)

# the number of missing values for column Embarked is 2, drop these observations from the dataset

input_train.drop( input_train[ input_train['Embarked'].isnull() ].index , inplace=True)

# Replace age null values by median age for the class

grouped_df_train = input_train.groupby(['Sex','Pclass']) 

grouped_df_test = input_test.groupby(['Sex', 'Pclass'])

grouped_df_train.Age.median()

grouped_df_test.Age.median()

input_train.Age = grouped_df_train.Age.apply(lambda x: x.fillna(x.median()))

input_test.Age = grouped_df_test.Age.apply(lambda x: x.fillna(x.median()))

# sanity check on null values in the dataset

#input_train.isnull().sum()
# exploring the 'Name' column to create categorical variables

#print(input_train['Name'].head())

# as seen, Name variable has a title associated with it, this can give information of the passengers and should be used as a feature

input_train['Title'] = [i.split(",")[1].split(".")[0].strip() for i in input_train['Name']]

print(input_train['Title'].head())

input_test['Title'] = [i.split(",")[1].split(".")[0].strip() for i in input_test['Name']]

print(input_test['Title'].head())
# plotting this new feature

title_plot = sns.countplot(x="Title",data=input_train)

title_plot = plt.setp(title_plot.get_xticklabels(), rotation = 90)
# binning the title values for test and train data

# o -> young male, 1 -> young female, 1 -> adult male, 2 -> adult female, 3 -> rare title

input_train['Title'] = input_train['Title'].replace(['Don','Rev', 'Dr', 'Major', 'Col', 'Capt', 'the Countess','Jonkheer'], 'Other')

input_train['Title'] = input_train['Title'].map({"Master":0, "Miss":1, "Mme":1, 'Ms':1, "Mlle":1, "Mrs":3, "Mr":2, 'Lady':3,'Sir':2, "Other":3})



input_test['Title'] = input_test['Title'].replace(['Don','Rev', 'Dr', 'Major', 'Col', 'Capt', 'the Countess','Jonkheer'], 'Other')

input_test["Title"] = input_test["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":3, "Mr":2, 'Lady':3,'Sir':2, "Other":2})

# Distribution of the title feature

title_plot = sns.countplot(x="Title",data=input_train)

title_plot = plt.setp(title_plot.get_xticklabels(), rotation = 90)

#print(input_train['Title'].count())



title_plot = sns.countplot(x="Title",data=input_test)

title_plot = plt.setp(title_plot.get_xticklabels(), rotation = 90)

#print(input_test['Title'].count())
# Two features -> SibSp, which denotes sibling and Parch -> which denotes parents/children can be combined to form a feature 'family'

# formula -> sibSp+Parch+1(individual)

input_train['Family'] = input_train['SibSp'] + input_train['Parch'] + 1

input_test['Family'] = input_test['SibSp'] + input_test['Parch'] + 1

# visualizing the correlation between all the input variables and the probability of survival

input_train['Survived_Dead'] = input_train['Survived'].apply(lambda x : 'Survived' if x == 1 else 'Dead')

plt.figure(figsize= (10,5))

sns.heatmap(input_train.corr(),annot=True, linewidth = 0.5, cmap='Reds')
input_train.drop([ 'Ticket', 'Name'], axis=1, inplace = True)

input_test.drop(['Ticket', 'Name'], axis=1, inplace = True)
# Visualizing pclass,sex, embarked columns with survival

sns.countplot('Survived_Dead', data = input_train)
sns.countplot('Survived_Dead', hue = 'Sex', data = input_train)
sns.countplot('Survived_Dead', hue = 'Pclass', data = input_train)
sns.barplot(x = 'Pclass', y = 'Fare', data = input_train)
sns.countplot('Survived_Dead', hue = 'Embarked', data = input_train)
sns.countplot('Survived_Dead', hue = 'Title', data = input_train)
# detecting the outliers in the numeric variables -> Parch, SibSp, Fare, Age

#sns.boxplot(x= input_train['Age'])

from collections import Counter

def delete_outliers(input_df, feature):

    Q1 = np.percentile(input_df[feature],25)

    Q3 = np.percentile(input_df[feature],75)

    IQR = Q3 - Q1

    step = 1.5*IQR

    indexes = input_df[(input_df[feature] < Q1 - step) | (input_df[feature] > Q3 + step )].index

    return indexes



index_list = []

for feature in ['Parch', 'SibSp', 'Fare', 'Age']:

    index_list.extend(delete_outliers(input_train, feature))

outliers = Counter(index_list)        

outliers_remove = list( k for k, v in outliers.items() if v > 2 )

print(input_train.loc[outliers_remove].head(), len(outliers_remove))

input_train.info()
# Transforming categoricals

sex_map = {'male': 0, 'female':1}

input_train['sex_dummy'] = input_train['Sex'].map(sex_map)



sex_map = {'male': 0, 'female':1}

input_test['sex_dummy'] = input_test['Sex'].map(sex_map)



embarked_map = {'Q':0, 'C': 1, 'S': 2 }

input_train['embarked_dummy'] = input_train['Embarked'].map(embarked_map)



embarked_map = {'Q':0, 'C': 1, 'S': 2 }

input_test['embarked_dummy'] = input_test['Embarked'].map(embarked_map)





input_train.drop(['Embarked', 'Sex'], axis=1, inplace = True)

input_test.drop(['Embarked', 'Sex'], axis=1, inplace = True)
# Transforming numerical variables

input_train.info()
input_train["has_cabin"] = [0 if i == 'N'else 1 for i in input_train.Cabin]

input_test["has_cabin"] = [0 if i == 'N'else 1 for i in input_test.Cabin]



input_train.drop(['Cabin', 'Survived_Dead'], axis=1, inplace = True)

input_test.drop(['Cabin'], axis=1, inplace = True)

from sklearn.preprocessing import StandardScaler

y_train = input_train['Survived']

input_train.drop(['Survived'], axis=1, inplace = True)

X_train = input_train

X_test = input_test[input_train.columns]

# replacing the fare in test set witht the mean value 

X_test['Fare'].fillna(X_test['Fare'].mean(), inplace=True)

# replacing the Nan value with the most frequent label from the title column and changing the type to int

X_test['Title'] = X_test['Title'].fillna(0.0).astype(int)

X_test2 = X_test.copy()

X_test.drop(['PassengerId'], axis=1, inplace = True)

X_train.drop(['PassengerId'], axis=1, inplace = True)

# scaling the input to the models

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)

# imports for machine learning models

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, log_loss



# initialize dictionary to record scores

accuracy_scores = {}
# Applying machine learning algorithms to the training dataset and recording the scores obtained

rfc = RandomForestClassifier(random_state = 0).fit(X_train, y_train)

accuracy_scores['RandomForestClassifier'] = rfc.score(X_train, y_train)

print('Random Forest Classifier score = {0}'.format(rfc.score(X_train, y_train)))

print('************************************************')

lr = LogisticRegression(random_state = 0).fit(X_train, y_train)

accuracy_scores['LogisticRegression'] = lr.score(X_train, y_train)

print('Logistic Regression score = {0}'.format(lr.score(X_train, y_train)))

print('************************************************')

svc = SVC(random_state = 0, kernel = 'rbf').fit(X_train, y_train)

accuracy_scores['SVC_radial_basis'] = svc.score(X_train, y_train)

print('SVC radial basis score = {0}'.format(svc.score(X_train, y_train)))

print('************************************************')

knn = KNeighborsClassifier(n_neighbors = 3).fit(X_train, y_train)

accuracy_scores['KNeighborsClassifier'] = knn.score(X_train, y_train)

print('KNeighbors Classifier score = {0}'.format(knn.score(X_train, y_train)))

print('************************************************')

gnb = GaussianNB().fit(X_train, y_train)

accuracy_scores['GaussianNB'] = gnb.score(X_train, y_train)

print('GaussianNB score = {0}'.format(gnb.score(X_train, y_train)))

print('************************************************')
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

k_fold = StratifiedKFold(n_splits=3)
from sklearn.model_selection import RandomizedSearchCV



# parameters in randomforest classifier

rf = RandomForestClassifier(random_state = 0)

print(rf.get_params())



n_estimators = [int(x) for x in np.linspace(start = 10, stop = 2000, num = 10)]

max_features = ['auto','sqrt']

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

min_samples_split = [2, 5, 10]

min_samples_leaf = [1, 2, 4]

bootstrap = [True, False]



# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}



rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = k_fold, verbose=2, random_state=0, n_jobs = -1)



rf_random.fit(X_train, y_train)



print('Random Forest Classifier with random search best estimator:')

print(rf_random.best_params_)

print('************************************************')

print('Random Forest Classifier with random search score = {0}'.format(rf_random.best_score_))

# creating a parameter grid

param_grid = {

    'bootstrap': [True],

    'max_depth': [50, 60, 70, 80, 90, 100],

    'max_features': ['sqrt'],

    'min_samples_leaf': [2, 3, 4],

    'min_samples_split': [8, 10, 12],

    'n_estimators': [800, 1000, 1200],

    'criterion' : ['gini']

}



grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 

                          cv = k_fold,  scoring="accuracy", n_jobs = -1, verbose = 2)



grid_search.fit(X_train, y_train)



print('Random Forest Classifier with grid search best estimator:')

print(grid_search.best_params_)

print('************************************************')

print('Random Forest Classifier with grid search score = {0}'.format(grid_search.best_score_))

# using the classifier above to predict survival rate and submitting the results

y_pred = grid_search.predict(X_test)

submission = pd.DataFrame({

        "PassengerId": X_test2["PassengerId"],

        "Survived": y_pred

    })



submission.to_csv('submission.csv', index=False)