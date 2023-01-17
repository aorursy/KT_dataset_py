# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



#!pip install -U sklearn
dataset_train = pd.read_csv('../input/titanic/train.csv', index_col='PassengerId')

dataset_test = pd.read_csv('../input/titanic/test.csv')

dataset_train = dataset_train.drop_duplicates()



X_full_train = dataset_train.iloc[:, 1:]

y_full_train = dataset_train.iloc[:, 0]

X_full_test = dataset_test.iloc[:, 1:]



# print(X_full_train[:5])

# print(y_full_train[:5])

# print(X_full_test[:5])



X_full_train.head(30)
# Check whether dataset contains empty values

print("TRAIN ---------------")

dataset_train.info()

print("TEST ----------------")

dataset_test.info()
# Simple feature engineering 

# https://triangleinequality.wordpress.com/2013/09/08/basic-feature-engineering-with-the-titanic-data/

# import strings

def substrings_in_string(big_string, substrings):

    if not big_string:

        return np.nan

    for substring in substrings:

        if big_string.find(substring) != -1:

            return substring

    print(big_string)

    return np.nan



title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',

                    'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',

                    'Don', 'Jonkheer']



X_full_train['Title'] = X_full_train['Name'].map(lambda x: substrings_in_string(x, title_list))

X_full_test['Title'] = X_full_test['Name'].map(lambda x: substrings_in_string(x, title_list))



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

X_full_train['Title'] = X_full_train.apply(replace_titles, axis=1)

X_full_test['Title'] = X_full_test.apply(replace_titles, axis=1)



X_full_train.drop('Name', axis=1, inplace=True)

X_full_test.drop('Name', axis=1, inplace=True)



# Drop 'Cabin' column- it's mainly NaN column

X_full_train.drop("Cabin", axis=1, inplace=True)

X_full_test.drop("Cabin", axis=1, inplace=True)



# Drop 'Ticket' column- it does not seem to be helpful data

X_full_train.drop("Ticket", axis=1, inplace=True)

X_full_test.drop("Ticket", axis=1, inplace=True)



#Creating new family_size column

X_full_train['Family_Size'] = X_full_train['SibSp'] + X_full_train['Parch']

X_full_test['Family_Size'] = X_full_test['SibSp'] + X_full_test['Parch']





X_full_train['Age'] = round(X_full_train['Age'])

X_full_test['Age'] = round(X_full_test['Age'])



X_full_train['Age*Class'] = X_full_train['Age'] + X_full_train['Pclass']

X_full_test['Age*Class'] = X_full_test['Age'] + X_full_test['Pclass']



X_full_train['Fare_Per_Person'] = X_full_train['Fare'] / (X_full_train['Family_Size']+1)

X_full_test['Fare_Per_Person'] = X_full_test['Fare'] / (X_full_test['Family_Size']+1)



ind = X_full_train[X_full_train.Embarked.isnull()].index

X_full_train.loc[ind, 'Embarked'] = 'S'



X_full_train.head(5)
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X_full_train, y_full_train, test_size=0.25, random_state=42)

print(X_train.shape)

print(X_valid.shape)

print(y_train.shape)

print(y_valid.shape)
# Check cardinality of categorical columns

object_cols_train = [cname for cname in X_train.columns 

               if X_train[cname].dtype == 'object']

object_nunique_train = list(map(lambda col: X_train[col].nunique(), object_cols_train))

d_train = dict(zip(object_cols_train, object_nunique_train))



object_cols_valid = [cname for cname in X_valid.columns 

               if X_valid[cname].dtype == 'object']

object_nunique_valid = list(map(lambda col: X_valid[col].nunique(), object_cols_valid))

d_valid = dict(zip(object_cols_valid, object_nunique_valid))



object_cols_test = [cname for cname in X_full_test.columns 

               if X_full_test[cname].dtype == 'object']

object_nunique_test = list(map(lambda col: X_full_test[col].nunique(), object_cols_test))

d_test = dict(zip(object_cols_test, object_nunique_test))



object_cols = object_cols_train



print(sorted(d_train.items(), key=lambda x: x[1]))

print(sorted(d_valid.items(), key=lambda x: x[1]))

print(sorted(d_test.items(), key=lambda x: x[1]))
# Impute missing values

from sklearn.impute import SimpleImputer



X_train_columns = X_train.columns

X_valid_columns = X_valid.columns

X_test_columns = X_full_test.columns



imputer = SimpleImputer(strategy='most_frequent')

X_train = pd.DataFrame(imputer.fit_transform(X_train))

X_valid = pd.DataFrame(imputer.transform(X_valid))

X_full_test = pd.DataFrame(imputer.transform(X_full_test))



# Imputation removed column names; put them back

X_train.columns = X_train_columns

X_valid.columns = X_valid_columns

X_full_test.columns = X_test_columns



# Check whether dataset contains empty values

print(X_train.isnull().sum())

print(X_valid.isnull().sum())

print(X_full_test.isnull().sum())



X_train.describe()
# One hot encode categorical features

from sklearn.preprocessing import OneHotEncoder



ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')



OH_cols_train = pd.DataFrame(ohe.fit_transform(X_train[object_cols])) 

OH_cols_valid = pd.DataFrame(ohe.transform(X_valid[object_cols]))

OH_cols_test = pd.DataFrame(ohe.transform(X_full_test[object_cols]))



OH_cols_train.index = X_train.index

OH_cols_valid.index = X_valid.index

OH_cols_test.index = X_full_test.index



num_X_train = X_train.drop(object_cols, axis=1)

num_X_valid = X_valid.drop(object_cols, axis=1)

num_X_test = X_full_test.drop(object_cols, axis=1)



X_train = pd.concat([num_X_train, OH_cols_train], axis=1)

X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

X_test = pd.concat([num_X_test, OH_cols_test], axis=1)



X_train.head()
# Standard scale numerical features



from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

X_train = ss.fit_transform(X_train)

X_valid = ss.transform(X_valid)

X_test = ss.transform(X_test)



print(X_train[:3])
from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV



def check_accuracy(model, param_grid):

    gs = GridSearchCV(model, param_grid, cv=5, verbose=False, n_jobs=-1, scoring='accuracy')

    gs = gs.fit(X_train, y_train)

    y_pred = gs.predict(X_valid)

    print(model.__class__.__name__, gs.best_params_)

    return accuracy_score(y_valid, y_pred), gs.best_params_
# Create model

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.svm import LinearSVC

from sklearn.tree import DecisionTreeClassifier



models = []

param_grids = []

best_params = []

models.append(SVC(random_state=42))

param_grids.append({'C':[0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]})



models.append(KNeighborsClassifier())

param_grids.append({'n_neighbors':[2, 3, 4, 5, 6, 7, 8, 9],

                   'leaf_size':[5, 10, 15, 20, 25, 30, 35, 40]})



models.append(RandomForestClassifier(random_state=42, n_jobs=-1))

param_grids.append({'criterion':['gini', 'entropy'],

                    'n_estimators':[50, 100, 150, 200, 250],

                   'max_depth': [8, 16, 32, 64],

                   'max_leaf_nodes':[8, 16, 32, 64]})



models.append(GaussianNB())

param_grids.append({})



models.append(Perceptron(random_state=42, n_jobs=-1))

param_grids.append({'penalty':['l1', 'l2', 'elasticnet'],

                   'alpha':[0.00001, 0.0001, 0.0005, 0.001]

                   })



models.append(SGDClassifier(random_state=42, n_jobs=-1))

param_grids.append({'penalty':['l1', 'l2', 'elasticnet'],

                   'alpha':[0.00001, 0.0001, 0.0005, 0.001],

                  'l1_ratio':[0.0001, 0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25]}

                  )



models.append(LinearSVC(random_state=42))

param_grids.append({'penalty':['l1', 'l2'],

                   'dual':[False],

                   'C':[0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.6, 0.8, 1.0, 1.2],

                   })



models.append(DecisionTreeClassifier(random_state=42))

param_grids.append({'max_depth': [2, 3, 4, 5, 6, 7, 8, 16, 32, 64],

                   'max_leaf_nodes':[2, 3, 4, 5, 6, 7, 8, 16, 32, 64]})

models.append(DecisionTreeClassifier(random_state=42))

param_grids.append({'max_depth': [6, 7, 8, 16, 32, 64],

                   'max_leaf_nodes':[6, 7, 8, 16, 32, 64]})



accuracies = []

for model, param_grid in zip(models, param_grids):

    acc, params = check_accuracy(model, param_grid)

    accuracies.append(acc)

    best_params.append(params)

    print(model.__class__.__name__, acc)



    

rf = RandomForestClassifier(criterion='gini', 

                             n_estimators=700,

                             min_samples_split=16,

                             min_samples_leaf=1,

                             max_features='auto',

                             oob_score=True,

                             random_state=1,

                             n_jobs=-1).fit(X_train, y_train)

y_pred_rf = rf.predict(X_valid)

print(accuracy_score(y_valid, y_pred_rf))

    

clf = KNeighborsClassifier().fit(X_train, y_train)

y_pred = clf.predict(X_valid)
models_score = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree1', 'Decision Tree2'],

    'Score': accuracies})

models_score.sort_values(by='Score', ascending=False)



best_parameters = best_params[models_score.Score.idxmax()]

best_model = models[models_score.Score.idxmax()]

best_model.set_params(**best_parameters, random_state=42)

print(best_model)



best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_valid)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_valid, y_pred)

acc = accuracy_score(y_valid, y_pred)



print(acc)

print(cm)
y_pred_test = best_model.predict(X_test)

y_pred_test_rf = rf.predict(X_test)

submission = pd.DataFrame({

        "PassengerId": dataset_test['PassengerId'],

        "Survived": y_pred_test

    })

submission.to_csv('submission.csv', index=False)