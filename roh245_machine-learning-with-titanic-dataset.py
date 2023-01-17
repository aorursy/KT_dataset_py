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
titanictrain=pd.read_csv('/kaggle/input/titanic/train.csv')

titanictest=pd.read_csv('/kaggle/input/titanic/test.csv')
titanictrain.isnull().sum()
titanictest.isnull().sum()
titanictrain.info()
# Store our passenger ID for easy access

PassengerId = titanictest['PassengerId']
titanictrain.describe()
titanictrain.describe(include=['O'])
titanictrain[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
titanictrain[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
titanictrain[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
titanictrain[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline





g = sns.FacetGrid(titanictrain, col='Survived')

g.map(plt.hist, 'Age', bins=20)
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')

grid = sns.FacetGrid(titanictrain, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
# grid = sns.FacetGrid(titanictrain, col='Embarked')

grid = sns.FacetGrid(titanictrain, row='Embarked', size=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()
# grid = sns.FacetGrid(titanictrain, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})

grid = sns.FacetGrid(titanictrain, row='Embarked', col='Survived', size=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

grid.add_legend()
import re



full_data = [titanictrain, titanictest]



# Feature that tells whether a passenger had a cabin on the Titanic

titanictrain['Has_Cabin'] = titanictrain["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

titanictest['Has_Cabin'] = titanictest["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

# Feature engineering steps taken from Sina

# Create new feature FamilySize as a combination of SibSp and Parch

for dataset in full_data:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

# Create new feature IsAlone from FamilySize

for dataset in full_data:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# Remove all NULLS in the Embarked column

for dataset in full_data:

    dataset['Embarked'] = dataset['Embarked'].fillna(dataset['Embarked'].mode()[0])

# Remove all NULLS in the Fare column and create a new feature CategoricalFare

for dataset in full_data:

    dataset['Fare'] = dataset['Fare'].fillna(titanictrain['Fare'].median())

titanictrain['CategoricalFare'] = pd.qcut(titanictrain['Fare'], 4)

# Create a New feature CategoricalAge

for dataset in full_data:

    age_avg = dataset['Age'].mean()

    age_std = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)

titanictrain['CategoricalAge'] = pd.cut(titanictrain['Age'], 5)

# Define function to extract titles from passenger names

def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""

# Create a new feature Title, containing the titles of passenger names

for dataset in full_data:

    dataset['Title'] = dataset['Name'].apply(get_title)

# Group all non-common titles into one single grouping "Rare"

for dataset in full_data:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
for dataset in full_data:

    # Mapping Sex

    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    

    # Mapping titles

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

    

    # Mapping Embarked

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    

    # Mapping Fare

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

     # Mapping Age

    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4 ;
#Adding a new feature Age*pclass.

for dataset in full_data:

    dataset['Age_Class']= dataset['Age']* dataset['Pclass']
# Feature selection

drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']

titanictrain = titanictrain.drop(drop_elements, axis = 1)

titanictrain = titanictrain.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)

titanictest  = titanictest.drop(drop_elements, axis = 1)
#profile report

import pandas_profiling

titanictrain.profile_report()
titanictrain = titanictrain.values

titanictest  = titanictest.values
from sklearn.model_selection import StratifiedShuffleSplit



sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)



X = titanictrain[0::, 1::]

y = titanictrain[0::, 0]



for train_index, test_index in sss.split(X, y):

	X_train, X_test = X[train_index], X[test_index]

	y_train, y_test = y[train_index], y[test_index]
#Logistic Regression

from sklearn.linear_model import LogisticRegression

logit_model = LogisticRegression(solver = 'saga', random_state = 0,C=100)

logit_model.fit(X_train, y_train)



y_logit_pred = logit_model.predict(X_test)



from sklearn import metrics

count_misclassified = (y_test != y_logit_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_logit_pred)

print('Accuracy: {:.2f}'.format(accuracy))

precision = metrics.precision_score(y_test, y_logit_pred, average= 'macro')

print('Precision: {:.2f}'.format(precision))

recall = metrics.recall_score(y_test, y_logit_pred, average= 'macro')

print('Recall: {:.2f}'.format(recall))

f1_score = metrics.f1_score(y_test, y_logit_pred, average = 'macro')

print('F1 score: {:.2f}'.format(f1_score))
#Support Vector Classifier with Linear



from sklearn.svm import SVC

svc_linear_model = SVC(kernel='linear', C=0.01, gamma= 'scale', decision_function_shape='ovo', random_state = 42)



svc_linear_model.fit(X_train, y_train)

y_svc_linear_pred = svc_linear_model.predict(X_test)



from sklearn import metrics

count_misclassified = (y_test != y_svc_linear_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_svc_linear_pred)

print('Accuracy: {:.2f}'.format(accuracy))

precision = metrics.precision_score(y_test, y_svc_linear_pred, average= 'macro')

print('Precision: {:.2f}'.format(precision))

recall = metrics.recall_score(y_test, y_svc_linear_pred, average= 'macro')

print('Recall: {:.2f}'.format(recall))

f1_score = metrics.f1_score(y_test, y_svc_linear_pred, average = 'macro')

print('F1 score: {:.2f}'.format(f1_score))
#Support Vector Classifier with Polynomial



from sklearn.svm import SVC

svc_ply_model = SVC(kernel='poly', C=100, gamma= 'scale', decision_function_shape='ovo', random_state = 42)



svc_ply_model.fit(X_train, y_train)

y_svc_ply_pred = svc_ply_model.predict(X_test)



from sklearn import metrics

count_misclassified = (y_test != y_svc_linear_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_svc_linear_pred)

print('Accuracy: {:.2f}'.format(accuracy))

precision = metrics.precision_score(y_test, y_svc_linear_pred, average= 'macro')

print('Precision: {:.2f}'.format(precision))

recall = metrics.recall_score(y_test, y_svc_linear_pred, average= 'macro')

print('Recall: {:.2f}'.format(recall))

f1_score = metrics.f1_score(y_test, y_svc_linear_pred, average = 'macro')

print('F1 score: {:.2f}'.format(f1_score))
#Support Vector Classifier with RBF



from sklearn.svm import SVC

svc_rbf_model = SVC(kernel='rbf', C=10, gamma= 'scale', decision_function_shape='ovo', random_state = 42)



svc_rbf_model.fit(X_train, y_train)

y_svc_rbf_pred = svc_rbf_model.predict(X_test)



from sklearn import metrics

count_misclassified = (y_test != y_svc_linear_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_svc_linear_pred)

print('Accuracy: {:.2f}'.format(accuracy))

precision = metrics.precision_score(y_test, y_svc_linear_pred, average= 'macro')

print('Precision: {:.2f}'.format(precision))

recall = metrics.recall_score(y_test, y_svc_linear_pred, average= 'macro')

print('Recall: {:.2f}'.format(recall))

f1_score = metrics.f1_score(y_test, y_svc_linear_pred, average = 'macro')

print('F1 score: {:.2f}'.format(f1_score))
#Ensemble Models

#Random Forest Classifier with Hyper Parameter tuning



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

# Create the parameter grid based on the results of random search 

param_grid = {

    "criterion" : ["gini", "entropy"],

    'bootstrap': [True],

    'max_depth': [1,16,32,64,128],#list(range(90,100)),

    'max_features': ['auto'],

    'min_samples_leaf': [1,5,10,15,20,25,50],

    'min_samples_split': [2,4,8,10,12,16,20,25,35],

    'n_estimators': [256,500,1000],

    'random_state': [0]

}

# Create a based model

rf_model_classification = RandomForestClassifier()

# Instantiate the grid search model

grid_search_rf_model_classificaiton = GridSearchCV(estimator = rf_model_classification, param_grid = param_grid, 

                          cv = 3, n_jobs = -1, verbose = 2)



grid_search_rf_model_classificaiton.fit(X_train, y_train)

print(grid_search_rf_model_classificaiton.best_params_)

y_rf_classification_pred = grid_search_rf_model_classificaiton.predict(X_test)



from sklearn import metrics

count_misclassified = (y_test != y_rf_classification_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_rf_classification_pred)

print('Accuracy: {:.2f}'.format(accuracy))

precision = metrics.precision_score(y_test, y_rf_classification_pred, average= 'macro')

print('Precision: {:.2f}'.format(precision))

recall = metrics.recall_score(y_test, y_rf_classification_pred, average= 'macro')

print('Recall: {:.2f}'.format(recall))

f1_score = metrics.f1_score(y_test, y_rf_classification_pred, average = 'macro')

print('F1 score: {:.2f}'.format(f1_score))
#Gradient Boost Classifier with Hyper Parameter Tuning



from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV

# Create the parameter grid based on the results of random search 

param_grid = {

    'max_depth': [1,16,32,64,128],

    'max_features': [3],

    'min_samples_leaf': [3],

    'min_samples_split': [8,10],

    'n_estimators': [1,2,4,8,16,32,64,128,256],

    'learning_rate' : [1, 0.5, 0.25, 0.1, 0.05, 0.01],

    'random_state' : [0]

}

# Create a based model

gbc_model_classification = GradientBoostingClassifier()

# Instantiate the grid search model

grid_search_gbc_model_classificaiton = GridSearchCV(estimator = gbc_model_classification, param_grid = param_grid, 

                          cv = 3, n_jobs = -1, verbose = 2)



grid_search_gbc_model_classificaiton.fit(X_train, y_train)

print(grid_search_gbc_model_classificaiton.best_params_)

y_gbc_model_pred = grid_search_gbc_model_classificaiton.predict(X_test)



from sklearn import metrics

count_misclassified = (y_test != y_gbc_model_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_gbc_model_pred)

print('Accuracy: {:.2f}'.format(accuracy))

precision = metrics.precision_score(y_test, y_gbc_model_pred, average= 'macro')

print('Precision: {:.2f}'.format(precision))

recall = metrics.recall_score(y_test, y_gbc_model_pred, average= 'macro')

print('Recall: {:.2f}'.format(recall))

f1_score = metrics.f1_score(y_test, y_gbc_model_pred, average = 'macro')

print('F1 score: {:.2f}'.format(f1_score))
#XG Boost Classifier with Hyper Parameter Tuning

from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV

param_grid = {

    'min_child_weight': [1, 5, 10],

     'gamma': [0.5, 1, 1.5, 2, 5],

     'max_depth': [1,16,32,64,128],

     'eta' : [1, 0.5, 0.25, 0.1, 0.05, 0.01],

     'n_estimators': [1,2,4,8,16,32,64,128,256],

     'random_state': [0]

}

# Create a based model

xgb_model_classification = XGBClassifier()

# Instantiate the grid search model

grid_search_xgb_model_classificaiton = GridSearchCV(estimator = xgb_model_classification, param_grid = param_grid, 

                          cv = 3, n_jobs = -1, verbose = 2)



grid_search_xgb_model_classificaiton.fit(X_train, y_train)

print(grid_search_xgb_model_classificaiton.best_params_)

y_xgb_classification_pred = grid_search_xgb_model_classificaiton.predict(X_test)



from sklearn import metrics

count_misclassified = (y_test != y_xgb_classification_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_xgb_classification_pred)

print('Accuracy: {:.2f}'.format(accuracy))

precision = metrics.precision_score(y_test, y_xgb_classification_pred, average= 'macro')

print('Precision: {:.2f}'.format(precision))

recall = metrics.recall_score(y_test, y_xgb_classification_pred, average= 'macro')

print('Recall: {:.2f}'.format(recall))

f1_score = metrics.f1_score(y_test, y_xgb_classification_pred, average = 'macro')

print('F1 score: {:.2f}'.format(f1_score))
#K nearest neighbor Classifier with Hyper Parameter Tuning

from sklearn.neighbors import KNeighborsClassifier

param_grid = {

    'n_neighbors': [3,5,7,9,11],

    'weights' : ['uniform', 'distance'],

    'metric' : ['euclidean', 'manhattan']

}

# Create a based model

KNeighbors_classification = KNeighborsClassifier()

# Instantiate the grid search model

grid_search_KNeighbors_classificaiton = GridSearchCV(estimator = KNeighbors_classification, param_grid = param_grid, 

                          cv = 3, n_jobs = -1, verbose = 2)



grid_search_KNeighbors_classificaiton.fit(X_train, y_train)

print(grid_search_KNeighbors_classificaiton.best_params_)

y_xgb_classification_pred = grid_search_KNeighbors_classificaiton.predict(X_test)



from sklearn import metrics

count_misclassified = (y_test != y_xgb_classification_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_xgb_classification_pred)

print('Accuracy: {:.2f}'.format(accuracy))

precision = metrics.precision_score(y_test, y_xgb_classification_pred, average= 'macro')

print('Precision: {:.2f}'.format(precision))

recall = metrics.recall_score(y_test, y_xgb_classification_pred, average= 'macro')

print('Recall: {:.2f}'.format(recall))

f1_score = metrics.f1_score(y_test, y_xgb_classification_pred, average = 'macro')

print('F1 score: {:.2f}'.format(f1_score))
# Generate Submission File 

predictions = grid_search_xgb_model_classificaiton.predict(titanictest)

Submission = pd.DataFrame({ 'PassengerId': PassengerId,

                            'Survived': predictions })

Submission.to_csv("Submission.csv", index=False)


