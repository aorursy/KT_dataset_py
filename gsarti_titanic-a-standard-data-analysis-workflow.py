# General libraries for scientific purposes
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib
import sklearn
import os
import sys

# Preprocessing
from sklearn.preprocessing import LabelEncoder

# Plotting and visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Data models
from sklearn.linear_model import LinearRegression, LogisticRegressionCV, Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import cross_validate, ShuffleSplit, GridSearchCV
from xgboost import XGBClassifier

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Libraries versions
print('Python: {}'.format(sys.version))
print('numpy: {}'.format(np.__version__))
print('pandas: {}'.format(pd.__version__))
print('scipy: {}'.format(sp.__version__))
print('matplotlib: {}'.format(matplotlib.__version__))
print('sklearn: {}'.format(sklearn.__version__))
print('-'*30)

# Print local folder content
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Importing local datasets
train = pd.read_csv('../input/train.csv')
test  = pd.read_csv('../input/test.csv')

# Reference of both train and test, for cleaning purposes
datasets = [train, test]
train.head()
train.info()
print('-'*30)
test.info()
train.describe(include = 'all')
print('Null training values:\n', train.isnull().sum())
print("-"*30)
print('Test/Validation columns with null values:\n', test.isnull().sum())
drop_column = ['Cabin', 'Ticket']

for d in datasets:    
    d['Age'].fillna(d['Age'].median(), inplace = True)
    d['Fare'].fillna(d['Fare'].median(), inplace = True)
    d['Embarked'].fillna(d['Embarked'].mode()[0], inplace = True)
    d.drop(drop_column, axis=1, inplace = True)    

print(train.isnull().sum())
print("-"*30)
print(test.isnull().sum())
for d in datasets:
    d['FamilySize'] = d['SibSp'] + d['Parch'] + 1

    d['IsAlone'] = 1
    d['IsAlone'].loc[d['FamilySize'] > 1] = 0
    
    # Extract titles from names
    d['Title'] = d['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
    # Group uncommon titles under "other" label
    d['Title'] = d['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')
    
    # Group synonyms together 
    d['Title'] = d['Title'].replace('Mlle', 'Miss')
    d['Title'] = d['Title'].replace('Ms', 'Miss')
    d['Title'] = d['Title'].replace('Mme', 'Mrs')

    # Grouping continuous values into categories
    d['FareBand'] = pd.qcut(d['Fare'], 4)
    d['AgeBand'] = pd.cut(d['Age'].astype(int), 5)

train.info()
test.info()
train.head()
le = LabelEncoder()

titanic_train = train.copy(deep = 'True')
titanic_test = test.copy(deep = 'True')
titanic_train['Age'] = titanic_train['AgeBand']
titanic_train['Fare'] = titanic_train['FareBand']
titanic_test['Age'] = titanic_test['AgeBand']
titanic_test['Fare'] = titanic_test['FareBand']

column_transformed = ['Sex', 'Embarked', 'Title', 'AgeBand', 'FareBand']
column_transform = ['Sex', 'Embarked', 'Title', 'Age', 'Fare']

for d in datasets:
    for i in range(len(column_transform)):
        d[column_transform[i]] = le.fit_transform(d[column_transformed[i]])

datasets.append(titanic_train)
datasets.append(titanic_test)
drop_column = ['Name', 'SibSp', 'Parch', 'FareBand', 'AgeBand']
for d in datasets:
        d.drop(drop_column, axis=1, inplace = True)
train.drop('PassengerId', axis=1, inplace = True)
titanic_train.drop('PassengerId', axis=1, inplace = True)
train.head()
titanic_train.head()
feature_names = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'IsAlone', 'Title']
corr_matrix = train.corr()

print(corr_matrix["Survived"].sort_values(ascending=False))
print("-"*30)

for feature in feature_names:
    print('Correlation between Survived and', feature)
    print(titanic_train[[feature, 'Survived']].groupby([feature], as_index=False).mean())
    print("-"*30)
X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]
X_test  = test.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape
# The set of models I am going to compare
models = [
            LinearRegression(),
            LogisticRegressionCV(),
            Perceptron(),
            GaussianNB(),
            KNeighborsClassifier(),
            SVC(probability=True),
            DecisionTreeClassifier(),
            AdaBoostClassifier(),
            RandomForestClassifier(),
            XGBClassifier()    
        ]

# Create a table of comparison for models
models_columns = ['Name', 'Parameters','Train Accuracy', 'Validation Accuracy', 'Execution Time']
models_df = pd.DataFrame(columns = models_columns)
predictions = pd.DataFrame(columns = ['Survived'])

cv_split = ShuffleSplit(n_splits = 10, test_size = .2, train_size = .8, random_state = 0 )

index = 0
for model in models:
    models_df.loc[index, 'Name'] = model.__class__.__name__
    models_df.loc[index, 'Parameters'] = str(model.get_params())
    
    scores = cross_validate(model, X_train, Y_train, cv= cv_split)

    models_df.loc[index, 'Execution Time'] = scores['fit_time'].mean()
    models_df.loc[index, 'Train Accuracy'] = scores['train_score'].mean()
    models_df.loc[index, 'Validation Accuracy'] = scores['test_score'].mean()   
    
    index += 1

models_df.sort_values(by = ['Validation Accuracy'], ascending = False, inplace = True)
models_df
param_grid = {
              'criterion': ['gini', 'entropy'],
              'max_depth': [2,4,6,8,10,None],
              'random_state': [0]
             }

tree = DecisionTreeClassifier(random_state = 0)
score = cross_validate(tree, X_train, Y_train, cv  = cv_split)
tree.fit(X_train, Y_train)

grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid, scoring = 'roc_auc', cv = cv_split)
grid_search.fit(X_train, Y_train)

print('Before GridSearch:')
print('Parameters:', tree.get_params())
print("Training score:", score['train_score'].mean()) 
print("Validation score", score['test_score'].mean())
print('-'*30)
print('After GridSearch:')
print('Parameters:', grid_search.best_params_)
print("Training score:", grid_search.cv_results_['mean_train_score'][grid_search.best_index_]) 
print("Validation score", grid_search.cv_results_['mean_test_score'][grid_search.best_index_])
final_tree = GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid, scoring = 'roc_auc', cv = cv_split)
final_tree.fit(X_train, Y_train)
test['Survived'] = final_tree.predict(X_test)

submission = test[['PassengerId','Survived']]
submission.to_csv("submission.csv", index=False)
