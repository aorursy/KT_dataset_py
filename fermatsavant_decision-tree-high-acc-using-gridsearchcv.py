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
#Libraries
from sklearn import tree
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
#For work encode categorical atrubuts
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
#For do a best a work flow
from sklearn.pipeline import Pipeline
#Missing values
from sklearn.impute import SimpleImputer

#Load data
train_set = pd.read_csv('/kaggle/input/titanic/train.csv')
test_set = pd.read_csv('/kaggle/input/titanic/test.csv')
#'PassengerId', 'Name', 'Ticket', 'Cabin', 'Age' 
#Discretization needed
#'Fare'
train_set.dtypes


print(train_set.columns)
train_set.describe()
#Train
y_train = train_set['Survived'].copy()
X_train = train_set.drop(['Survived', 'PassengerId', 'Name', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis=1)
#Train
X_test = test_set
X_train.head()
num_attribs = X_train.select_dtypes(exclude=['object', 'category']).columns
cat_attribs = X_train.select_dtypes(include=['object', 'category']).columns
cat_attribs, num_attribs
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="mean")),
    ('mean', StandardScaler()),#std_scaler#Standarization
    ])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
    ]) 
x_train = full_pipeline.fit_transform(X_train)
x_test = full_pipeline.fit_transform(X_test)
print(X_train.shape, y_train.shape, X_test.shape)

from sklearn.model_selection import GridSearchCV

param_grid = {'max_features': ['auto', 'sqrt', 'log2'],
              'ccp_alpha': [0.1, .01, .001],
              'max_depth' : [5, 6, 7, 8, 9],
              'criterion' :['gini', 'entropy']
             }
tree_clas = DecisionTreeClassifier(random_state=1024)
grid_search = GridSearchCV(estimator=tree_clas, param_grid=param_grid, cv=5, verbose=True)
grid_search.fit(x_train, y_train)
final_model = grid_search.best_estimator_
final_model
#Training the model
tree_clas = DecisionTreeClassifier(ccp_alpha=0.01, class_weight=None, criterion='entropy',
                       max_depth=5, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=1024, splitter='best')
tree_clas.fit(x_train, y_train)
y_predict = tree_clas.predict(x_test)