# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Set up workspace
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
#from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
# Load data, preserve original data set
train_orig = pd.read_csv('../input/train.csv')
test_orig = pd.read_csv('../input/test.csv')
# create train and test dataframes
train_X_df = pd.DataFrame(train_orig)
test_df = pd.DataFrame(test_orig)
train_X_df.shape
test_df.shape
# take a peek at the head of train_df
train_X_df.head()
# take a peek at the tail of train_df
train_X_df.tail()
train_X_df.describe()
# take a peek at the head of test_df
test_df.head()
# scatter matrix
scatters_train = pd.plotting.scatter_matrix(train_X_df, figsize=[40,40])
# Plot correlations as heatmap
# Adapted from here: https://www.kaggle.com/foutik/decision-tree
corr = train_X_df.corr()
_ , ax = plt.subplots( figsize =( 12 , 10 ) )
cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
_ = sns.heatmap(corr, cmap = cmap, square=True, cbar_kws={ 'shrink' : .9 }, ax=ax, annot = True, annot_kws = {'fontsize' : 12 })
# Create target_y_df from survived column in train_X_df
# Then, drop that column from train_df
train_y_df = pd.DataFrame(train_X_df['Survived'])
train_X_df = train_X_df.drop(['Survived'], axis=1)
# Drop these columns for now. will return to them later
train_X_df = train_X_df.drop(['PassengerId'], axis=1)
train_X_df = train_X_df.drop(['Embarked'], axis=1)
train_X_df = train_X_df.drop(['Ticket'], axis=1)
train_X_df = train_X_df.drop(['Name'], axis=1)
train_X_df = train_X_df.drop(['Cabin'], axis=1)
# Replace missing Fare data with average fare
# TODO: identify employees of the ship, and change respective data accordingly
train_X_df.Fare = train_X_df.Fare.fillna(train_X_df['Fare'].mean(skipna=True))
train_X_df.Fare = train_X_df.Fare.replace(0, train_X_df['Fare'].mean(skipna=True))
# Replace missing Age data with average age
train_X_df.Age = train_X_df.Age.fillna(train_X_df['Age'].mean(skipna=True))
train_X_df.Age = train_X_df.Age.replace(0, train_X_df['Age'].mean(skipna=True))
# check for missing values
train_X_df.isnull().values.any()
# check for missing values
train_X_df.isnull().values.any()
train_X_df.dtypes
# Convert Sex data to 0's or 1's
train_X_df.loc[train_X_df.Sex != 'male', 'Sex'] = 0
train_X_df.loc[train_X_df.Sex == 'male', 'Sex'] = 1
train_X_df.head()
train_y_df.shape
# check for missing values
train_y_df.isnull().values.any()
# check for missing values
train_y_df.isna().values.any()
# create test_target_df from survived column in target_df
test_gini_dt6_df = pd.DataFrame(test_df)
# Replace missing Fare data with average fare
# TODO: identify employees of the ship, and change respective data accordingly
test_gini_dt6_df.Fare = test_gini_dt6_df.Fare.fillna(test_gini_dt6_df['Fare'].mean(skipna=True))
test_gini_dt6_df.Fare = test_gini_dt6_df.Fare.replace(0, test_gini_dt6_df['Fare'].mean(skipna=True))
# Replace missing Age data with average age
test_gini_dt6_df.Age = test_gini_dt6_df.Age.fillna(test_gini_dt6_df['Age'].mean(skipna=True))
test_gini_dt6_df.Age = test_gini_dt6_df.Age.replace(0, test_gini_dt6_df['Age'].mean(skipna=True))
# Save 'PassengerId' to concatenate w/ test data output after predictions
test_gini_dt6_passId = test_gini_dt6_df.PassengerId
test_gini_dt6_passId.shape
# dropping these columns for now. will return to them later
test_gini_dt6_df = test_gini_dt6_df.drop(['PassengerId'], axis=1)
test_gini_dt6_df = test_gini_dt6_df.drop(['Embarked'], axis=1)
test_gini_dt6_df = test_gini_dt6_df.drop(['Ticket'], axis=1)
test_gini_dt6_df = test_gini_dt6_df.drop(['Name'], axis=1)
test_gini_dt6_df = test_gini_dt6_df.drop(['Cabin'], axis=1)
# Convert Sex data to 0's or 1's
test_gini_dt6_df.loc[test_gini_dt6_df.Sex != 'male', 'Sex'] = 0
test_gini_dt6_df.loc[test_gini_dt6_df.Sex == 'male', 'Sex'] = 1
test_gini_dt6_df.head()
# check for missing values
test_gini_dt6_df.isnull().values.any()
# check for missing values
test_gini_dt6_df.isna().values.any()
# Set up grid search, 5 folds
model_5fold = model = tree.DecisionTreeClassifier()
param_grid = {'max_depth': list(range(1,11)),
              'criterion': ['entropy', 'gini']
              }
grid_5fold = GridSearchCV(model_5fold, param_grid, cv=5)
# Perform grid search 
grid_5fold.fit(train_X_df, train_y_df)
# Print out best parameters
print("Best parameters: {}".format(grid_5fold.best_params_))
# Get the accuracy
# Evaluate the tree       
predict_train_y = grid_5fold.best_estimator_.predict(train_X_df)
actual_train_y = train_y_df
# Print accuracy          
print("Accuracy: {}".format(accuracy_score(actual_train_y, predict_train_y)))
# Add test Passenger ID's to output dataframe
titanic_gini_dt6_results = pd.DataFrame(test_gini_dt6_passId)
# Create 'Survived' column to store predicted values
titanic_gini_dt6_results.insert(1,'Survived', np.nan)
# Run prediction and place output in 'Survived' column
titanic_gini_dt6_results.Survived = grid_5fold.best_estimator_.predict(test_gini_dt6_df)
titanic_gini_dt6_results.head()
# Save output, test_predict_dt15, to .csv for submission
titanic_gini_dt6_results.to_csv('titanic_gini_dt6_results.csv', index = False)
