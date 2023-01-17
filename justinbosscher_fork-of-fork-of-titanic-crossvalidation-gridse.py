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
from sklearn.neural_network import MLPClassifier
# Load data, preserve original data set
train_orig = pd.read_csv('../input/train.csv')
test_orig = pd.read_csv('../input/test.csv')
# create train and test dataframes
train_X = pd.DataFrame(train_orig)
test_X = pd.DataFrame(test_orig)
train_X.shape
test_X.shape
# take a peek at the head of train_df
train_X.head()
# take a peek at the tail of train_df
train_X.tail()
train_X.describe()
# take a peek at the head of test_df
test_X.head()
# scatter matrix
scatters_train = pd.plotting.scatter_matrix(train_X, figsize=[40,40])
# Plot correlations as heatmap
# Adapted from here: https://www.kaggle.com/foutik/decision-tree
corr = train_X.corr()
_ , ax = plt.subplots( figsize =( 12 , 10 ) )
cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
_ = sns.heatmap(corr, cmap = cmap, square=True, cbar_kws={ 'shrink' : .9 }, ax=ax, annot = True, annot_kws = {'fontsize' : 12 })
# Create target_y_df from survived column in train_X_df
# Then, drop that column from train_df
actual_y = pd.DataFrame(train_X['Survived'])
train_X = train_X.drop(['Survived'], axis=1)
# Drop these columns for now. will return to them later
train_X = train_X.drop(['PassengerId'], axis=1)
train_X = train_X.drop(['Embarked'], axis=1)
train_X = train_X.drop(['Ticket'], axis=1)
train_X = train_X.drop(['Name'], axis=1)
train_X = train_X.drop(['Cabin'], axis=1)
# Replace missing Fare data with average fare
# TODO: identify employees of the ship, and change respective data accordingly
train_X.Fare = train_X.Fare.fillna(train_X['Fare'].mean(skipna=True))
train_X.Fare = train_X.Fare.replace(0, train_X['Fare'].mean(skipna=True))
# Replace missing Age data with average age
train_X.Age = train_X.Age.fillna(train_X['Age'].mean(skipna=True))
train_X.Age = train_X.Age.replace(0, train_X['Age'].mean(skipna=True))
# check for missing values
train_X.isnull().values.any()
# check for missing values
train_X.isnull().values.any()
train_X.dtypes
# Convert Sex data to 0's or 1's
train_X.loc[train_X.Sex != 'male', 'Sex'] = 0
train_X.loc[train_X.Sex == 'male', 'Sex'] = 1
train_X.head()
actual_y.shape
# check for missing values
actual_y.isnull().values.any()
# check for missing values
actual_y.isna().values.any()
# Replace missing Fare data with average fare
# TODO: identify employees of the ship, and change respective data accordingly
test_X.Fare = test_X.Fare.fillna(test_X['Fare'].mean(skipna=True))
test_X.Fare = test_X.Fare.replace(0, test_X['Fare'].mean(skipna=True))
# Replace missing Age data with average age
test_X.Age = test_X.Age.fillna(test_X['Age'].mean(skipna=True))
test_X.Age = test_X.Age.replace(0, test_X['Age'].mean(skipna=True))
# Save 'PassengerId' to concatenate w/ test data output after predictions
test_X_passId = test_X.PassengerId
test_X_passId.shape
# dropping these columns for now. will return to them later
test_X = test_X.drop(['PassengerId'], axis=1)
test_X = test_X.drop(['Embarked'], axis=1)
test_X = test_X.drop(['Ticket'], axis=1)
test_X = test_X.drop(['Name'], axis=1)
test_X = test_X.drop(['Cabin'], axis=1)
# Convert Sex data to 0's or 1's
test_X.loc[test_X.Sex != 'male', 'Sex'] = 0
test_X.loc[test_X.Sex == 'male', 'Sex'] = 1
test_X.head()
# check for missing values
test_X.isnull().values.any()
# check for missing values
test_X.isna().values.any()
# Set up neural network
model = MLPClassifier(max_iter=2000)
# Grid Search
param_grid = {'hidden_layer_sizes': [ (5,), (6,), (7,), (8,), (9,), (10,),
                        (11,), (12,), (13,), (14,), (15,), (16,),
                        (17,), (18,), (19,), (20,)]}
grid = GridSearchCV(model,param_grid,cv=5)
grid.fit(train_X,actual_y)
print("Grid Search: best parameters:{}".format(grid.best_params_))
# Evaluate best model
best_model = grid.best_estimator_

predict_y = pd.DataFrame(best_model.predict(train_X))
print("Accuracy:{}".format(accuracy_score(actual_y,predict_y)))
results = pd.DataFrame(best_model.predict(test_X))
# Add test Passenger ID's to output dataframe
titanic_mlp_results = pd.DataFrame(test_X_passId)
# Create 'Survived' column to store predicted values
titanic_mlp_results.insert(1,'Survived', np.nan)
# Enter results into survived column of results dataframe
titanic_mlp_results.Survived = results
# Save output, test_predict_dt15, to .csv for submission
titanic_mlp_results.to_csv('titanic_mlp_results.csv', index = False)
