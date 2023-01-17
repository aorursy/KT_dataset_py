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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
# Load data, preserve original data set
train_orig = pd.read_csv('../input/train.csv')
test_orig = pd.read_csv('../input/test.csv')
# create train and test dataframes
train_df = pd.DataFrame(train_orig)
test_df = pd.DataFrame(test_orig)
train_df.shape
test_df.shape
# take a peek at the head of train_df
train_df.head()
# take a peek at the tail of train_df
train_df.tail()
train_df.describe()
# take a peek at the head of test_df
test_df.head()
# scatter matrix
scatters_train = pd.plotting.scatter_matrix(train_df, figsize=[40,40])
# Plot correlations as heatmap
# Adapted from here: https://www.kaggle.com/foutik/decision-tree
corr = train_df.corr()
_ , ax = plt.subplots( figsize =( 12 , 10 ) )
cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
_ = sns.heatmap(corr, cmap = cmap, square=True, cbar_kws={ 'shrink' : .9 }, ax=ax, annot = True, annot_kws = {'fontsize' : 12 })
# create target_df from survived column in train_df
# then drop that column from train_df
target_df = pd.DataFrame(train_df['Survived'])
train_df = train_df.drop(['Survived'], axis=1)
# dropping these columns for now. will return to them later
train_df = train_df.drop(['PassengerId'], axis=1)
train_df = train_df.drop(['Embarked'], axis=1)
train_df = train_df.drop(['Ticket'], axis=1)
train_df = train_df.drop(['Name'], axis=1)
train_df = train_df.drop(['Cabin'], axis=1)
# check for missing values
train_df.isnull().values.any()
# Replace missing Fare data with average fare
# TODO: identify employees of the ship, and change respective data accordingly
train_df.Fare = train_df.Fare.fillna(train_df['Fare'].mean(skipna=True))
train_df.Fare = train_df.Fare.replace(0, train_df['Fare'].mean(skipna=True))
# Replace missing Age data with average age
train_df.Age = train_df.Age.fillna(train_df['Age'].mean(skipna=True))
train_df.Age = train_df.Age.replace(0, train_df['Age'].mean(skipna=True))
# check for missing values
#train_df.isnull().values.any()
train_df.isna()
# check for missing values
train_df.isnull().values.any()
train_df.dtypes
# Convert Sex data to 0's or 1's
train_df.loc[train_df.Sex != 'male', 'Sex'] = 0
train_df.loc[train_df.Sex == 'male', 'Sex'] = 1
train_df.head()
# check for missing values
target_df.isnull().values.any()
# check for missing values
target_df.isna().values.any()
# create test_target_df from survived column in target_df
test_dt15_df = pd.DataFrame(test_df)
# Replace missing Fare data with average fare
# TODO: identify employees of the ship, and change respective data accordingly
test_dt15_df.Fare = test_dt15_df.Fare.fillna(test_dt15_df['Fare'].mean(skipna=True))
test_dt15_df.Fare = test_dt15_df.Fare.replace(0, test_dt15_df['Fare'].mean(skipna=True))
# Replace missing Age data with average age
test_dt15_df.Age = test_dt15_df.Age.fillna(test_dt15_df['Age'].mean(skipna=True))
test_dt15_df.Age = test_dt15_df.Age.replace(0, test_dt15_df['Age'].mean(skipna=True))
# Save 'PassengerId' to concatenate w/ test data output after predictions
# test_passId = pd.DataFrame(['PassengerId'])
test_passId = test_dt15_df.PassengerId
test_passId.shape
# dropping these columns for now. will return to them later
test_dt15_df = test_dt15_df.drop(['PassengerId'], axis=1)
test_dt15_df = test_dt15_df.drop(['Embarked'], axis=1)
test_dt15_df = test_dt15_df.drop(['Ticket'], axis=1)
test_dt15_df = test_dt15_df.drop(['Name'], axis=1)
test_dt15_df = test_dt15_df.drop(['Cabin'], axis=1)
# Convert Sex data to 0's or 1's
test_dt15_df.loc[test_dt15_df.Sex != 'male', 'Sex'] = 0
test_dt15_df.loc[test_dt15_df.Sex == 'male', 'Sex'] = 1
test_dt15_df.head()
# check for missing values
test_dt15_df.isnull().values.any()
# check for missing values
test_dt15_df.isna().values.any()
# Set up model, max depth of 12
train_dtree_12 = tree.DecisionTreeClassifier(criterion='entropy', max_depth=12)
# Build the model
train_dtree_12.fit(train_df, target_df)
# build df of predicted vals for train_df
train_predict_array_12 = train_dtree_12.predict(train_df)              # produces an array of labels
train_predicted_labels_12 = pd.DataFrame(train_predict_array_12)     # turn it into a DF
train_predicted_labels_12.columns = ['Survived']                  # name the column - same name as in target
# check that the predicted labels are the same as the originals
train_predicted_labels_12.equals(target_df)
# Print model accuracy
print("Our model accuracy is: {} at a depth of 12.".format(accuracy_score(target_df, train_predicted_labels_12)))
# Set up model, max depth of 15
train_dtree_15 = tree.DecisionTreeClassifier(criterion='entropy', max_depth=15)
# Build the model
train_dtree_15.fit(train_df, target_df)
# check accuracy of 2nd array of predicted values
train_predict_array_15 = train_dtree_15.predict(train_df)                  # produces an array of labels
train_predicted_labels_15 = pd.DataFrame(train_predict_array_15)         # turn it into a DF
train_predicted_labels_15.columns = ['Survived']                         # name the column - same name as in target!
# check that the predicted labels are the same as the originals
train_predicted_labels_15.equals(target_df)
# Print model accuracy
print("Our model accuracy is: {} at a depth of 15.".format(accuracy_score(target_df, train_predicted_labels_15)))
### Print accuracy of 96% at depth of 15
# Add test passenger ID's to output dataframe, test_predict_15
test_predict_dt15 = pd.DataFrame(test_passId)
# Create 'Survived' column to store predicted values
test_predict_dt15.insert(1, 'Survived', np.nan)
test_predict_dt15.head()
test_predict_dt15.tail()
# Run prediction and place output in 'Survived' column
test_predict_dt15.Survived = train_dtree_15.predict(test_dt15_df)
# Check output
test_predict_dt15.head()

# Save output, test_predict_dt15, to .csv for submission
test_predict_dt15.to_csv('test_predict_dt15.csv', index = False)
