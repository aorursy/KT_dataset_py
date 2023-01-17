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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
# Load data, preserve original data set
train_orig = pd.read_csv('../input/train.csv')
test_orig = pd.read_csv('../input/test.csv')
# Create train and test dataframes
train_df = pd.DataFrame(train_orig)
test_df = pd.DataFrame(test_orig)
# How many rows / columns ?
train_df.shape
# How many rows / columns ?
train_df.shape
# Take a peek at the head of train_df
train_df.head()
# Take a peek at the tail of train_df
train_df.tail()
# Descriptive stats
train_df.describe()
# Take a peek at the head of test_df
train_df.head()
# Scatter matrix
scatters_train = pd.plotting.scatter_matrix(train_df, figsize=[40,40])
# Plot correlations as heatmap
# Adapted from here: https://www.kaggle.com/foutik/decision-tree
corr = train_df.corr()
_ , ax = plt.subplots( figsize =( 12 , 10 ) )
cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
_ = sns.heatmap(corr, cmap = cmap, square=True, cbar_kws={ 'shrink' : .9 }, ax=ax, annot = True, annot_kws = {'fontsize' : 12 })
# Impact of Sex on survival rate
# Adapted from: https://towardsdatascience.com/play-with-data-2a5db35b279c
print(train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean())
# Impact of Embarked on Survival rate
print(train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())
# Impact of SibSp on Survival rate
print(train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean())
# Impact of Embarked on Survival rate
print(train_df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean())
# Create X dataframe from train_df
X = pd.DataFrame(train_df)
# Create 'FamilySize' column that is sum of SibSp and Parch and 1
X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
X.head()
# Impact of FamilySize on Survival rate
print(X[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())
# Bin FamilySize based on mean survival rates
# If FamilySize is greater than 8, 1; else, 0
# Represents mean 0% survival rate
X['FamSize_8+'] = 0
X.loc[(X['FamilySize'] >= 8), 'FamSize_8+'] = 1
# If FamilySize is 1 or 7, 1; else, 0
# Represents mean 30-33% survival rate
X['FamSize_1|7'] = 0
X.loc[(X['FamilySize'] == 1) | (X['FamilySize'] == 7), 'FamSize_1|7'] = 1
# If FamilySize is 5 or 6, 1; else, 0
# Represents mean 13-20% survival rate
X['FamSize_5|6'] = 0
X.loc[(X['FamilySize'] == 5) | (X['FamilySize'] == 6), 'FamSize_5|6'] = 1
# If FamilySize is 4, 1; else, 0
# Represents mean 72% survival rate
X['FamSize_4'] = 0
X.loc[X['FamilySize'] == 4, 'FamSize_4'] = 1
# If FamilySize is 2 or 3, 1; else, 0
# Represents mean 55-57% survival rate
X['FamSize_2|3'] = 0
X.loc[(X['FamilySize'] == 2) | (X['FamilySize'] == 3), 'FamSize_2|3'] = 1
# Check
X.tail()
# Check percent of 'Fare' data present
print(X[X['Fare'].isnull()==True].shape[0] / X.shape[0])
# Check percent of 'Age' data present
print(X[X['Age'].isnull()==True].shape[0] / X.shape[0])
X.loc[(X['FamilySize'] == 11)]
# Find mean age of all individuals of familysize of 11
# Return mean age of FamilySize of 7
mean7 = X.loc[(X['FamilySize'] == 7), ['Age']].mean(skipna=True)
# Return mean age of FamilySize of 8
mean8 = X.loc[(X['FamilySize'] == 8), ['Age']].mean(skipna=True)
# Return mean age of FamilySize of 11
mean11 = ((mean7 + mean8) / 2)
print(mean11)
# Replace all Age data for FamilySize == 11 with 15.42
X.loc[(X['FamilySize'] == 11) & (X['Age'].isnull()), 'Age'] = 15.42
# Find average age for familysize == 1 AND sex == male
meanMale1 = X.loc[(X['FamilySize'] == 1) & (X['Sex'] == "male"), ['Age']].mean(skipna=True)
print(meanMale1)
# Replace all Age data for FamilySize == 1 AND Sex == male with 32.9
X.loc[(X['FamilySize'] == 1) & (X['Age'].isnull()) & (X['Sex'] == "male"), 'Age'] = 32.9
# Find average age for familysize == 1 AND sex == female
meanFemale1 = X.loc[(X['FamilySize'] == 1) & (X['Sex'] == "female"), ['Age']].mean(skipna=True)
print(meanFemale1)
# Replace all Age data for FamilySize == 1 AND Sex == female with 30.15
X.loc[(X['FamilySize'] == 1) & (X['Age'].isnull()) & (X['Sex'] == "female"), 'Age'] = 30.15
# Find average age for familysize == 2
mean2 = X.loc[(X['FamilySize'] == 2), ['Age']].mean(skipna=True)
print(mean2)
# Replace all Age data for FamilySize == 2 with 31.39
X.loc[(X['FamilySize'] == 2) & (X['Age'].isnull()), 'Age'] = 31.39
# Find average age for familysize == 3
mean3 = X.loc[(X['FamilySize'] == 3), ['Age']].mean(skipna=True)
print(mean3)
# Replace all Age data for FamilySize == 3 with 26.04
X.loc[(X['FamilySize'] == 3) & (X['Age'].isnull()), 'Age'] = 26.04
# Find average age for familysize == 4
mean4 = X.loc[(X['FamilySize'] == 4), ['Age']].mean(skipna=True)
print(mean4)
# Replace all Age data for FamilySize == 4 with 26.04
X.loc[(X['FamilySize'] == 4) & (X['Age'].isnull()), 'Age'] = 18.28
# Find average age for familysize == 5
mean5 = X.loc[(X['FamilySize'] == 5), ['Age']].mean(skipna=True)
print(mean5)
# Replace all Age data for FamilySize == 5 with 20.82
X.loc[(X['FamilySize'] == 5) & (X['Age'].isnull()), 'Age'] = 20.82
# Find average age for familysize == 6
mean6 = X.loc[(X['FamilySize'] == 6), ['Age']].mean(skipna=True)
print(mean6)
# Replace all Age data for FamilySize == 6 with 18.41
X.loc[(X['FamilySize'] == 6) & (X['Age'].isnull()), 'Age'] = 18.41
print(mean7)
print(mean8)
# Replace all Age data for FamilySize == 7 with 15.17
X.loc[(X['FamilySize'] == 7) & (X['Age'].isnull()), 'Age'] = 15.17
# Replace all Age data for FamilySize == 8 with 15.67
X.loc[(X['FamilySize'] == 8) & (X['Age'].isnull()), 'Age'] = 15.67
# Impact of Pclass on Survival rate
print(X[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())
# Impact of Pclass on Fare
print(X[['Pclass', 'Fare']].groupby(['Pclass'], as_index=False).mean())
# Find average Fare for Pclass == 1
mean_class1 = X.loc[(X['Pclass'] == 1), ['Fare']].mean(skipna=True)
print(mean_class1)
# Replace all Fare data for Pclass == 1 with 84.16
X.loc[(X['Pclass'] == 1) & (X['Fare'].isnull()), 'Fare'] = 84.16
# Find average Fare for Pclass == 2
mean_class2 = X.loc[(X['Pclass'] == 2), ['Fare']].mean(skipna=True)
print(mean_class2)
# Replace all Fare data for Pclass == 2 with 20.66
X.loc[(X['Pclass'] == 2) & (X['Fare'].isnull()), 'Fare'] = 20.66
# Find average Fare for Pclass == 3
mean_class3 = X.loc[(X['Pclass'] == 3), ['Fare']].mean(skipna=True)
print(mean_class3)
# Replace all Fare data for Pclass == 3 with 13.68
X.loc[(X['Pclass'] == 3) & (X['Fare'].isnull()), 'Fare'] = 13.68
# Create 'FamilySize' column that is sum of SibSp and Parch and 1
X['$*Class'] = X['Fare'] * X['Pclass']
X['$*Class'].isnull().sum()
# Check percent of 'Cabin' data present
print(X[X['Cabin'].isnull()==True].shape[0] / X.shape[0])
# Drop 'Cabin' data because there just isn't enough of it
X = X.drop(['Cabin'], axis=1)
# Check percent of 'Sex' data present
print(X[X['Sex'].isnull()==True].shape[0] / X.shape[0])
# Convert Sex data to 0's or 1's
X.loc[X.Sex != 'male', 'Sex'] = 0
X.loc[X.Sex == 'male', 'Sex'] = 1
# Check percent of 'Embarked' data present
print(X[X['Embarked'].isnull()==True].shape[0] / X.shape[0])
# Find unique Embarked values
X['Embarked'].unique()
# Fill all missing 'Embarked' data with 0 for unknown
X.loc[(X['Embarked'].isnull()), 'Embarked'] = 0
# Replace 'Embarked' == S with 1
X.loc[(X['Embarked'] == 'S'), 'Embarked'] = 1
# Replace 'Embarked' == C with 2
X.loc[(X['Embarked'] == 'C'), 'Embarked'] = 2
# Replace 'Embarked' == Q with 3
X.loc[(X['Embarked'] == 'Q'), 'Embarked'] = 3
# Check for missing data
X.isna().values.any()
# Check for missing data
X.isnull().values.any()
# Drop ticket, passengerID, pclass, name, fare, parch, sibsp
# Come back to name data to see if I can pull out titles
X = X.drop(['Ticket'], axis=1)
X = X.drop(['PassengerId'], axis=1)
X = X.drop(['Pclass'], axis=1)
X = X.drop(['Name'], axis=1)
X = X.drop(['Fare'], axis=1)
X = X.drop(['Parch'], axis=1)
X = X.drop(['SibSp'], axis=1)
X = X.drop(['FamilySize'], axis=1)
# Check for missing values
X.isnull().values.any()
# Check for missing values
X.isnull().values.any()
# Check data types
X.dtypes
# Take a peek
X.head()
# Create actual_y from survived column in train_X_df
actual_y = pd.DataFrame(X['Survived'])
# Drop that column from train_df
X = X.drop(['Survived'], axis=1)
# How many rows?
actual_y.shape
# Check for missing values
actual_y.isnull().values.any()
# Check for missing values
actual_y.isna().values.any()
# Check data types
actual_y.dtypes
# Create test_target_df from survived column in target_df
test_X = pd.DataFrame(test_df)
# Save 'PassengerId' to concatenate w/ test data output after predictions
test_X_passId = test_X.PassengerId
# Check
test_X_passId.shape
# Create 'FamilySize' column that is sum of SibSp and Parch and 1
test_X['FamilySize'] = test_X['SibSp'] + test_X['Parch'] + 1
test_X.head()
# Bin FamilySize based on mean survival rates
# If FamilySize is greater than 8, 1; else, 0
# Represents mean 0% survival rate
test_X['FamSize_8+'] = 0
test_X.loc[(test_X['FamilySize'] >= 8), 'FamSize_8+'] = 1
# If FamilySize is 1 or 7, 1; else, 0
# Represents mean 30-33% survival rate
test_X['FamSize_1|7'] = 0
test_X.loc[(test_X['FamilySize'] == 1) | (test_X['FamilySize'] == 7), 'FamSize_1|7'] = 1
# If FamilySize is 5 or 6, 1; else, 0
# Represents mean 13-20% survival rate
test_X['FamSize_5|6'] = 0
test_X.loc[(test_X['FamilySize'] == 5) | (test_X['FamilySize'] == 6), 'FamSize_5|6'] = 1
# If FamilySize is 4, 1; else, 0
# Represents mean 72% survival rate
test_X['FamSize_4'] = 0
test_X.loc[test_X['FamilySize'] == 4, 'FamSize_4'] = 1
# If FamilySize is 2 or 3, 1; else, 0
# Represents mean 55-57% survival rate
test_X['FamSize_2|3'] = 0
test_X.loc[(test_X['FamilySize'] == 2) | (test_X['FamilySize'] == 3), 'FamSize_2|3'] = 1
# Check
test_X.tail()
# Check percent of 'Fare' data present
print(test_X[test_X['Fare'].isnull()==True].shape[0] / test_X.shape[0])
# Find unique Pclass values
test_X['Pclass'].unique()
# Find average fare for pclass == 1
meanFare1 = test_X.loc[(test_X['Pclass'] == 1), ['Fare']].mean(skipna=True)
print(meanFare1)
# Replace all Fare data for Pclass == 1 with 94.28
test_X.loc[(test_X['Pclass'] == 1) & (test_X['Fare'].isnull()), 'Fare'] = 94.28
# Find average Fare for Pclass == 2
meanFare2 = test_X.loc[(test_X['Pclass'] == 2), ['Fare']].mean(skipna=True)
print(meanFare2)
# Replace all Fare data for Pclass == 2 with 20.2
test_X.loc[(test_X['Pclass'] == 2) & (test_X['Fare'].isnull()), 'Fare'] = 20.02
# Find average Fare for Pclass == 3
meanFare3 = test_X.loc[(test_X['Pclass'] == 3), ['Fare']].mean(skipna=True)
print(meanFare3)
# Replace all Fare data for Pclass == 3 with 12.46
test_X.loc[(test_X['Pclass'] == 3) & (test_X['Fare'].isnull()), 'Fare'] = 12.46
# Create 'FamilySize' column that is sum of SibSp and Parch and 1
test_X['$*Class'] = test_X['Fare'] * test_X['Pclass']
test_X['$*Class'].isnull().sum()
# Check percent of 'Age' data present
print(test_X[test_X['Age'].isnull()==True].shape[0] / test_X.shape[0])
test_X.loc[(test_X['FamilySize'] == 11)]
# Find mean age of all individuals of familysize of 11
# Return mean age of FamilySize of 7
avg7 = test_X.loc[(test_X['FamilySize'] == 7), ['Age']].mean(skipna=True)
# Return mean age of FamilySize of 8
avg8 = test_X.loc[(test_X['FamilySize'] == 8), ['Age']].mean(skipna=True)
# Return mean age of FamilySize of 11
avg11 = ((avg7 + avg8) / 2)
print(avg11)
# Replace all Age data for FamilySize == 11 with 24.5
test_X.loc[(test_X['FamilySize'] == 11) & (test_X['Age'].isnull()), 'Age'] = 24.5
# Find average age for familysize == 1 AND sex == male
avgMale1 = test_X.loc[(test_X['FamilySize'] == 1) & (test_X['Sex'] == "male"), ['Age']].mean(skipna=True)
print(avgMale1)
# Replace all Age data for FamilySize == 1 AND Sex == male with 30.26
test_X.loc[(test_X['FamilySize'] == 1) & (test_X['Age'].isnull()) & (test_X['Sex'] == "male"), 'Age'] = 30.26
# Find average age for familysize == 1 AND sex == female
avgFemale1 = test_X.loc[(test_X['FamilySize'] == 1) & (test_X['Sex'] == "female"), ['Age']].mean(skipna=True)
print(avgFemale1)
# Replace all Age data for FamilySize == 1 AND Sex == female with 29.21
test_X.loc[(test_X['FamilySize'] == 1) & (test_X['Age'].isnull()) & (test_X['Sex'] == "female"), 'Age'] = 29.21
# Find average age for familysize == 2
avg2 = test_X.loc[(test_X['FamilySize'] == 2), ['Age']].mean(skipna=True)
print(avg2)
# Replace all Age data for FamilySize == 2 with 35.5
test_X.loc[(test_X['FamilySize'] == 2) & (test_X['Age'].isnull()), 'Age'] = 35.5
# Find average age for familysize == 3
avg3 = test_X.loc[(test_X['FamilySize'] == 3), ['Age']].mean(skipna=True)
print(avg3)
# Replace all Age data for FamilySize == 3 with 27.44
test_X.loc[(test_X['FamilySize'] == 3) & (test_X['Age'].isnull()), 'Age'] = 27.44
# Find average age for familysize == 4
avg4 = test_X.loc[(test_X['FamilySize'] == 4), ['Age']].mean(skipna=True)
print(avg4)
# Replace all Age data for FamilySize == 4 with 22.01
test_X.loc[(test_X['FamilySize'] == 4) & (test_X['Age'].isnull()), 'Age'] = 22.01
# Find average age for familysize == 5
avg5 = test_X.loc[(test_X['FamilySize'] == 5), ['Age']].mean(skipna=True)
print(avg5)
# Replace all Age data for FamilySize == 5 with 29.17
test_X.loc[(test_X['FamilySize'] == 5) & (test_X['Age'].isnull()), 'Age'] = 29.17
# Find average age for familysize == 6
avg6 = test_X.loc[(test_X['FamilySize'] == 6), ['Age']].mean(skipna=True)
print(avg6)
# Replace all Age data for FamilySize == 6 with 32.67
test_X.loc[(test_X['FamilySize'] == 6) & (test_X['Age'].isnull()), 'Age'] = 32.67
print(avg7)
print(avg8)
# Replace all Age data for FamilySize == 7 with 25
test_X.loc[(test_X['FamilySize'] == 7) & (test_X['Age'].isnull()), 'Age'] = 24
# Replace all Age data for FamilySize == 8 with 25
test_X.loc[(test_X['FamilySize'] == 8) & (test_X['Age'].isnull()), 'Age'] = 25
# Replace missing Age data with average age
test_X.Age = test_X.Age.fillna(test_X['Age'].mean(skipna=True))
test_X.Age = test_X.Age.replace(0, test_X['Age'].mean(skipna=True))
# Check percent of 'Cabin' data present
print(test_X[test_X['Cabin'].isnull()==True].shape[0] / test_X.shape[0])
# Check percent of 'Sex' data present
print(test_X[test_X['Sex'].isnull()==True].shape[0] / test_X.shape[0])
# Convert Sex data to 0's or 1's
test_X.loc[test_X.Sex != 'male', 'Sex'] = 0
test_X.loc[test_X.Sex == 'male', 'Sex'] = 1
# Check percent of 'Embarked' data present
print(test_X[test_X['Embarked'].isnull()==True].shape[0] / test_X.shape[0])
# Fill all missing 'Embarked' data with 0 for unknown
test_X.loc[(test_X['Embarked'].isnull()), 'Embarked'] = 0
# Replace 'Embarked' == S with 1
test_X.loc[(test_X['Embarked'] == 'S'), 'Embarked'] = 1
# Replace 'Embarked' == C with 2
test_X.loc[(test_X['Embarked'] == 'C'), 'Embarked'] = 2
# Replace 'Embarked' == Q with 3
test_X.loc[(test_X['Embarked'] == 'Q'), 'Embarked'] = 3
# Drop ticket, pclass, name, fare, parch, sibsp, cabin
# Come back to name data to see if I can pull out titles
test_X = test_X.drop(['Ticket'], axis=1)
test_X = test_X.drop(['Pclass'], axis=1)
test_X = test_X.drop(['Name'], axis=1)
test_X = test_X.drop(['Fare'], axis=1)
test_X = test_X.drop(['Parch'], axis=1)
test_X = test_X.drop(['SibSp'], axis=1)
test_X = test_X.drop(['FamilySize'], axis=1)
test_X = test_X.drop(['Cabin'], axis=1)
test_X = test_X.drop(['PassengerId'], axis=1)
# Take a peek
test_X.head()
# Check for missing values
test_X.isnull().values.any()
# Check for missing values
test_X.isna().values.any()
# Set up grid search, 5 folds
model_5fold = model = tree.DecisionTreeClassifier()
param_grid = {'max_depth': list(range(1,11)),
              'criterion': ['entropy', 'gini']
              }
grid_5fold = GridSearchCV(model_5fold, param_grid, cv=5)
# Perform grid search 
grid_5fold.fit(X, actual_y)
# Print out best parameters
print("Best parameters: {}".format(grid_5fold.best_params_))
# Get the accuracy
# Evaluate the tree       
y = grid_5fold.best_estimator_.predict(X)
# Print accuracy          
print("Accuracy: {}".format(accuracy_score(actual_y, y)))
# Add test Passenger ID's to output dataframe
back_to_CV_GS_results = pd.DataFrame(test_X_passId)
# Create 'Survived' column to store predicted values
back_to_CV_GS_results.insert(1,'Survived', np.nan)
# Add test Passenger ID's to output dataframe
titanic_CV_GS_results = pd.DataFrame(test_X_passId)
# Create 'Survived' column to store predicted values
titanic_CV_GS_results.insert(1,'Survived', np.nan)
# Run prediction and place output in 'Survived' column
titanic_CV_GS_results.Survived = grid_5fold.best_estimator_.predict(test_X)
# Take a peek
titanic_CV_GS_results.head()
# Save output, test_predict_dt15, to .csv for submission
titanic_CV_GS_results.to_csv('titanic_CV_GS_results.csv', index = False)
