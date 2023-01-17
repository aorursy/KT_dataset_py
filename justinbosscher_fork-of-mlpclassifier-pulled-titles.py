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
from sklearn.neural_network import MLPClassifier
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
# Create total df from which mean values will be calculated
combine = pd.DataFrame(train_orig)
combine = combine.append(test_orig, sort=False)
combine.shape
# How many rows / columns ?
combine.shape
# How many rows / columns ?
combine.shape
# Take a peek at the head of train_df
combine.head()
# Take a look at number of null values
combine.info()
# Take a look at number of null values
combine.info()
# Descriptive stats
combine.describe()
# Take a peek at the head of test_df
combine.head()
# Scatter matrix
scatters_combine = pd.plotting.scatter_matrix(combine, figsize=[40,40])
# Plot correlations as heatmap
# Adapted from here: https://www.kaggle.com/foutik/decision-tree
corr = combine.corr()
_ , ax = plt.subplots( figsize =( 12 , 10 ) )
cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
_ = sns.heatmap(corr, cmap = cmap, square=True, cbar_kws={ 'shrink' : .9 }, ax=ax, annot = True, annot_kws = {'fontsize' : 12 })
# Impact of Sex on survival rate
# Adapted from: https://towardsdatascience.com/play-with-data-2a5db35b279c
print(combine[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean())
# Impact of Embarked on Survival rate
print(combine[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())
# Impact of SibSp on Survival rate
print(combine[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean())
# Impact of Embarked on Survival rate
print(combine[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean())
# Impact of Sex on Age
print(combine[['Sex', 'Age']].groupby(['Sex'], as_index=False).mean())
# Check percent of 'Cabin' data present
print(combine[combine['Cabin'].isnull()==True].shape[0] / combine.shape[0])
# Create hasCabin column; passenger has a cabin = 1, else 0
combine.loc[(combine['Cabin']).isna(), 'hasCabin'] = 0
combine.loc[(combine['Cabin']).notna(), 'hasCabin'] = 1
combine.head(5)
# Impact of hasCabin on Survived
print(combine[['hasCabin', 'Survived']].groupby(['hasCabin'], as_index=False).mean())
# Function to return the title of a name;
# Return 'U' if there is no title
# Adapted from here: https://www.kaggle.com/manuelatadvice/feature-engineering-titles
def get_title(name):
    if '.' in name:
        return name.split(',')[1].split('.')[0].strip()
    else:
        return 'U'
# Create column of titles for each dataframe
combine['Title'] = combine['Name'].map(lambda x: get_title(x))
# Take a look at the impact of titles on survival
print(combine[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())
# Take a look at the impact of titles on age
print(combine[['Title', 'Age']].groupby(['Title'], as_index=False).mean())
# How many people lack titles?
combine.Title.isna().sum()
# Can crew members be identified by a combination of title and fare
# What does the Fare data look like for crew members?
combine.loc[combine['Title'] == 'Capt']
# Well,if the captain paid to get on board, the other crew members probably did, too.
# Identify crew members
# If a passenge isCrew, 1 else 0
combine['isCrew'] = 0
combine.loc[(combine['Title'] == 'Captain') | (combine['Title'] == 'Col') | (combine['Title'] == 'Major'), 'isCrew'] = 1

# Take a look at the impact of staff on survived
print(combine[['isCrew', 'Survived']].groupby(['isCrew'], as_index=False).mean())
# Identify nobility
# If a passenge isNoble, 1 else 0
combine['isNoble'] = 0
combine.loc[(combine['Title'] == 'Don') | (combine['Title'] == 'Jonkheer')
             | (combine['Title'] == 'Sir') | (combine['Title'] == 'Count')
             | (combine['Title'] == 'the Countess') | (combine['Title'] == 'Lady'),
             'isNoble'] = 1

# Take a look at the impact of staff on survived
print(combine[['isNoble', 'Survived']].groupby(['isNoble'], as_index=False).mean())
# How many Dr's and Rev's are there?
# I would assume that there might be 1 of each as crew members
# If there is only one Dr., assume that it is a staff member and that the rest are passengers
combine.Title.value_counts()
# What is the impact of isCrew == 1 on Fare
print(combine[['isCrew', 'Fare']].groupby(['isCrew'], as_index=False).mean())
# Create 'FamilySize' column that is sum of SibSp and Parch and 1
combine['FamilySize'] = combine['SibSp'] + combine['Parch'] + 1
combine.head()
# Impact of Sex on Age
print(combine[['Sex', 'Age']].groupby(['Sex'], as_index=False).mean())
# Bin FamilySize based on mean survival rates
# If FamilySize is greater than 8, 1; else, 0
# Represents mean 0% survival rate
combine['FamSize_8+'] = 0
combine.loc[(combine['FamilySize'] >= 8), 'FamSize_8+'] = 1
# If FamilySize is 1 or 7, 1; else, 0
# Represents mean 30-33% survival rate
combine['FamSize_1|7'] = 0
combine.loc[(combine['FamilySize'] == 1) | (combine['FamilySize'] == 7), 'FamSize_1|7'] = 1
# If FamilySize is 5 or 6, 1; else, 0
# Represents mean 13-20% survival rate
combine['FamSize_5|6'] = 0
combine.loc[(combine['FamilySize'] == 5) | (combine['FamilySize'] == 6), 'FamSize_5|6'] = 1
# If FamilySize is 4, 1; else, 0
# Represents mean 72% survival rate
combine['FamSize_4'] = 0
combine.loc[combine['FamilySize'] == 4, 'FamSize_4'] = 1
# If FamilySize is 2 or 3, 1; else, 0
# Represents mean 55-57% survival rate
combine['FamSize_2|3'] = 0
combine.loc[(combine['FamilySize'] == 2) | (combine['FamilySize'] == 3), 'FamSize_2|3'] = 1
# Check
combine.tail()
# Check percent of 'Age' data present
print(combine[combine['Age'].isnull()==True].shape[0] / combine.shape[0])
# Why is mean age for FamilySize == 11 NaN?
combine.loc[(combine['FamilySize'] == 11)]
# Impact of FamilySize on Age
print(combine[['FamilySize', 'Age']].groupby(['FamilySize'], as_index=False).mean())
# Impact of Title on Age
print(combine[['Title', 'Age']].groupby(['Title'], as_index=False).mean())
# How many people lack age data?
combine.Age.isna().sum()
# Which titles have missing age data?
#combine.loc[(combine['FamilySize'] == 11)]
age_na_df = pd.DataFrame(combine.loc[(combine['Age'].isna())])
age_na_df.shape
# Find those titles that have missing age data so that I don't have to go through each one
# when calculating mean values
age_na_df.Title.unique().sum()
# Find those family sizes that have missing age data so that I don't have to go through each one
# when calculating mean values
age_na_df.FamilySize.unique()
#### Return mean age values for FamilySize == 4 and Title == Mr, Mrs, Miss, Master, Dr, Ms
# Mr, 31.64
print("Mr:\n")
print(combine.loc[(combine['FamilySize'] == 1) & (combine['Title'] == "Mr"), ['Age']].mean(skipna=True))
combine.loc[(combine['FamilySize'] == 1) & (combine['Title'] == "Mr") & (combine['Age'].isnull()), 'Age'] = 34.78
# Mrs, 40.67
print("Mrs:\n")
print(combine.loc[(combine['FamilySize'] == 1) & (combine['Title'] == "Mrs"), ['Age']].mean(skipna=True))
combine.loc[(combine['FamilySize'] == 1) & (combine['Title'] == "Mrs") & (combine['Age'].isnull()), 'Age'] = 40.67
# Miss, 27.23
print("Miss:\n")
print(combine.loc[(combine['FamilySize'] == 1) & (combine['Title'] == "Miss"), ['Age']].mean(skipna=True))
combine.loc[(combine['FamilySize'] == 1) & (combine['Title'] == "Miss") & (combine['Age'].isnull()), 'Age'] = 27.23
# Master, 5.65 ; Using mean age value from FamilySize == 2
print("Master:\n")
print(combine.loc[(combine['FamilySize'] == 2) & (combine['Title'] == "Master"), ['Age']].mean(skipna=True))
combine.loc[(combine['FamilySize'] == 1) & (combine['Title'] == "Master") & (combine['Age'].isnull()), 'Age'] = 5.65
# Dr, 39.50
print("Dr:\n")
print(combine.loc[(combine['FamilySize'] == 1) & (combine['Title'] == "Dr"), ['Age']].mean(skipna=True))
combine.loc[(combine['FamilySize'] == 1) & (combine['Title'] == "Dr") & (combine['Age'].isnull()), 'Age'] = 39.50
# Ms, 39.50
print("Ms:\n")
print(combine.loc[(combine['FamilySize'] == 1) & (combine['Title'] == "Ms"), ['Age']].mean(skipna=True))
combine.loc[(combine['FamilySize'] == 1) & (combine['Title'] == "Ms") & (combine['Age'].isnull()), 'Age'] = 39.50
#### Return mean age values for FamilySize == 4 and Title == Mr, Mrs, Miss, Master, Dr, Ms
# Mr, 34.77
print("Mr:\n")
print(combine.loc[(combine['FamilySize'] == 2) & (combine['Title'] == "Mr"), ['Age']].mean(skipna=True))
combine.loc[(combine['FamilySize'] == 2) & (combine['Title'] == "Mr") & (combine['Age'].isnull()), 'Age'] = 34.78
# Mrs, 35.84
print("Mrs:\n")
print(combine.loc[(combine['FamilySize'] == 2) & (combine['Title'] == "Mrs"), ['Age']].mean(skipna=True))
combine.loc[(combine['FamilySize'] == 2) & (combine['Title'] == "Mrs") & (combine['Age'].isnull()), 'Age'] = 35.84
# Miss, 21.18
print("Miss:\n")
print(combine.loc[(combine['FamilySize'] == 2) & (combine['Title'] == "Miss"), ['Age']].mean(skipna=True))
combine.loc[(combine['FamilySize'] == 2) & (combine['Title'] == "Miss") & (combine['Age'].isnull()), 'Age'] = 21.18
# Master, 5.65
print("Master:\n")
print(combine.loc[(combine['FamilySize'] == 2) & (combine['Title'] == "Master"), ['Age']].mean(skipna=True))
combine.loc[(combine['FamilySize'] == 2) & (combine['Title'] == "Master") & (combine['Age'].isnull()), 'Age'] = 5.65
# Dr, 49.00 ; Using mean age value from FamilySize = 3
print("Dr:\n")
print(combine.loc[(combine['FamilySize'] == 3) & (combine['Title'] == "Dr"), ['Age']].mean(skipna=True))
combine.loc[(combine['FamilySize'] == 2) & (combine['Title'] == "Dr") & (combine['Age'].isnull()), 'Age'] = 49.00
# Ms, 33.75
print("Ms:\n")
print(combine.loc[(combine['FamilySize'] == 1) & (combine['Title'] == "Ms"), ['Age']].mean(skipna=True))
combine.loc[(combine['FamilySize'] == 2) & (combine['Title'] == "Ms") & (combine['Age'].isnull()), 'Age'] = 33.75
#### Return mean age values for FamilySize == 4 and Title == Mr, Mrs, Miss, Master, Dr, Ms
# Mr, 33.16
print("Mr:\n")
print(combine.loc[(combine['FamilySize'] == 3) & (combine['Title'] == "Mr"), ['Age']].mean(skipna=True))
combine.loc[(combine['FamilySize'] == 3) & (combine['Title'] == "Mr") & (combine['Age'].isnull()), 'Age'] = 33.16
# Mrs, 36.91
print("Mrs:\n")
print(combine.loc[(combine['FamilySize'] == 3) & (combine['Title'] == "Mrs"), ['Age']].mean(skipna=True))
combine.loc[(combine['FamilySize'] == 3) & (combine['Title'] == "Mrs") & (combine['Age'].isnull()), 'Age'] = 36.91
# Miss, 12.68
print("Miss:\n")
print(combine.loc[(combine['FamilySize'] == 3) & (combine['Title'] == "Miss"), ['Age']].mean(skipna=True))
combine.loc[(combine['FamilySize'] == 3) & (combine['Title'] == "Miss") & (combine['Age'].isnull()), 'Age'] = 12.68
# Master, 4.68
print("Master:\n")
print(combine.loc[(combine['FamilySize'] == 3) & (combine['Title'] == "Master"), ['Age']].mean(skipna=True))
combine.loc[(combine['FamilySize'] == 3) & (combine['Title'] == "Master") & (combine['Age'].isnull()), 'Age'] = 4.68
# Dr, 49.00
print("Dr:\n")
print(combine.loc[(combine['FamilySize'] == 3) & (combine['Title'] == "Dr"), ['Age']].mean(skipna=True))
combine.loc[(combine['FamilySize'] == 3) & (combine['Title'] == "Dr") & (combine['Age'].isnull()), 'Age'] = 49.00
# Ms, 28.00 ; Using mean age value for all FamilySizes
print("Ms:\n")
print(combine.loc[(combine['Title'] == "Ms"), ['Age']].mean(skipna=True))
combine.loc[(combine['FamilySize'] == 3) & (combine['Title'] == "Ms") & (combine['Age'].isnull()), 'Age'] = 28.00
#### Return mean age values for FamilySize == 4 and Title == Mr, Mrs, Miss, Master, Dr, Ms
# Mr, 27.00 ; Using age data from FamilySize == 8
print("Mr:\n")
print(combine.loc[(combine['FamilySize'] == 8) & (combine['Title'] == "Mr"), ['Age']].mean(skipna=True))
combine.loc[(combine['FamilySize'] == 11) & (combine['Title'] == "Mr") & (combine['Age'].isnull()), 'Age'] = 27.00
# Mrs, 43.00 ; Using the age data from FamilySize == 8
print("Mrs:\n")
print(combine.loc[(combine['FamilySize'] == 8) & (combine['Title'] == "Mrs"), ['Age']].mean(skipna=True))
combine.loc[(combine['FamilySize'] == 11) & (combine['Title'] == "Mrs") & (combine['Age'].isnull()), 'Age'] = 43.00
# Miss, 13.00 ; Using the age data from FamilySize == 8
print("Miss:\n")
print(combine.loc[(combine['FamilySize'] == 8) & (combine['Title'] == "Miss"), ['Age']].mean(skipna=True))
combine.loc[(combine['FamilySize'] == 11) & (combine['Title'] == "Miss") & (combine['Age'].isnull()), 'Age'] = 13.00
# Master, 7.00 ; Using the age data from FamilySize == 8
print("Master:\n")
print(combine.loc[(combine['FamilySize'] == 8) & (combine['Title'] == "Master"), ['Age']].mean(skipna=True))
combine.loc[(combine['FamilySize'] == 11) & (combine['Title'] == "Master") & (combine['Age'].isnull()), 'Age'] = 7.00
# Dr, 49.00 ; Using mean age value from FamilySize == 3
print("Dr:\n")
print(combine.loc[(combine['FamilySize'] == 3) & (combine['Title'] == "Dr"), ['Age']].mean(skipna=True))
combine.loc[(combine['FamilySize'] == 11) & (combine['Title'] == "Dr") & (combine['Age'].isnull()), 'Age'] = 49.00
# Ms, 28.00 ; using the average from across all family size values b/c there just doesn't seem to be enough
print("Ms:\n")
print(combine.loc[(combine['Title'] == "Ms"), ['Age']].mean(skipna=True))
combine.loc[(combine['FamilySize'] == 11) & (combine['Title'] == "Ms") & (combine['Age'].isnull()), 'Age'] = 28.00
#### Return mean age values for FamilySize == 4 and Title == Mr, Mrs, Miss, Master, Dr, Ms
# Mr, 31.00
print("Mr:\n")
print(combine.loc[(combine['FamilySize'] == 5) & (combine['Title'] == "Mr"), ['Age']].mean(skipna=True))
combine.loc[(combine['FamilySize'] == 5) & (combine['Title'] == "Mr") & (combine['Age'].isnull()), 'Age'] = 31.00
# Mrs, 44.75
print("Mrs:\n")
print(combine.loc[(combine['FamilySize'] == 5) & (combine['Title'] == "Mrs"), ['Age']].mean(skipna=True))
combine.loc[(combine['FamilySize'] == 5) & (combine['Title'] == "Mrs") & (combine['Age'].isnull()), 'Age'] = 44.75
# Miss, 13.33
print("Miss:\n")
print(combine.loc[(combine['FamilySize'] == 5) & (combine['Title'] == "Miss"), ['Age']].mean(skipna=True))
combine.loc[(combine['FamilySize'] == 5) & (combine['Title'] == "Miss") & (combine['Age'].isnull()), 'Age'] = 13.33
# Master, 7.00
print("Master:\n")
print(combine.loc[(combine['FamilySize'] == 5) & (combine['Title'] == "Master"), ['Age']].mean(skipna=True))
combine.loc[(combine['FamilySize'] == 5) & (combine['Title'] == "Master") & (combine['Age'].isnull()), 'Age'] = 7.00
# Dr, 49.00 ; using the average age of Dr's from family size of 3
print("Dr:\n")
print(combine.loc[(combine['FamilySize'] == 3) & (combine['Title'] == "Dr"), ['Age']].mean(skipna=True))
combine.loc[(combine['FamilySize'] == 5) & (combine['Title'] == "Dr") & (combine['Age'].isnull()), 'Age'] = 49.00
# Ms, 28.00 ; using the average from across all family size values b/c there just doesn't seem to be enough
print("Ms:\n")
print(combine.loc[(combine['Title'] == "Ms"), ['Age']].mean(skipna=True))
combine.loc[(combine['FamilySize'] == 5) & (combine['Title'] == "Ms") & (combine['Age'].isnull()), 'Age'] = 28.00
#### Return mean age values for FamilySize == 4 and Title == Mr, Mrs, Miss, Master, Dr, Ms
# Mr, 32.00
print("Mr:\n")
print(combine.loc[(combine['FamilySize'] == 4) & (combine['Title'] == "Mr"), ['Age']].mean(skipna=True))
combine.loc[(combine['FamilySize'] == 4) & (combine['Title'] == "Mr") & (combine['Age'].isnull()), 'Age'] = 32.00
# Mrs, 30.92
print("Mrs:\n")
print(combine.loc[(combine['FamilySize'] == 4) & (combine['Title'] == "Mrs"), ['Age']].mean(skipna=True))
combine.loc[(combine['FamilySize'] == 4) & (combine['Title'] == "Mrs") & (combine['Age'].isnull()), 'Age'] = 30.92
# Miss, 7.77
print("Miss:\n")
print(combine.loc[(combine['FamilySize'] == 4) & (combine['Title'] == "Miss"), ['Age']].mean(skipna=True))
combine.loc[(combine['FamilySize'] == 4) & (combine['Title'] == "Miss") & (combine['Age'].isnull()), 'Age'] = 7.77
# Master, 3.48
print("Master:\n")
print(combine.loc[(combine['FamilySize'] == 4) & (combine['Title'] == "Master"), ['Age']].mean(skipna=True))
combine.loc[(combine['FamilySize'] == 4) & (combine['Title'] == "Master") & (combine['Age'].isnull()), 'Age'] = 3.48
# Dr, 49.00 ; using the average age of Dr's from family size of 3
print("Dr:\n")
print(combine.loc[(combine['FamilySize'] == 3) & (combine['Title'] == "Dr"), ['Age']].mean(skipna=True))
combine.loc[(combine['FamilySize'] == 4) & (combine['Title'] == "Dr") & (combine['Age'].isnull()), 'Age'] = 49.00
# Ms, 28.00 ; using the average from across all family size values b/c there just doesn't seem to be enough
print("Ms:\n")
print(combine.loc[(combine['Title'] == "Ms"), ['Age']].mean(skipna=True))
combine.loc[(combine['FamilySize'] == 4) & (combine['Title'] == "Ms") & (combine['Age'].isnull()), 'Age'] = 28.00
# How many people lack age data?
combine.Age.isna().sum()
combine.loc[(combine['FamilySize'] == 1) & (combine['Title'] == "Mr") & (combine['Age'].isnull())]
# Check percent of 'Age' data present
print(combine[combine['Age'].isnull()==True].shape[0] / combine.shape[0])
# Check percent of 'Fare' data present
print(combine[combine['Fare'].isnull()==True].shape[0] / combine.shape[0])
# Print missing Fare data rows
combine.loc[(combine['Fare'].isna())]
# Impact of Pclass on Survived
print(combine[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())
# Impact of Pclass on Fare
print(combine[['Pclass', 'Fare']].groupby(['Pclass'], as_index=False).mean())
# Find average Fare for Pclass == 2
mean_class1 = combine.loc[(combine['Pclass'] == 1), ['Fare']].mean(skipna=True)
print(mean_class1)
# Replace all Fare data for Pclass == 1 with 87.51
combine.loc[(combine['Pclass'] == 1) & (combine['Fare'].isnull()), 'Fare'] = 87.51
# Find average Fare for Pclass == 2
mean_class2 = combine.loc[(combine['Pclass'] == 2), ['Fare']].mean(skipna=True)
print(mean_class2)
# Replace all Fare data for Pclass == 2 with 21.78
combine.loc[(combine['Pclass'] == 2) & (combine['Fare'].isnull()), 'Fare'] = 21.78
# Find average Fare for Pclass == 3
mean_class3 = combine.loc[(combine['Pclass'] == 3), ['Fare']].mean(skipna=True)
print(mean_class3)
# Replace all Fare data for Pclass == 3 with 13.30
combine.loc[(combine['Pclass'] == 3) & (combine['Fare'].isnull()), 'Fare'] = 13.30
# Create 'Fare_x_Class' column that is product of Fare and Pclass
#combine['Fare_x_Class'] = combine['Fare'] * combine['Pclass']
#print(combine['Fare_x_Class'].isnull().sum())
#print(combine['Fare_x_Class'].isna().sum())
# Check percent of 'Sex' data present
print(combine[combine['Sex'].isnull()==True].shape[0] / combine.shape[0])
# Convert Sex data to 0's or 1's
combine.loc[combine.Sex != 'male', 'Sex'] = 0
combine.loc[combine.Sex == 'male', 'Sex'] = 1
# Check percent of 'Embarked' data present
print(combine[combine['Embarked'].isnull()==True].shape[0] / combine.shape[0])
# Find unique Embarked values
combine['Embarked'].unique()
# Fill all missing 'Embarked' data with 0 for unknown
combine.loc[(combine['Embarked'].isnull()), 'Embarked'] = 0
# Replace 'Embarked' == S with 1
combine.loc[(combine['Embarked'] == 'S'), 'Embarked'] = 1
# Replace 'Embarked' == C with 2
combine.loc[(combine['Embarked'] == 'C'), 'Embarked'] = 2
# Replace 'Embarked' == Q with 3
combine.loc[(combine['Embarked'] == 'Q'), 'Embarked'] = 3
# Drop ticket, passengerID, pclass, name, fare, parch, sibsp
# Come back to name data to see if I can pull out titles
combine = combine.drop(['Ticket'], axis=1)
combine = combine.drop(['Pclass'], axis=1)
combine = combine.drop(['Name'], axis=1)
combine = combine.drop(['Title'], axis=1)
combine = combine.drop(['Fare'], axis=1)
combine = combine.drop(['Parch'], axis=1)
combine = combine.drop(['SibSp'], axis=1)
combine = combine.drop(['Cabin'], axis=1)
combine = combine.drop(['FamilySize'], axis=1)
# Create X df
X = combine[combine['Survived'].notna()]
# Drop PassengerID
X = X.drop(['PassengerId'], axis=1)
# Take a peek
X.head()
# Create test df# Create X df
test = combine[combine['Survived'].isna()]
# Save PassengerId's for later
test_ID = pd.DataFrame(test['PassengerId'])
# Drop passenger ID's
test = test.drop(['PassengerId'], axis=1)
test = test.drop(['Survived'], axis=1)
# Take a peek
test.head()
# Check shape
test.shape
# Create actual_y from survived column in X
actual_y = X['Survived']
# Drop that column from X
X = X.drop(['Survived'], axis=1)
# How many rows?
actual_y.shape
# Check for missing values
actual_y.isnull().values.any()
# Check for missing values
actual_y.isna().values.any()
# Check data types
actual_y.dtypes
# Set up neural network
model = MLPClassifier(max_iter=2000)
# Grid Search
param_grid = {'hidden_layer_sizes': [ (5,), (6,), (7,), (8,), (9,), (10,),
                        (11,), (12,), (13,), (14,), (15,), (16,),
                        (17,), (18,), (19,), (20,)]}
grid = GridSearchCV(model,param_grid,cv=5)
grid.fit(X,actual_y)
print("Grid Search: best parameters:{}".format(grid.best_params_))
# Evaluate best model
best_model = grid.best_estimator_

predict_y = pd.DataFrame(best_model.predict(X)).astype(int)
print("Accuracy:{}".format(accuracy_score(actual_y,predict_y)))
# Add test Passenger ID's to output dataframe
results = pd.DataFrame(test_ID)
# Run prediction and to store values predict_y
predict_y = best_model.predict(test).astype(int)
# Insert predicted values to results dataframe
results.insert(1,'Survived', predict_y)
results.head()
# Save output, test_predict_dt15, to .csv for submission
results.to_csv('results.csv', index = False)