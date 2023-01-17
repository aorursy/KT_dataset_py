# import libraries

import pandas as pd

import numpy as np
train = pd.read_csv(r'/kaggle/input/titanic/train.csv')

test = pd.read_csv(r'/kaggle/input/titanic/test.csv')
train.head(5)
train.isna().sum()
dfs = [train ,test]



for df in dfs:

    df['Age'].fillna(df['Age'].median(), inplace = True)
train.isna().sum()
train['Cabin'].value_counts()
train['Cabin'].unique()
for df in dfs:

    df['Cabin'].fillna(0)
cabins = []

for i in train['Cabin']:

    cabins.append(str(i))
letters = []

for i in cabins:

    letter= i[0]

    letters.append(letter)
train['Cabin'] = letters
cabins = []

for i in test['Cabin']:

    cabins.append(str(i))
letters = []

for i in cabins:

    letter = i[0]

    letters.append(letter)
test['Cabin'] = letters
train['Cabin'].head()
train['Embarked'].value_counts()
for df in dfs:

    df['Embarked'].fillna('S')
import seaborn as sns

import matplotlib.pyplot as plt

#seaborn & matplotlib are excellent python libraries to perform clean visualizations.

#I highly suggest you get familiar with them!



#correlation matrix 

corr_matrix = train.corr()

fig, ax = plt.subplots(figsize = (10,8))

sns.heatmap(corr_matrix, annot = True, fmt='.2g', vmin = -1,

            vmax = 1, center = 0, cmap = 'coolwarm')
train.dtypes
#boxplot

numeric_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

fig, ax = plt.subplots(figsize = (10,5))

sns.boxplot(data = train[numeric_cols], orient = 'h', palette = 'Set2')
from pandas.plotting import scatter_matrix

scatter_matrix(train[numeric_cols], figsize= (12,8))
train.hist(bins = 20, figsize = (12,8))
sns.countplot(train[train['Survived'] == 1]['Pclass']).set_title('Count Survived for each Class')
len(train[train['Pclass'] == 1]), len(train[train['Pclass'] == 2]), len(train[train['Pclass'] == 3])
train[train['Pclass'] == 1]['Survived'].sum(), train[train['Pclass'] == 2]['Survived'].sum(), train[train['Pclass'] == 3]['Survived'].sum()   
percentages = []

first = 136 / 216

second = 87/ 184

third = 119/491

percentages.append(first)

percentages.append(second)

percentages.append(third)
percents = pd.DataFrame(percentages)

percents.index+=1
percents['PClass'] = ['1', '2', '3']

cols= ['Percent', 'PClass']

percents.columns = [i for i in cols]

sns.barplot(y = 'Percent', x = 'PClass', data = percents).set_title('Percent Survived for Passenger Class')
train['Family'] = train.apply(lambda x: x['SibSp'] + x['Parch'], axis = 1)

test['Family'] = test.apply(lambda x: x['SibSp'] + x['Parch'], axis = 1)
#dropping columns from the dataframe 

train.drop(['SibSp', 'Parch', 'Name', 'Ticket'], axis = 1, inplace = True)

test.drop(['SibSp', 'Parch', 'Name', 'Ticket'], axis = 1, inplace = True)
train.head(5)
test.isna().sum()
test['Fare'].fillna(test['Fare'].median(), inplace = True)
train_df = pd.get_dummies(train)

test_df = pd.get_dummies(test)
#axis 1 refers to columns!

train_df.drop('PassengerId', axis = 1, inplace = True)
y = train_df['Survived']

train_df.drop('Survived', axis = 1, inplace = True)

train_df.drop('Cabin_T', axis = 1, inplace = True)

test_df.drop('PassengerId', axis = 1, inplace = True)
X_test = test_df

X_train = train_df
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier
param_grid = {

    'n_estimators': [200, 500, 1000],

    'max_features': ['auto'],

    'max_depth': [6, 7, 8],

    'criterion': ['entropy']

}
CV = GridSearchCV(estimator = rfc, param_grid = param_grid, cv = 5)

CV.fit(X_train, y)

CV.best_estimator_
rfc.fit(X_train, y)
y_pred = rfc.predict(X_test)
#we can reshape this array using .reshape(-1, 1)

y_pred
#reshape array so that it can be used in a dataframe for easy submission!

submission1 = y_pred.reshape(-1, 1)
sub_df = pd.DataFrame(submission1)
sub_df['PassengerId'] = test['PassengerId']

sub_df['Survived'] = submission1

cols = ['PassengerId',

       'Survived']

sub_df.drop(0, axis = 1, inplace = True)

sub_df.columns = [i for i in cols]

sub_df = sub_df.set_index('PassengerId')
sub_df.head(10)
#put file path in string!

sub_df.to_csv(r'submission13.csv')