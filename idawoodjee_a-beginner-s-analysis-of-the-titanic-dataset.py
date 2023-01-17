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
# Import data analysis libraries and future

import __future__

import numpy as np

import pandas as pd
# Import visualization libraries and set favourite grid style

import seaborn as sns

import matplotlib.pyplot as plt

sns.set_style('whitegrid')
# Read in the dataset

titanic_train = pd.read_csv('../input/titanic/train.csv')
# Check out the first 10 rows of the training dataset

titanic_train.head(10)
# Info about data frame dimensions, column types, and file size

titanic_train.info()
# Generate a summary of statistics for each numerical column of the data frame

titanic_train.describe()
titanic_train.nunique()
lessthanten = []

for col in titanic_train.columns:

    lessthanten.append(titanic_train[col].nunique() < 10)



for col in titanic_train[titanic_train.columns[lessthanten]]:

    print(col, titanic_train[col].unique())
titanic_train['Embarked'].value_counts()
pd.DataFrame({'Integer': ['Survived','Pclass','SibSp, Parch','-'], 

              'Float': ['-','-','-','Age, Fare'], 

              'Object': ['Sex, Name, Ticket, Cabin, Embarked','-','-','-']}, 

              index = ['Nominal','Ordinal','Discrete','Continuous'])
# First 10 rows of the data

titanic_train.head(10)
# Define the count_n_plot function

def count_n_plot(df, col_name, countsplit = None, bar = False, barsplit = None):

    

    """

    Creates countplots and barplots of the specified feature 

    (with options to split the columns) and generates the 

    corresponding table of counts and percentages.

    

    Parameters

    ----------

    df : DataFrame

        Dataset for plotting.

    col_name : string

        Name of column/feature in "data".

    countsplit : string

        Use countsplit to specify the "hue" argument of the countplot.

    bar : Boolean

        If True, a barplot of the column col_name is created, showing

        the fraction of survivors on the y-axis.

    barsplit: string

        Use barsplit to specify the "hue" argument of the barplot.

    """

    

    if (countsplit != None) & bar & (barsplit != None):

        col_count1 = df[[col_name]].groupby(by = col_name).size()

        col_perc1 = col_count1.apply(lambda x: x / sum(col_count1) * 100).round(1)

        tcount1 = pd.DataFrame({'Count': col_count1, 'Percentage': col_perc1})

        

        col_count2 = df[[col_name,countsplit]].groupby(by = [col_name,countsplit]).size()

        col_perc2 = col_count2.apply(lambda x: x / sum(col_count2) * 100).round(1)

        tcount2 = pd.DataFrame({'Count': col_count2, 'Percentage': col_perc2})

        display(tcount1, tcount2) 

        

        figc, axc = plt.subplots(1, 2, figsize = (10,4))

        sns.countplot(data = df, x = col_name, hue = None, ax = axc[0])

        sns.countplot(data = df, x = col_name, hue = countsplit, ax = axc[1])

        

        figb, axb = plt.subplots(1, 2, figsize = (10,4))

        sns.barplot(data = df, x = col_name, y = 'Survived', hue = None, ax = axb[0])

        sns.barplot(data = df, x = col_name, y = 'Survived', hue = barsplit, ax = axb[1])

        

    elif (countsplit != None) & bar:

        col_count1 = df[[col_name]].groupby(by = col_name).size()

        col_perc1 = col_count1.apply(lambda x: x / sum(col_count1) * 100).round(1)

        tcount1 = pd.DataFrame({'Count': col_count1, 'Percentage': col_perc1})

        

        col_count2 = df[[col_name,countsplit]].groupby(by = [col_name,countsplit]).size()

        col_perc2 = col_count2.apply(lambda x: x / sum(col_count2) * 100).round(1)

        tcount2 = pd.DataFrame({'Count': col_count2, 'Percentage': col_perc2})

        display(tcount1, tcount2)

        

        fig, axes = plt.subplots(1, 3, figsize = (15,4))

        sns.countplot(data = df, x = col_name, hue = None, ax = axes[0])

        sns.countplot(data = df, x = col_name, hue = countsplit, ax = axes[1])

        sns.barplot(data = df, x = col_name, y = 'Survived', hue = None, ax = axes[2])

        

    elif countsplit != None:

        col_count1 = df[[col_name]].groupby(by = col_name).size()

        col_perc1 = col_count1.apply(lambda x: x / sum(col_count1) * 100).round(1)

        tcount1 = pd.DataFrame({'Count': col_count1, 'Percentage': col_perc1})

        

        col_count2 = df[[col_name,countsplit]].groupby(by = [col_name,countsplit]).size()

        col_perc2 = col_count2.apply(lambda x: x / sum(col_count2) * 100).round(1)

        tcount2 = pd.DataFrame({'Count': col_count2, 'Percentage': col_perc2})

        display(tcount1, tcount2)

        

        fig, axes = plt.subplots(1, 2, figsize = (10,4))

        sns.countplot(data = df, x = col_name, hue = None, ax = axes[0])

        sns.countplot(data = df, x = col_name, hue = countsplit, ax = axes[1])

        

    else:

        col_count = df[[col_name]].groupby(by = col_name).size()

        col_perc = col_count.apply(lambda x: x / sum(col_count) * 100).round(1)

        tcount1 = pd.DataFrame({'Count': col_count, 'Percentage': col_perc})

        display(tcount1)        

        

        sns.countplot(data = df, x = col_name)
count_n_plot(df = titanic_train, col_name = 'Survived')
count_n_plot(df = titanic_train, col_name = 'Pclass', countsplit = 'Survived')
count_n_plot(df = titanic_train, col_name = 'Pclass', countsplit = 'Survived', bar = True)
count_n_plot(df = titanic_train, col_name = 'Sex', countsplit = 'Survived', bar = True)
count_n_plot(df = titanic_train, col_name = 'Sex', countsplit = 'Pclass', bar = True, barsplit = 'Pclass')
count_n_plot(df = titanic_train, col_name = 'Embarked', countsplit = 'Survived', bar = True)
count_n_plot(df = titanic_train, col_name = 'Embarked', countsplit = 'Sex', bar = True, barsplit = 'Sex')
count_n_plot(df = titanic_train, col_name = 'Embarked', countsplit = 'Pclass', bar = True, barsplit = 'Pclass')
count_n_plot(df = titanic_train, col_name = 'SibSp', countsplit = 'Survived', bar = True, barsplit = 'Sex')
count_n_plot(df = titanic_train, col_name = 'Parch', countsplit = 'Survived', bar = True, barsplit = 'Sex')
titanic_train.isnull().sum()*100 / len(titanic_train)
titanic_train[titanic_train['Age'].isnull()]
mean_age = round(titanic_train['Age'].mean(), 1)

print(mean_age)
titanic_train['Age'].replace(np.nan, mean_age, inplace=True)
# iloc means index location

titanic_train.iloc[[5,19,28,863,878],:]
mode_embarked = titanic_train['Embarked'].mode()[0] 

print(mode_embarked) 
print(type(titanic_train['Embarked'].mode()))

print(type(titanic_train['Embarked'].mode()[0]))
# look at missing values for Embarked column

titanic_train[titanic_train['Embarked'].isnull()]
# replace them with mode

titanic_train['Embarked'].replace(np.nan, mode_embarked, inplace=True)
# check if replaced correctly

titanic_train.iloc[[61,829],:]
titanic_train.drop(columns='Cabin', inplace=True)
# Check new dataset

titanic_train.head()
titanic_train.drop(columns=['PassengerId','Name','Ticket','Embarked'], inplace=True)
titanic_train.head()
titanic_train = pd.get_dummies(titanic_train, columns=['Sex'])
titanic_train.head()
# Drop Sex_female

titanic_train.drop('Sex_female', axis=1, inplace=True)
titanic_train.head()
titanic_train = pd.get_dummies(titanic_train, columns=['Pclass'], drop_first=True)
titanic_train.head()
# Read in the test dataset

titanic_test = pd.read_csv('../input/titanic/test.csv')
# Inspect the test dataset

titanic_test.head()
titanic_test.info()
# Check for null values

titanic_test.isnull().sum()
# Cleaning Age column

mean_age_test = titanic_test['Age'].mean()

print(mean_age_test)
titanic_test['Age'].replace(np.nan, mean_age_test, inplace=True)
# Cleaning Fare column

mean_fare_test = titanic_test['Fare'].mean()

print(mean_fare_test)
titanic_test['Fare'].replace(np.nan, mean_fare_test, inplace=True)
# Remove irrelevant columns but keep a copy of PassengerId column

eye_dee = titanic_test['PassengerId']

titanic_test.drop(columns=['PassengerId','Name','Ticket','Cabin','Embarked'], inplace=True)
titanic_test.head()
titanic_test.isnull().sum()
# One-hot encoding Pclass and Sex

titanic_test = pd.get_dummies(data=titanic_test, columns=['Pclass','Sex'], drop_first=True)
titanic_test.head()
# independent varibles, represented by a capital X; dependent variables represented by lowercase y

# [:,1:] means select all rows, and columns from 1st column onwards

X_train = titanic_train.iloc[:,1:]

y_train = titanic_train['Survived']

X_test = titanic_test
from sklearn.linear_model import LogisticRegression
# Specify the optimisation algorithm as 'lbfgs' (to silence the warning)

logistic_model = LogisticRegression(solver='lbfgs')
logistic_model.fit(X_train, y_train)
pred = logistic_model.predict(X_test)
pred
len(pred)
type(pred)
predictions = pd.Series(data=pred, name='Survived')
# The prediction values are inside a data series

type(predictions)
predictions.head()
sub = pd.concat([eye_dee, predictions], axis=1)
sub.head()
# Specify index=False so we don't get the index column when exporting to excel

submission1 = sub.to_csv('submission1.csv', index=False)