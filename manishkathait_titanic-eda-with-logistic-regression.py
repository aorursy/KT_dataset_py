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
#importing necessary packages#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn import linear_model as lm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#reading teh train and test files and saving them as pandas dataframe#
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
#dimensions of the input data#
print("Train dataframe shape is: ", train_df.shape)
print("Test dataframe shape is: ", test_df.shape)
#name of the columns#
train_df.columns
test_df.columns
#checking the first 10 rows#
train_df.head(10)
#getting the summary statistics of the numerical columns#
train_df.describe()
#getting the data types of all columns#
train_df.dtypes
#more information about the dataset#
train_df.info()
test_df.info()
#dropping the Cabin variable
train_df.drop(['Cabin'], axis = 1, inplace = True)
test_df.drop(['Cabin'], axis = 1, inplace = True)
sns.countplot(x = train_df['Survived'])
sns.countplot(x = 'Survived', hue = 'Sex', data = train_df)
sns.countplot(x = 'Survived', hue = 'Pclass', data = train_df)
sns.distplot(train_df['Age'].dropna(), kde = False, bins = 30)
sns.countplot(x = 'SibSp', data = train_df)
train_df['Fare'].hist()
train_df['Fare'].hist(bins = 40, figsize = (10,4))
plt.figure(figsize = (10,7))
sns.boxplot( x = 'Pclass', y = 'Age', data = train_df)
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
        
    else:
        return Age
train_df['Age'] = train_df[['Age', 'Pclass']].apply(impute_age, axis = 1)
#lets check the missing vaues now
sns.heatmap(train_df.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')
#Converting Categorical variables to Indicator variables
sex = pd.get_dummies(train_df['Sex'], drop_first = True)
embark = pd.get_dummies(train_df['Embarked'], drop_first = True)
# adding the 2 new columns to the dataframe
train_df = pd.concat([train_df, sex, embark], axis = 1)
train_df.head()
#dropiing the columns not needed anymore
train_df.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis = 1, inplace = True)
train_df.head(5)
train_df.drop(['PassengerId'], axis = 1, inplace = True)
train_df.head(5)
#Considering train_df as our entire data and splitting it into training & testing dataset
X = train_df.drop('Survived', axis = 1)
y = train_df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 101)
logmodel =  lm.LogisticRegression()
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)
print(classification_report(y_test, predictions))
confusion_matrix(y_test, predictions)