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
# Importing all the required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set_style('whitegrid')
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
# importing the datasets into dataframes
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv("../input/test.csv")
train_df.head()
# remove the non relevant columns viz. Name, Ticket, PassengerId
train_df.drop(['PassengerId', 'Name', 'Ticket'], axis = 'columns', inplace = True)
train_df.info()
print('--------------------------------------------------------------------------------------')
test_df.info()
# Lets see the distribution of the Embarked variables for the purpose of imputing the 2 missing values in the Embarked variable
train_df.Embarked.value_counts(dropna = False)
train_df['Embarked'] = train_df['Embarked'].fillna('S')
train_df.Embarked.value_counts()
# Lets draw the factor plot as Factor plots make it easy to separate plots by categorical classes.
sns.factorplot(x = 'Embarked', y = 'Survived', data = train_df, size = 4, aspect = 3)
# Lets draw the distribution of the embarked classess now
sns.countplot(x='Embarked', data = train_df)
sns.countplot(x='Survived', hue = 'Embarked', data = train_df)
# group by embarked, and get the mean for survived passengers for each value in Embarked
embark_perc = train_df[['Survived', 'Embarked']].groupby(['Embarked']).mean()
embark_perc
embark_perc.plot(kind = 'bar')
# Create a new dataframe for embarked dummy variables, we will join later with train_df and test df
embark_dummies_train = pd.get_dummies(train_df['Embarked'], prefix = 'Embarked')
embark_dummies_test = pd.get_dummies(test_df['Embarked'], prefix = 'Embarked')

# drop one of the variables, lets drop the Embarked_C
embark_dummies_train.drop('Embarked_C', axis = 'columns', inplace = True)
embark_dummies_test.drop('Embarked_C', axis = 'columns', inplace = True)

embark_dummies_test.head()
train_df = train_df.join(embark_dummies_train)
test_df = test_df.join(embark_dummies_test)

train_df.head()
# Drop the original Embarked variable from the train_df and test_df
train_df.drop('Embarked', axis = 'columns', inplace = True)
test_df.drop('Embarked', axis = 'columns', inplace = True)
test_df["Fare"].fillna(test_df["Fare"].median(), inplace = True)
#convert fare from flat to int
train_df['Fare'] = train_df.Fare.astype(int)
test_df['Fare'] = test_df.Fare.astype(int)
train_df.Fare.plot(kind = 'hist', bins = 100)
g = sns.distplot(train_df["Fare"], color = "b", label ="Skewness: %.2f")
train_df.Fare = train_df.Fare.map(lambda i: np.log(i) if i > 0 else 0)
g = sns.distplot(train_df.Fare, color = 'm', label = "Skewness: %.2f")
test_df.Fare = test_df.Fare.map(lambda i: np.log(i) if i > 0 else 0 )
train_df.Sex = train_df.Sex.map({'Male': 0, 'Female': 1})
test_df.Sex = test_df.Sex.map({'Male': 0, 'Female':1})
train_df['Age'].isnull().sum()
X_train = train_df.drop('Survived', axis = 'columns')
