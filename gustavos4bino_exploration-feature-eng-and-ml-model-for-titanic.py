# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



%matplotlib inline
data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")



data.head()
# I want to use Age to create another column, but Age has 177 missing values. 

# 177 people are 19% of the data. Drop this observations is a loss of significant data



data.isna().sum()
data.Age.describe()
data.Age.dropna().plot(kind='hist')
# using the age column is a viable option due to its value 

data.Age.fillna(data.Age.mean(), inplace=True)
sns.heatmap(data.drop(["PassengerId", 'Name', 'Ticket', 'Cabin',], 1).corr().abs(), annot=True)
# Just because this is a notebook, doesn't mean you have to forget to use functions.

# Dont write the same thing over and over again, create a function

def calculate_mean_survavibility(data, feature):

    """Method for calculate the mean survavibility for a given feature.

    """

    use_this_columns = [feature,'Survived']

    return data[use_this_columns].groupby(feature, as_index=False).mean()
# Bining Age data



data['age_c'] = pd.cut(x=data["Age"], bins=5, 

       labels=['child','young','young-adult', 'adult' ,'old']) # Bining data



# must do the same for test data



test_data['age_c'] = pd.cut(x=test_data["Age"], bins=5, 

       labels=['child','young','young-adult', 'adult' ,'old']) 



calculate_mean_survavibility(data, 'age_c')
sns.swarmplot(x='Pclass', y='Age', hue='Survived', data=data)
sns.swarmplot(x='age_c', y='Age', hue='Survived', data=data)
calculate_mean_survavibility(data, 'Sex')
sns.swarmplot(x='Sex', y='Age', hue='Survived', data=data)
calculate_mean_survavibility(data, 'Parch')
sns.swarmplot(x='Parch', y='Age', hue='Survived', data=data)
calculate_mean_survavibility(data, 'SibSp')
sns.swarmplot(x='SibSp', y='Age', hue='Survived', data=data)
data['total_family'] = data['SibSp'] + data['Parch']

test_data['total_family'] = test_data['SibSp'] + test_data['Parch']
calculate_mean_survavibility(data, 'Pclass') # class 1 survived more than others
calculate_mean_survavibility(data, 'Embarked') # The people who embarked at port C were the ones who survived the most. Could this be a missleading correlation?
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.metrics import classification_report
data.head(2)
keep_this_columns = ['Pclass', 'Sex','age_c','total_family'] # Features that we found relevant.



X = data[keep_this_columns]

X = pd.get_dummies(X, columns=['age_c', 'Sex', 'Pclass'])

y = data['Survived']



X_train, X_test, y_train, y_test = train_test_split(X, y)
def fit_predict(model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)

    print(classification_report(y_test, model.predict(X_test)))

    print("Score: ", model.score(X_train, y_train))
model = LogisticRegression()

fit_predict(model, X_train, X_test, y_train, y_test)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

fit_predict(model, X_train, X_test, y_train, y_test)
# Test data prep



test_x = test_data[keep_this_columns]

test_x = pd.get_dummies(test_x, columns=['age_c', 'Sex', 'Pclass'])

# y_test is the prediction of my





predictions = model.predict(test_x)
pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived':predictions}).to_csv('titanic.csv', index=False)