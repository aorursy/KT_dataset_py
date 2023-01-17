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
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test  = pd.read_csv('/kaggle/input/titanic/test.csv')

test_result = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
train.sample(10)
test.sample(10)
test_result.sample(10)
#Columns:

#PassengerId : unique ID number to each passenger

#Survived : Survived(1) or Died(0) passenger

#Pclass : Ticket class of each passenger; 1(upper class), 2(medium class), 3(lower class)

#Name : Passenger name

#Sex : Sex of passenger

#Age : Age of passenger

#SibSp : # of siblings

#Parch : # of parents / children

#Ticket : Ticker number

#Fare : Amount of money spent on ticket

#Cabin : Cabin number

#Embarked : Port of embarkation

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

def standardize_dataframe(df):

    sdf = df;

    if 'Survived' in df.columns:

        sdf = sdf.drop(['Survived'], axis=1)

    sdf = sdf.drop(['Name'] , axis=1).drop(['Ticket'] , axis=1).drop(['Cabin'] , axis=1).drop(['Embarked'] , axis=1)

    sdf['Sex'] = le.fit_transform(sdf['Sex'])

    return sdf.fillna(0)



data_train = standardize_dataframe(train)

data_test = standardize_dataframe(test)

x = data_train

print('test:\n' ,data_test)

print('x:\n' , x)

y = train['Survived'].values;

print('y:\n' , y)
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()

dtc.fit(x , y)

prediction = dtc.predict(data_test)

from sklearn.metrics import accuracy_score

print('Decision Tree Accuracy: ' , accuracy_score(prediction, test_result['Survived']))