# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # graphics

import seaborn # pretty graphics



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Read in data to pandas dataframes

training = pd.read_csv("/kaggle/input/titanic/train.csv")

testing = pd.read_csv("/kaggle/input/titanic/test.csv")
# Take a look at the data

training.head(10)
# Look at what is missing

training.info()

testing.info()
training['Age'].plot(kind='hist',title='Age')
training['Embarked'].value_counts().plot(kind='bar', title='Embarked')
from sklearn.preprocessing import OneHotEncoder



# OneHot encode the "Sex" property

sex = training["Sex"]

data_sex, sex_categories = sex.factorize()

encoder = OneHotEncoder(categories='auto')

sex_OH = pd.DataFrame(encoder.fit_transform(data_sex.reshape(-1, 1)).toarray(), columns=sex_categories)

training = pd.concat([training, sex_OH], axis=1)



sex = testing["Sex"]

data_sex, sex_categories = sex.factorize()

encoder = OneHotEncoder(categories='auto')

sex_OH = pd.DataFrame(encoder.fit_transform(data_sex.reshape(-1, 1)).toarray(), columns=sex_categories)

testing = pd.concat([testing, sex_OH], axis=1)



# Combine the SibSp and Parch to a single column

training['Family'] = training['SibSp'] + training['Parch']

testing['Family'] = testing['SibSp'] + testing['Parch']



# Fill the missing values of Embarked with 'S' as that appears to have

# by far the most occurences

training['Embarked'] = training['Embarked'].fillna('S')

testing['Embarked'] = testing['Embarked'].fillna('S')



# OneHot encode where the passengers embarked

embarked = training["Embarked"]

data_embarked, embarked_categories = embarked.factorize()

embarked_OH = pd.DataFrame(encoder.fit_transform(data_embarked.reshape(-1,1)).toarray(), columns=embarked_categories)

training = pd.concat([training, embarked_OH], axis=1)



embarked = testing["Embarked"]

data_embarked, embarked_categories = embarked.factorize()

embarked_OH = pd.DataFrame(encoder.fit_transform(data_embarked.reshape(-1,1)).toarray(), columns=embarked_categories)

testing = pd.concat([testing, embarked_OH], axis=1)



# Fill in the missing fare with the median of the passenger class

training['Fare'] = training.groupby('Pclass')['Fare'].transform(lambda x: x.fillna(x.median()))

testing['Fare'] = testing.groupby('Pclass')['Fare'].transform(lambda x: x.fillna(x.median()))



# Drop unnecessary columns

training = training.drop(['Sex', 'SibSp', 'Parch', 'Embarked', 'Cabin', 'Ticket'], axis=1)

testing = testing.drop(['Sex', 'SibSp', 'Parch', 'Embarked', 'Cabin', 'Ticket'], axis=1)
# Parse the passenger's title from 'Name'

import re



def get_title(name):

    """ Use regex to get the title form the passenger's name"""

    title = re.search('(\w+)\.', name)

    return title.group(1)

    

training['Title'] = training['Name'].apply(lambda x: get_title(x))

testing['Title'] = testing['Name'].apply(lambda x: get_title(x))



training['Title'].value_counts().plot(kind='bar')
# Combine to Miss

training['Title'] = training['Title'].replace(['Lady', 'Ms', 'Mlle'], 'Miss')

testing['Title'] = testing['Title'].replace(['Lady', 'Ms', 'Mlle'], 'Miss')



# Combine to Mrs

training['Title'] = training['Title'].replace('Mme', 'Mrs')

testing['Title'] = testing['Title'].replace('Mme', 'Mrs')



# Combine to Mr

training['Title'] = training['Title'].replace('Sir', 'Mr')

testing['Title'] = testing['Title'].replace('Sir', 'Mr')



# Combine to Misc

training['Title'] = training['Title'].replace(['Capt', 'Col', 'Countess', 'Don', 'Dona', 

                                               'Dr', 'Jonkheer', 'Major', 'Rev'], 'Misc')

testing['Title'] = testing['Title'].replace(['Capt', 'Col', 'Countess', 'Don', 'Dona', 

                                               'Dr', 'Jonkheer', 'Major', 'Rev'], 'Misc')



# Fill missing age values with the median of the Title group

training['Age'] = training.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))

testing['Age'] = testing.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))
training.info()

testing.info()
training.hist(column='Age', by=['Pclass','Survived'], grid=True, 

              sharex=True, sharey=True, figsize=(12,10),

              bins=range(0,90,10))
features = ['Pclass', 'male', 'female', 'Age', 'Fare', 'S', 'Q', 'C', 'Family']

X = training[features]

y = training['Survived']
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



# Make the x for train & validation

x_train,x_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=1234)
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestClassifier



model = RandomForestClassifier(n_estimators=5, max_depth=10,random_state=1234)

model.fit(X, y)

model.score(X, y)
X = testing[features]

#y = testing['Survived']

testing['Survived'] = np.round(model.predict(X),0)

testing.to_csv('model_prediction.csv', columns=['PassengerId','Survived'], index=False)