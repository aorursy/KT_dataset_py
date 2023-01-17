# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import warnings

import re



# Any results you write to the current directory are saved as output.
# loading the data 

train = pd.read_csv('../input/train.csv', header = 0, dtype={'Age': np.float64})

test  = pd.read_csv('../input/test.csv' , header = 0, dtype={'Age': np.float64})

full_data = [train, test]

# Saving passengerIds for submission

passenger_ids = test['PassengerId']

print(train.info())
# Effect of class on target

print(train[['Pclass','Survived']].groupby(['Pclass'], as_index = False).mean())

# Effect of Sex on target 

print(train[['Sex','Survived']].groupby(['Sex'], as_index = False).mean())
# Creating a new feature FamilySize by using siblings spouse and parent children features 

for dataset in full_data:

    dataset['FamilySize'] =  dataset['SibSp'] + dataset['Parch'] + 1
# Effect of family size on our target 

print (train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())
for dataset in full_data:

    dataset['isAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1,'isAlone'] = 1

print(train[['isAlone','Survived']].groupby(['isAlone'], as_index = False).mean())
for dataset in full_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

print (train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())
for dataset in full_data:

    dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())

train['CategoricalFare'] = pd.qcut(train['Fare'],4)

print (train[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean())
# Age as a lot missing values so we will fill them with some random values

# b/w age_mean - age_std , age_mean + age_std



for dataset in full_data:

    age_avg 	   = dataset['Age'].mean()

    age_std 	   = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)

train['CategoricalAge'] = pd.qcut(train['Age'],5)

print(train[['CategoricalAge','Survived']].groupby(['CategoricalAge'],as_index = False).mean())

print(train.columns)
def get_title(name):

    title_search = re.search(' ([A-za-z]+)\.',name)

    if title_search:

        return title_search.group(1)

    else: 

        return ""

    

    

print(train.columns)
for dataset in full_data:

    dataset['Title'] = dataset['Name'].apply(get_title)



print(pd.crosstab(train['Title'], train['Sex']))

print(train.columns)
# replacing the rare titles like lady Ms Sir with Rare 

for dataset in full_data:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

print(train[['Title','Survived']].groupby(['Title'],as_index = False).mean())

print(train.columns)
# DataCleaning 

# Cleaning the data and mapping the feature in numerical nummbers

for dataset in full_data:

    # Mapping Sex

    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    

    # Mapping titles

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

    

    # Mapping Embarked

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

#   Mapping Fare

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

    

    # Mapping Age

    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']                           = 4



    

    

print(train.columns)


drop_elements = ['Name','Ticket','PassengerId','Cabin','SibSp','Parch', 'FamilySize']

train_x = train.drop(drop_elements, axis = 1,inplace = True)

train_x = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1, inplace = True)



test_x = test.drop(drop_elements, axis = 1, inplace = True)

test.head()


y = train['Survived']

train.drop(['Survived'],axis = 1,inplace = True)

train.head()
train_x = train.values

test_x = test.values
from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(train_x, y,train_size = 0.8, random_state = 0)

print(x_train)
# classifier comparison 

from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier

models = [

    ('KNeighborsClassifier',KNeighborsClassifier(4)),

    ('Support Vector machines ',SVC(probability = True)),

    ('XGBoost ',XGBClassifier(n_estimators = 1000, learning_rate = 0.05)),

    ('RandomForest ',RandomForestClassifier(n_estimators= 500, random_state = 0)),

    ('Gaussian Niave Bayes ', GaussianNB())

]

scores = []

for model in models:

    model = model[1]

    model.fit(x_train, y_train)

    preds = model.predict(x_valid)

    accuracy = accuracy_score(preds, y_valid)

    scores.append(accuracy)

print(scores)
model = models[1][1]



xg_clf = XGBClassifier(n_estimators=1000, learning_rate = 0.05)

xg_clf.fit(x_train, y_train)

pred = xg_clf.predict(x_valid)

print(accuracy_score(pred, y_valid))

test_preds = xg_clf.predict(test_x)

test_preds_df = pd.DataFrame({'PassengerId': passenger_ids,

                             'Survived':test_preds})

test_preds_df
test_preds_df.to_csv('submission.csv',index = False)