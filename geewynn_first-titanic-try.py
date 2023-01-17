# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas_profiling as pp

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
combined = pd.concat([train, test], ignore_index=True)

combined.head()
target = train['Survived']

print('Sum of missing values')

print(combined.isnull().sum())

print('-------------------------------')



print('Percentage of missing values')

print(combined.isnull().mean())
# drop this column becaues they are not useful and cabin features has a lot of missing values

combined = combined.drop(['Survived', 'Cabin', 'Ticket', 'PassengerId'], axis=1)
combined['Age'].describe()
#Age cleaning



# replace null values with mean of age

combined['Age'] = combined['Age'].replace(np.NaN, np.mean(combined['Age']))



# create a bin for the age group

bins = (-1, 0, 5, 12, 18, 40, 60, 120)

agegroup = ['Unknown', 'Baby', 'Child', 'Teenager', 'Youth', 'Adult', 'Elder']



# create new feature agegroup

combined['Agegroup'] = pd.cut(combined['Age'], bins, labels=agegroup)



# drop age column

combined = combined.drop(combined[['Age']], axis=1)



combined.head()
# Handling the fare column

#fill missing value

combined['Fare'] = combined['Fare'].replace(np.NaN, np.mean(combined['Fare']))
# Handling Names

#create new feature by extracting the titles from the names idea gotten from

#https://www.kaggle.com/startupsci/titanic-data-science-solutions



combined['Title'] = combined.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



# After extraction group titles

combined['Title'] = combined['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don',

                                                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 

                                                'Dona'], 'Rare')

combined['Title'] = combined['Title'].replace('Mlle', 'Miss')

combined['Title'] = combined['Title'].replace('Ms', 'Miss')

combined['Title'] = combined['Title'].replace('Mme', 'Mrs')



#drop name column

combined = combined.drop(combined[['Name']], axis=1)
combined['FamilySize'] = combined['SibSp'] + combined['Parch'] + 1
# embarked column

#fill missing value with the most frequent 

combined['Embarked'] = combined['Embarked'].fillna('S')
combined = combined.drop(combined[['Parch', 'SibSp']], axis=1)
combined.head()
# get the nominal and ordinal columns that needs to be label encoded or one hot encoded



nominal = combined[['Embarked', 'Sex', 'Title']]

ordinal = combined[['Agegroup']]


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

# One hot encode the norminal columns



for col in nominal:

    dummy = pd.get_dummies(nominal, drop_first=True)

    combineddf = pd.concat([combined, dummy], axis=1)

    

    

#label encode the ordinal features

combineddf['Agegroup'] = le.fit_transform(combineddf['Agegroup'])

combineddf['Title'] = le.fit_transform(combineddf['Title'])

    

combineddf = combineddf.drop(nominal, axis=1)
combineddf.head()
#Split the data into train and test



traindf = combineddf.loc[:890,:]

testdf = combineddf.loc[891:, :]
print(traindf.shape)

print(testdf.shape)
#use train_test_split to creata train and validation data



X_train, X_test, y_train, y_test = train_test_split(traindf, target, test_size= 0.2, random_state=42)
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators=100)

rf.fit(X_train, y_train)
pred = rf.predict(X_test)
from sklearn.metrics import classification_report



print(classification_report(y_test, pred))
rf1_pred = rf.predict(testdf)

sub = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived':rf1_pred})



sub = sub.to_csv('rf1.csv', index=False)
clf = RandomForestClassifier()



parameters = {'n_estimators': [100, 200, 300],

              'max_features': ['sqrt', 'auto', 'log2', None],

              'criterion': ['entropy', 'gini'],

              'bootstrap': [True, False],

              'max_depth': [2, 4,5, 6, 7],

              'min_samples_leaf': [2, 3, 4,5,6,7]}





cv = GridSearchCV(clf, param_grid=parameters, cv=5, n_jobs=-1)



cv.fit(X_train, y_train)
best_params = cv.best_params_
rfr = RandomForestClassifier(max_depth=best_params["max_depth"], 

                            n_estimators=best_params["n_estimators"],

                            bootstrap=best_params["bootstrap"],

                            min_samples_leaf=best_params["min_samples_leaf"],

                            max_features=best_params['max_features'],

                            criterion = best_params['criterion'],

                        )



rfr.fit(X_train, y_train)
rfr_pred = rfr.predict(X_test)

print(classification_report(y_test, rfr_pred))
sub = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived':rf1_pred})



sub = sub.to_csv('rf.csv', index=False)
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)

X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel())



rfs = RandomForestClassifier()

rfs.fit(X_train, y_train)



pred = rfs.predict(X_test)



print(classification_report(y_test, pred))
rfs_pred = rfs.predict(testdf)

sub = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived':rfs_pred})



sub = sub.to_csv('rfs.csv', index=False)