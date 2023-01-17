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
import numpy as np

import pandas as pd



# Modelling Helpers

from sklearn.preprocessing import Imputer , Normalizer , scale, LabelEncoder, OneHotEncoder

from sklearn.model_selection import train_test_split , StratifiedKFold

from sklearn.feature_selection import RFECV



# Data Visualisation

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns



#Modellling Algorithms



from sklearn.ensemble import RandomForestClassifier #Randomforest

from sklearn.tree import DecisionTreeClassifier #DecisionTrees

from xgboost import XGBClassifier

from keras.models import Sequential  #Sequential

from keras.layers import Dense

from keras.layers import Dropout
#Read the Training and Test Dataset

train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")



y_train = train['Survived']



X_train = train.drop(['Survived'], axis = 1)



X_test = test
y_train.head()



X_test.head()
# in my opinion the following columns hold no importance to the Survival of an individual. 

#Hence dropping them, may be later after few runs will bring them back.



X_train = X_train.drop(['PassengerId','Cabin', 'Name', 'Ticket'], axis = 1)



X_test = X_test.drop(['PassengerId','Cabin', 'Name', 'Ticket'], axis = 1)

X_test.head()
# The Age and Fare column has missing or NaN values which is replaced by the mean of the values.

X_train['Age'].fillna(X_train['Age'].mean(), inplace=True)



X_test['Age'].fillna(X_test['Age'].mean(), inplace=True)



X_test['Fare'].fillna(X_test['Fare'].mean(), inplace=True)



#dataframe.Column_Name.fillna(dataframe.Column_Name.mode()[0], inplace=True)



# Embarked also had missing values, but Embarked being Non-numeric, 

#we need to use mode which is replacing the missing values using the most commonly used value.

X_train['Embarked'].fillna(X_train['Embarked'].mode()[0], inplace=True)



X_test['Embarked'].fillna(X_test['Embarked'].mode()[0], inplace=True)

# We need to combine the Parch field and SibSp,which are short for Parent Child and Sibling Spouse.

#The Size of the family seems to have a relation with the Survival, smaller the better



family_train = pd.DataFrame()



family_train['FamilySize'] = X_train['Parch'] + X_train['SibSp'] + 1



family_train['Family_Single'] = family_train['FamilySize'].map( lambda s : 1 if s == 1 else 0 )

family_train['Family_Small']  = family_train['FamilySize'].map( lambda s : 1 if 2 <= s <= 4 else 0 )

family_train['Family_Large']  = family_train['FamilySize'].map( lambda s : 1 if 5 <= s else 0 )





family_test = pd.DataFrame()



family_test['FamilySize'] = X_test['Parch'] + X_test['SibSp'] + 1



family_test['Family_Single'] = family_test['FamilySize'].map( lambda s : 1 if s == 1 else 0 )

family_test['Family_Small']  = family_test['FamilySize'].map( lambda s : 1 if 2 <= s <= 4 else 0 )

family_test['Family_Large']  = family_test['FamilySize'].map( lambda s : 1 if 5 <= s else 0 )



family_train = family_train.drop(['FamilySize'], axis = 1)



X_train = X_train.drop(['SibSp', 'Parch'], axis = 1)



X_train = pd.concat([X_train, family_train], axis = 1)



family_test = family_test.drop(['FamilySize'], axis = 1)



X_test = X_test.drop(['SibSp', 'Parch'], axis = 1)



X_test = pd.concat([X_test, family_test], axis = 1)

X_test.head()

#Sex and Embarked has Categorical value. Sex has two categories whereas Embarked has three

#Following block of code is the Onehotencode the categorical data.

#Before doing that we need to make a copy of X_Train, to retain the sense of the actual columns

#Pclass also has categorical data, however, 

#the rank in this particular column is directly related to the survival chance



X_train_copy = X_train.copy()



labelencoder_X = LabelEncoder()

X_train[['Sex']] = labelencoder_X.fit_transform(X_train[['Sex']])

X_test[['Sex']] = labelencoder_X.fit_transform(X_test[['Sex']])



X_train[['Embarked']] = labelencoder_X.fit_transform(X_train[['Embarked']])

X_test[['Embarked']] = labelencoder_X.fit_transform(X_test[['Embarked']])



onehotencoder = OneHotEncoder(categorical_features = [X_train.columns.get_loc('Sex')])

onehotencoder = OneHotEncoder(categorical_features = [X_train.columns.get_loc('Embarked')])

X_train= onehotencoder.fit_transform(X_train).toarray()



onehotencoder = OneHotEncoder(categorical_features = [X_test.columns.get_loc('Sex')])

onehotencoder = OneHotEncoder(categorical_features = [X_test.columns.get_loc('Embarked')])

X_test = onehotencoder.fit_transform(X_test).toarray()

X_test
'''classifier = XGBClassifier()

classifier.fit(X_train, y_train)'''



'''classifier = RandomForestClassifier(n_estimators = 10, class_weight = 'balanced', random_state = 0)

classifier.fit(X_train, y_train)'''



classifier = RandomForestClassifier()
from sklearn.model_selection import cross_val_score

'''accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

accuracies.mean()

accuracies.std()'''



from sklearn.model_selection import GridSearchCV



#accuracies.mean()

parameters = {

                 'max_depth' : [4, 6, 8],

                 'n_estimators': [50, 10, 25],

                 'max_features': ['sqrt', 'auto', 'log2'],

                 'min_samples_split': [2, 3, 10],

                 'min_samples_leaf': [1, 3, 10],

                 'bootstrap': [True, False],

                 'criterion': ['entropy', 'gini']

                 }

grid_search = GridSearchCV(estimator = classifier,

                           param_grid = parameters,

                           scoring = 'accuracy',

                           cv = 10,

                           n_jobs = -1)

grid_search = grid_search.fit(X_train, y_train)

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_

best_accuracy

best_parameters
classifier = RandomForestClassifier(bootstrap = True, criterion = 'gini', max_depth = 8, max_features = 'auto', min_samples_leaf = 3, 

                                    min_samples_split = 10, n_estimators = 50)



classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_pred
passenger_id = test['PassengerId']

latest_Submission = pd.DataFrame( {'PassengerId': passenger_id , 'Survived': y_pred } )



#second_Submission = pd.DataFrame( {'PassengerId': passenger_id , 'Survived': y_pred } )
#second_Submission.to_csv('second_Submission.csv', index = False)



latest_Submission.to_csv('latest_Submission.csv', index = False)