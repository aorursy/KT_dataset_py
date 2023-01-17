import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('bmh')

%matplotlib inline

from sklearn import preprocessing



from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression



from sklearn.model_selection import KFold
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

test_id = test['PassengerId']

combine = [train, test] # combine train and test data, easy to do data manipulation



for df in combine: # add feature 'FamilySize'                                  

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1



for df in combine: # add feature 'Alone' 

    df['Alone'] = 0

    df.loc[df['FamilySize'] == 1, 'Alone'] = 1



for df in combine: # fill missing values for 'Embarked'

    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    

for df in combine: # fill missing values for 'Fare' and transform into categorical feature

    df['Fare'].fillna(df['Fare'].median(), inplace=True)

    df.loc[df['Fare'] <= 10.5, 'Fare'] = 0

    df.loc[(df['Fare'] > 10.5) & (df['Fare'] <= 21.679), 'Fare'] = 1

    df.loc[(df['Fare'] > 21.679) & (df['Fare'] <= 39.688), 'Fare'] = 2

    df.loc[(df['Fare'] > 39.688) & (df['Fare'] <= 512.329), 'Fare'] = 3

    df.loc[df['Fare'] > 512.329, 'Fare'] = 4

    

    

for df in combine: # fill missing values for 'Age' and transform into categorical feature

    avg = df['Age'].mean()

    std = df['Age'].std()

    NaN_count = df['Age'].isnull().sum()

    

    age_fill = np.random.randint(avg-std, avg+std, NaN_count)

    df.loc[df['Age'].isnull(), 'Age'] = age_fill

    df['Age'] = df['Age'].astype(int)

    

    df.loc[df['Age'] <= 16, 'Age'] = 0

    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1

    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2

    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3

    df.loc[df['Age'] > 64, 'Age'] = 4

    

#for df in combine:

#    df['Age*Pclass'] = df['Age'] * df['Pclass']

    

#for df in combine:

#    df['Age*Fare'] = df['Age'] * df['Fare']

    

import re

def only_title(name): # manipulation 'Name', extracting titles from names

    title = re.findall(' ([A-Za-z]+)\.', name)

    if title:

        return title[0]

    

for df in combine:

    df['Title'] = df['Name'].apply(only_title) 

    

for df in combine:

    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 

                                     'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    df['Title'] = df['Title'].replace('Mlle', 'Miss')

    df['Title'] = df['Title'].replace('Ms', 'Miss')

    df['Title'] = df['Title'].replace('Mme', 'Mrs')



############ Encoding features, make them ready for classifiers

feature_drop = ['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'FamilySize']

for df in combine:

    df.drop(feature_drop, axis=1, inplace=True)



def encode_features(train, test):

    features = ['Sex', 'Embarked', 'Fare', 'Age', 'Title']

    df_combined = pd.concat([train[features], test[features]])

    

    for feature in features:

        le = preprocessing.LabelEncoder()

        le = le.fit(df_combined[feature])

        train[feature] = le.transform(train[feature])

        test[feature] = le.transform(test[feature])

    return train, test

    

train, test = encode_features(train, test)

train.head()
# Create Numpy arrays of train, test and target (Survived) dataframes to feed into our models

y_train = train['Survived'].ravel()

train = train.drop(['Survived'], axis=1)

X_train = train.values # Creates an array of the train data

X_test = test.values # Creats an array of the test data



print(X_train.shape)

print(y_train.shape)

print(X_test.shape)
# Class to extend the Sklearn classifier

class SklearnHelper(object):

    def __init__(self, clf, seed=2017, params=None):

        params['random_state'] = seed

        self.clf = clf(**params)



    def train(self, X_train, y_train):

        self.clf.fit(X_train, y_train)



    def predict(self, x):

        return self.clf.predict(x)

    

    def fit(self, x, y):

        return self.clf.fit(x,y)

    

    def feature_importances(self, x, y):

        print(self.clf.fit(x, y).feature_importances_)
gb_clf = GradientBoostingClassifier()

gb_clf.fit(X_train, y_train)

pred = gb_clf.predict(X_test)
output = pd.DataFrame({'PassengerId' : test_id, 'Survived': pred})



output.to_csv('Predictions.csv', index = False)

output.head()