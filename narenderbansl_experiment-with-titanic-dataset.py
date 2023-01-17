# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



## Any results you write to the current directory are saved as output.
#Importing neccessary libraries



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.ensemble import VotingClassifier

from xgboost import XGBClassifier

from sklearn.model_selection import cross_val_score,KFold,train_test_split

import warnings

warnings.filterwarnings("ignore")
test  = pd.read_csv('../input/test.csv')

train = pd.read_csv('../input/train.csv')



# for checking the null values in each columns

train.info()

df= list(train.select_dtypes(exclude=['object']).columns)

print(df)

train[df].boxplot(return_type='axes')







3

4

5

6

7

8

# Get unique values in a numpy array

arr = np.array([11, 11, 12, 13, 14, 15, 16, 17, 12, 13, 11, 14, 18])

print('Original Numpy Array : ' , arr)

 

# Get unique values from a numpy array

uniqueValues = dict(zip(*np.unique(arr,return_counts=True)))

 

print('Unique Values : ',uniqueValues)
# for checking the null values in each columns

test.info()

max_Age=train['Age'].max()

min_Age=train['Age'].min()

print(max_Age)

print(min_Age)
# from above found age and fare having null values so filling will suitable values.

# Compute the average age per class using both training and testing datasets. 

Age_Pclass1 = .60*(train[train['Pclass']==1]['Age'].mean() + test[test['Pclass']==1]['Age'].mean())

Age_Pclass2 = .60*(train[train['Pclass']==2]['Age'].mean() + test[test['Pclass']==2]['Age'].mean())

Age_Pclass3 = .60*(train[train['Pclass']==3]['Age'].mean() + test[test['Pclass']==3]['Age'].mean())



print(Age_Pclass1, Age_Pclass2, Age_Pclass3)
max_Fare=train['Fare'].max()

min_Fare=train['Fare'].min()

print(max_Fare)

print(min_Fare)
# Compute the average fare per class using both training and testing datasets. 

Fare_Pclass1 = 0.5*(train[train['Pclass']==1]['Fare'].mean() + test[test['Pclass']==1]['Fare'].mean())

Fare_Pclass2 = 0.5*(train[train['Pclass']==2]['Fare'].mean() + test[test['Pclass']==2]['Fare'].mean())

Fare_Pclass3 = 0.5*(train[train['Pclass']==3]['Fare'].mean() + test[test['Pclass']==3]['Fare'].mean())



print(Fare_Pclass1, Fare_Pclass2, Fare_Pclass3)

def Calculate_age(cols):

    

    Pclass = cols[0]

    Age = cols[1]



    if pd.isnull(Age):

        if Pclass==1:

            return Age_Pclass1      

        if Pclass==2:

            return Age_Pclass2

        else:

            return Age_Pclass3

    else:

        return Age
def calculate_fare(cols):

    

    Pclass = cols[0]

    Fare = cols[1]



    if pd.isnull(Fare):

        if Pclass==1:

            return Fare_Pclass1      

        if Pclass==2:

            return Fare_Pclass2

        else:

            return Fare_Pclass3

    else:

        return Fare
# Replace missing values

train['Age'] = train[['Pclass','Age']].apply(Calculate_age, axis=1)

test['Age'] = test[['Pclass','Age']].apply(Calculate_age, axis=1)



train['Fare'] = train[['Pclass','Fare']].apply(calculate_fare, axis=1)

test['Fare'] = test[['Pclass','Fare']].apply(calculate_fare, axis=1)
train.info()
test.info()
train.drop(['Ticket','Name'], inplace=True, axis=1)

test.drop(['Ticket','Name'], inplace=True, axis=1)
train.drop(['Cabin'], inplace=True, axis=1)

test.drop(['Cabin'], inplace=True, axis=1)
train.info()

train.head(5)
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
test['Embarked'].fillna(test['Embarked'].mode()[0], inplace=True)
train['Sex'] = train['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
test['Sex'] = test['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
train.head(5)

set(train['Embarked'])
train['Embarked'] = train['Embarked'].map( {'C': 1, 'Q': 0,'S': 2} ).astype(int)
test['Embarked'] = test['Embarked'].map( {'C': 1, 'Q': 0,'S': 2} ).astype(int)
train.head(5)
train.info()
test.info()
tests=test

train.drop(['PassengerId'], inplace=True, axis=1)

test.drop(['PassengerId'], inplace=True, axis=1)
X = train.drop(['Survived'], axis=1)

y = train['Survived']

#spliting data

x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1)



# prepare models different models for opting best models

models = []

models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('XGBoosting',XGBClassifier(learning_rate =0.1,

 n_estimators=100,

 max_depth=10,

 min_child_weight=11,

 gamma=0.01,

 subsample=0.8)))



models.append(('Soft', VotingClassifier(estimators=[('LR', LogisticRegression()),('LDA', LinearDiscriminantAnalysis()),('KNN', KNeighborsClassifier()),('CART', DecisionTreeClassifier()),('NB', GaussianNB())], voting='soft')))

models.append(('Hard', VotingClassifier(estimators=[('LR', LogisticRegression()),('LDA', LinearDiscriminantAnalysis()),('KNN', KNeighborsClassifier()),('CART', DecisionTreeClassifier()),('NB', GaussianNB())], voting='hard')))

# evaluate each model in turn

results = []

names = []

scoring = 'accuracy'

for name,model in models:

    model.fit(x_train,y_train)

    kfold = KFold(n_splits=10, random_state=5)

    cv_results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f, (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)

# as we can clealy see the accuracy of XGBoosting is greater among all the algos so will apply this algo





XGBmodel = XGBClassifier(learning_rate =0.1,

 n_estimators=100,

 max_depth=10,

 min_child_weight=11,

 gamma=0.01,

 subsample=0.8)



XGBmodel.fit(x_train,y_train)

predictions = XGBmodel.predict(test)



score = XGBmodel.score(X,y)

print("XGBmodel train_score: {}".format(score))
submit = pd.read_csv('../input/gender_submission.csv')

submit.head(5)





submit.Survived = predictions



submit.to_csv('gender_submission.csv', index = False)



submit = pd.read_csv('../input/gender_submission.csv')

submit.head(5)





## end here ---------------------------------------------