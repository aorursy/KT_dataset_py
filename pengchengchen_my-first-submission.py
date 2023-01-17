# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing as pre

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# load the csv data

df = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

print(df.head())

print('=======================')

print(df_test.head())
# explore the dataset

print(df.info())

print('=========================================')

print(df_test.info())
# clean the data

# fill the missing value in training set

ageImp = pre.Imputer(strategy='median')

df.Age = ageImp.fit_transform(df.Age.reshape(-1, 1))



df.Embarked = df.Embarked.fillna('S')



print(df.info())
# encode the categorial value

sexLe = pre.LabelEncoder()

df.Sex = sexLe.fit_transform(df.Sex)



embarkedLe = pre.LabelEncoder()

df.Embarked = embarkedLe.fit_transform(df.Embarked)



print(df.Sex.head())

print('=================================')

print(df.Embarked.head())
# train machine-learning model

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

labels  = ['Survived']

sqrtfeat = np.sqrt(df.shape[1])

random_test = { "n_estimators"      : np.rint(np.linspace(df.shape[0]*2, df.shape[0]*4, 5)).astype(int),

                 "criterion"         : ["gini", "entropy"],

                 "max_features"      : np.rint(np.linspace(sqrtfeat/2, sqrtfeat*2, 5)).astype(int),

                 "min_samples_split" : np.rint(np.linspace(2, df.shape[0]/50, 10)).astype(int),

                 "min_samples_leaf"  : np.rint(np.linspace(1, df.shape[0]/200, 10)).astype(int), 

                 "max_leaf_nodes"    : np.rint(np.linspace(10, df.shape[0]/50, 10)).astype(int) }

clf = RandomForestClassifier()

clf = RandomizedSearchCV(clf, random_test, n_iter=50)

# print(df[labels].values.ravel())

clf = clf.fit(df[features], df[labels].values.ravel())
# predict using the trained model



# clean the test set

df_test.Age = ageImp.transform(df_test.Age.reshape(-1, 1))

df_test.Fare = df_test.Fare.fillna(df_test.Fare.median())

# print(df_test.info())



# encode the features

df_test.Sex = sexLe.transform(df_test.Sex)

df_test.Embarked = embarkedLe.transform(df_test.Embarked)

# print(df_test.Sex.head())

# print(df_test.Embarked.head())



# predict based on the test set

isSurvived = clf.predict(df_test[features])

# print(isSurvived)



# submission

submission = pd.DataFrame({'PassengerId': df_test.PassengerId, 

                          'Survived': isSurvived})

submission.to_csv('submission.csv', index=False)