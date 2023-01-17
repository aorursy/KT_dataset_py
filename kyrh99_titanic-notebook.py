# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os





print("setup completed")



# Any results you write to the current directory are saved as output.
paths = ["../input/titanic/train.csv","../input/titanic/test.csv"]

traindf = pd.read_csv(paths[0])

testdf = pd.read_csv(paths[1])
traindf = traindf.drop(['Ticket', 'Cabin','PassengerId'], axis=1)

testdf = testdf.drop(['Ticket', 'Cabin','PassengerId'], axis=1)

print("testdf.columns:")

print(testdf.columns)

print("\ntraindf.columns:")

print(traindf.columns)
nonumbercol = traindf.select_dtypes(exclude=['int','float']).columns

print(nonumbercol)
traindf['Title'] = traindf.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

testdf['Title'] = testdf.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



print(traindf.Title.unique())

print(testdf.Title.unique())



titles = np.union1d(traindf.Title.unique(), testdf.Title.unique())

titles = titles.tolist()

print(titles)

print(len(titles))



def t(row):

    row.Title = titles.index(row.Title) + 1

    return row



traindf1 = traindf.apply(t, axis='columns')

testdf1 = testdf.apply(t, axis='columns')

testdf1.Title.unique()
traindf1 = traindf1.drop(['Name'], axis=1)

testdf1 = testdf1.drop(['Name'], axis=1)

print(testdf1.columns)
testdf1['Sex'] = testdf1['Sex'].map({'female': 1, 'male': 0}).astype(int)

traindf1['Sex'] = traindf1['Sex'].map({'female': 1, 'male': 0}).astype(int)
nonumbercol = traindf1.select_dtypes(exclude=['int','float']).columns

print(nonumbercol)

traindf.info()
testmostf = testdf1.Embarked.dropna().mode()[0]

trainmostf = traindf1.Embarked.dropna().mode()[0]

testdf1['Embarked'] = testdf1['Embarked'].fillna(testmostf) 

traindf1['Embarked'] = traindf1['Embarked'].fillna(trainmostf)

testdf1['Embarked'] = testdf1['Embarked'].map({'C': 1, 'S': 2, 'Q': 3}).astype(int)

traindf1['Embarked'] = traindf1['Embarked'].map({'C': 1, 'S': 2, 'Q': 3}).astype(int)
testdf1.info()
testdf1['Fare'].fillna(testdf1['Fare'].dropna().median(), inplace=True)
testdf1.info()
import math



guess_ages = np.zeros((18,2,3))



for dataset in [testdf1,traindf1]:

    for i1 in range(0,18):

        for i in range(0, 2):

            for j in range(0, 3):

                guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1) & (dataset['Title'] == i1+1)]['Age'].dropna()

                age_guess = guess_df.median()

                if math.isnan(age_guess):

                    guess_ages[i1,i,j] = 0

                else:

                    guess_ages[i1,i,j] = int(age_guess/0.5 + 0.5) * 0.5

            

        for i1 in range(0,18):

            for i in range(0, 2):

                for j in range(0, 3):

                    dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1) & (dataset['Title'] == i1+1),'Age'] = guess_ages[i1,i,j]



    dataset['Age'] = dataset['Age'].astype(int)
testdf1.info()
traindf1.info()
def histogram_intersection(a, b):

    v = np.minimum(a, b).sum().round(decimals=1)

    return v



corr_mat = traindf1.corr(method='spearman')

np.fill_diagonal(corr_mat.values, 0)



corr_mat["max"] = corr_mat.apply(max, axis=1)

print(corr_mat)
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split







X = traindf1.drop(['Survived'], axis = 1)

y = traindf1['Survived']



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)



#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

y.unique()
# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier



'''

random_forest = RandomForestClassifier(n_estimators=200)

random_forest.fit(X_train, y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, y_train)

acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)

print(acc_random_forest)

acc_random_forest = round(random_forest.score(X_val, y_val) * 100, 2)

print(acc_random_forest)

acc_random_forest = round(random_forest.score(X_test, y_test) * 100, 2)

print(acc_random_forest)'''

from sklearn.metrics import accuracy_score



XGBR_model = XGBClassifier(n_estimators = 1000, learning_rate = 0.01, n_jobs = 4, objective='binary:logistic')



XGBR_model.fit(X_train, y_train, eval_metric = 'rmse', verbose = False)



predVal = XGBR_model.predict(X_val)

#predTest = XGBR_model.predict(X_test)

predTrain = XGBR_model.predict(X_train)



print("train score:")

print(accuracy_score(y_train, predTrain))

print("\nvalidation score:")

print(accuracy_score(y_val, predVal))

#print("\ntest score:")

#print(accuracy_score(y_test, predTest))



#n_estimators = 1000, learning_rate = 0.01 with early stoppping rounds

#0.8603351955307262



#n_estimators = 1000, learning_rate = 0.01 without early stoppping rounds

#0.8659217
X = traindf1.drop(['Survived'], axis = 1)

y = traindf1['Survived']
print(traindf1.head())
#submission = pd.DataFrame({

#        "PassengerId": testdf["PassengerId"],

#        "Survived": Y_pred

#    })

# submission.to_csv('../output/submission.csv', index=False)