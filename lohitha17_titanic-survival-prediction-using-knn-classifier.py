# Importing related Python libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import csv

import warnings

warnings.filterwarnings("ignore")
# Importing the training dataset

df = pd.read_csv('../input/titanic/train.csv')
df.head()
df.shape
df.dtypes
df = df.drop('Name', axis=1,)

df = df.drop('Ticket', axis=1,)

df = df.drop('Fare', axis=1,)

df = df.drop('Cabin', axis=1,)
df['Family'] = df['SibSp'] + df['Parch'] + 1
df = df.drop('SibSp', axis=1,)

df = df.drop('Parch', axis=1,)
df.describe()
# By describing data, we found out there few NAN's in Age

# so replacing Age with median of the column

df["Age"] = df["Age"].fillna(df["Age"].median())
df.describe()
df['Embarked'].value_counts()
#finding NAN's in Embarked column

df['Embarked'].isna().sum()
#Replacing the NAN's with most frequently used one i.e mode(metric that gives us most frequently used value)

print(df["Embarked"].mode())

df["Embarked"] = df["Embarked"].fillna("S")
df['Embarked'].describe()
# Replacing the categorical value Embarked into numerical value

df["Embarked"].unique()
df.Embarked.replace(['S', 'C', 'Q'], [1, 2, 3], inplace=True)
df.Sex.replace(['male', 'female'], [1,0], inplace=True)
df.head()
# importing libraries

from sklearn.model_selection import train_test_split

from sklearn.neighbors import NearestNeighbors

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_recall_fscore_support

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from collections import Counter

from sklearn.model_selection import cross_validate
X = np.array(df.filter(['Pclass','Sex','Embarked','Family','Age'], axis=1))
y = np.array(df.filter(['Survived'], axis=1))
# simple cross validation

X_1, X_test, y_1, y_test = train_test_split(X,y, test_size=0.3)

X_tr, X_cv, y_tr, y_cv = train_test_split(X_1, y_1, test_size=0.3)
final_scores = []

for i in range(1,30,2):

    knn = KNeighborsClassifier(n_neighbors = i)

    knn.fit(X_tr, y_tr)

    pred = knn.predict(X_cv)

    acc = accuracy_score(y_cv, pred, normalize=True) * float(100)

    final_scores.append(acc)

    print('\n CV accuracy for k=%d is %d'%(i,acc))
optimal_k = final_scores.index(max(final_scores))

print(optimal_k)
# getting accuracy if K=5 on the test data

df_test = pd.read_csv('../input/titanic/test.csv')

df_test = df_test.drop('Name', axis=1,)

df_test = df_test.drop('Ticket', axis=1,)

df_test = df_test.drop('Fare', axis=1,)

df_test = df_test.drop('Cabin', axis=1,)

df_test['Family'] = df_test['SibSp'] + df_test['Parch'] + 1

df_test = df_test.drop('SibSp', axis=1,)

df_test = df_test.drop('Parch', axis=1,)

df_test["Age"] = df_test["Age"].fillna(df["Age"].median())
df_test1 = pd.read_csv('../input/titanic/test.csv')

df_test1
print(df_test["Embarked"].mode())

df_test["Embarked"] = df_test["Embarked"].fillna("S")
df_test.Embarked.replace(['S', 'C', 'Q'], [1, 2, 3], inplace=True)

df_test.Sex.replace(['male', 'female'], [1,0], inplace=True)
X_test = np.array(df_test.filter(['Pclass','Sex','Embarked','Family','Age'], axis=1))

knn = KNeighborsClassifier(n_neighbors = optimal_k)

knn.fit(X_tr, y_tr)

pred = knn.predict(X_test)

print(pred)
#creating file for submission

df_test['Survived'] = pd.Series(pred, index=df_test.index)
df_test
final_df = df_test.filter(['PassengerId','Survived'], axis=1)
final_df.shape
final_df.to_csv("pred_survival.csv", encoding='utf-8')