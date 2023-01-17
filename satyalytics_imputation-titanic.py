import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno
train = pd.read_csv('https://raw.githubusercontent.com/satyalytics/mini_projects/master/titanic/data/train.csv')

test = pd.read_csv('https://raw.githubusercontent.com/satyalytics/mini_projects/master/titanic/data/test.csv')
train.info()
msno.bar(train)
msno.matrix(train.sort_values('Age'))
# There are lots of missing values in Cabin column. So I drop it.

df = train.drop('Cabin',axis=1)
df.info()
df['age_missing'] = 0

df.loc[df['Age'].isna(),'age_missing'] = 1

df['age_missing'].sum()
df.head(2)
df2 = df.drop(['Name','Ticket','PassengerId'],axis=1)
def test_mod(df):

    """

        Split the dataframe into train and test, apply random forest and print score

        Args:

            df: The df on which work will be done

        Returns:

            score: validation score on test set

    """

    X_train, X_test, y_train,y_test = train_test_split(df.drop('Survived',axis=1), df['Survived'], test_size=0.1)

    rf = RandomForestClassifier()

    rf.fit(X_train,y_train)

    return rf.score(X_test,y_test)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer
imput_d = {}

simple_imp = ['mean','median','most_frequent']

for i in simple_imp:

    df3 = df2.copy(deep=True)

    df3 = pd.get_dummies(df3)

    col = df3.columns

    s = SimpleImputer(strategy=i)

    df3 = s.fit_transform(df3)

    df3 = pd.DataFrame(df3, columns=col)

    score = test_mod(df3)

    imput_d[i] = score
imput_d
it = IterativeImputer()

df3 = df2.copy(deep=True)

df3 = pd.get_dummies(df3)

col = df3.columns

df3 = it.fit_transform(df3)

df3 = pd.DataFrame(df3, columns=col)

imput_d['Iter'] = test_mod(df3)
knn = KNNImputer()

df3 = df2.copy(deep=True)

df3 = pd.get_dummies(df3)

col = df3.columns

df3 = it.fit_transform(df3)

df3 = pd.DataFrame(df3, columns=col)

imput_d['knn'] = test_mod(df3)
imput_d
# In mean imputation and KNN Imputation gives heighest accuracy. So we select the mean imputation for further use.

df3 = df2.copy(deep=True)

sim = SimpleImputer(strategy='mean')

df3 = pd.get_dummies(df3)

col = df3.columns

df3 = sim.fit_transform(df3)

df3 = pd.DataFrame(df3, columns=col)