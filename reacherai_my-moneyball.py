# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn import tree



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# load data

df = pd.read_csv(r"../input/baseball.csv")
# impute data

imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)

imputer = imputer.fit(df[['RankSeason', 'RankPlayoffs', 'OOBP', 'OSLG']])

df[['RankSeason', 'RankPlayoffs', 'OOBP', 'OSLG']] = imputer.transform(df[['RankSeason', 'RankPlayoffs', 'OOBP', 'OSLG']])
# replace League with int

df.League.replace(['NL', 'AL'], [1, 0], inplace=True)
# drop useless cols

df = df[df.columns.difference(['RankPlayoffs', 'Team'])]

y = df[['Playoffs']]

y = np.ravel(y)

X = df[df.columns.difference(['Playoffs'])]
# split data

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
# standardize data

std = StandardScaler()

x_train = std.fit_transform(x_train)

x_test = std.transform(x_test)
# train svc model

svc = SVC(kernel='linear').fit(x_train, y_train)
# predict

predictions = svc.predict(x_test)

print(predictions)

print(accuracy_score(y_test, predictions))

print(classification_report(y_test, predictions))