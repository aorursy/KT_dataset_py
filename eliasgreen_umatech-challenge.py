# Imports:

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



RS = 1979
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_train = pd.read_csv("/kaggle/input/umatech-dataset/TRAIN_PREPARED.csv")

df_train_additional = pd.read_csv("/kaggle/input/umatech-dataset/TRAIN_PREPARED.csv")

df_test = pd.read_csv("/kaggle/input/umatech-dataset/TEST_PREPARED.csv")

df_test_additional = pd.read_csv("/kaggle/input/umatech-dataset/TRAIN_PREPARED.csv")
df_train.info()

df_train.describe()
df_train_additional.info()

df_train_additional.describe()
df_train.columns.to_series().groupby(df_train.dtypes).groups
train_with_outliers = df_train.drop(['cut_date', 'user', 'first_date', 'last_date'], axis=1)

train_with_outliers.info()

data_with_outliers= train_with_outliers.sample(n=20000, random_state=RS)



from sklearn.model_selection import train_test_split

X_train, X_validation, y_train, y_validation = train_test_split(data_with_outliers.drop(['label'], axis=1), data_with_outliers['label'], test_size=0.33, random_state=RS)
test_with_outliers = df_test.drop(['cut_date', 'first_date', 'last_date'], axis=1)

test_with_outliers.info()
#from sklearn.svm import SVC

#clf = SVC(gamma='auto', random_state=RS)

#clf.fit(X_train, y_train) 
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

parameters = {'n_estimators':[100, 150], 'max_depth':[4, 7, 10], 'max_features':[0.5, 0.7]}

RFC = RandomForestClassifier(random_state=RS)

clf = GridSearchCV(RFC, parameters, cv=5, scoring='accuracy')

clf.fit(X_train, y_train)



print('RandomForestClassifier...')

print('Best Params:')

print(clf.best_params_)

print('Best CV Score:')

print(clf.best_score_)
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=56, random_state=RS)

clf.fit(X_train, y_train)
validation_prediction = clf.predict(X_validation)
from sklearn.metrics import accuracy_score

accuracy_score(y_validation, validation_prediction)