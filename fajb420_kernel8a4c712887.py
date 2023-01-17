# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import numpy as np

from sklearn.linear_model import LinearRegression

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train_V2.csv', low_memory=False)



# No labels

df_test = pd.read_csv('../input/test_V2.csv', low_memory=False)
dummies_train = pd.get_dummies(df_train['matchType'])

dummies_test = pd.get_dummies(df_test['matchType'])
df_train = pd.concat([df_train, dummies_train], axis=1)

df_test = pd.concat([df_test, dummies_test], axis=1)
df_train
df_train.drop(['Id', 'groupId', 'matchId', 'matchType'], axis=1, inplace=True)
df_train.isna().sum()
df_train.dropna(0, inplace=True)
feature_cols = df_train.columns[df_train.columns!='winPlacePerc']



X = df_train[feature_cols]

y = df_train.winPlacePerc
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))



X_train_scaled = scaler.fit_transform(X_train)

X_train = pd.DataFrame(X_train_scaled)



X_test_scaled = scaler.transform(X_test)

X_test = pd.DataFrame(X_test_scaled)
def submission(df, features, algorithm):

    x_oos = df[features]

    algo = algorithm

    algo.fit(X_train, y_train)

    pred = algo.predict(x_oos)

    

    test_id = df["Id"]

    sub = pd.DataFrame({'Id': test_id, "winPlacePerc": pred} , columns=['Id', 'winPlacePerc'])

    return sub.head()
print(submission(df_test, feature_cols, LinearRegression()))
'''

Create a submission file

'''

import random



# predict class probabilities for the actual testing data (not X_test)

X_oos = df_test[feature_cols]



svr = LinearRegression()

svr.fit(X_train,y_train)

pred = svr.predict(X_oos)





test_id = df_test["Id"]

sub = pd.DataFrame({'Id': test_id, "winPlacePerc": pred} , columns=['Id', 'winPlacePerc'])

sub.to_csv("submission_V2.csv", index = False)