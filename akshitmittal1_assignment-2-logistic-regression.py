# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from pandas import Series, DataFrame

import matplotlib.pyplot as plt

import matplotlib as mpl

import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/train.csv')

df.head()
df_test = pd.read_csv('../input/test.csv')

df_test.head()
df.info()
df.describe()
correlation = df.corr()

plt.matshow(correlation, cmap= 'Reds')

plt.colorbar()
df['artistID'].value_counts().head()
#So far, we have got to know about our data. I don't think song name would be much useful in prediction.
df_y = df['Top10']

del df['Top10']
data = pd.concat([df,df_test])
data.head()
data.shape
year_dummies = pd.get_dummies(data['year'], prefix='year')

data = pd.concat([data, year_dummies], axis=1)

data.drop('year', axis=1, inplace=True)
data.drop(['songtitle', 'artistname', 'songID'], axis=1, inplace=True)
data.head()
artistID_dummies = pd.get_dummies(data['artistID'], prefix='ID')

data = pd.concat([data, artistID_dummies], axis=1)

data.drop('artistID', axis=1, inplace=True)
time_dummies = pd.get_dummies(data['timesignature'], prefix='time')

data = pd.concat([data, time_dummies], axis=1)

data.drop('timesignature', axis=1, inplace=True)
data.head()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

data_scaled = DataFrame(scaler.fit_transform(data))
data_scaled.head()
df_train = data_scaled[0:4999]

df_test = data_scaled[4999:]
from sklearn.model_selection import GridSearchCV

import sklearn.metrics as score

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression(max_iter=500)

grid={"C":[0.1,0.5,1,5, 10]}

gs=GridSearchCV(estimator=lr,param_grid=grid,scoring="roc_auc",cv=10)

gs.fit(df_train,df_y)
y_pred = gs.best_estimator_.predict_proba(df_train)

print("AUC on training data:",score.roc_auc_score(df_y,y_pred.T[1]))

print("AUC on testing data:",gs.best_score_)
y_pred = gs.predict_proba(df_test)

df = pd.DataFrame({})

df["songID"] = pd.read_csv("../input/test.csv").songID

df["Top10"] = y_pred.T[1]

df.to_csv("final.csv",index=False)