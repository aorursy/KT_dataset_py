# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from imblearn.under_sampling import ClusterCentroids

from sklearn.ensemble import RandomForestClassifier

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/killer-shrimp-invasion/train.csv')

test = pd.read_csv('/kaggle/input/killer-shrimp-invasion/test.csv',)

train.head(5)
plt.plot(train.Salinity_today,train.Presence,'.')
plt.plot(train.Temperature_today,train.Presence,'.')
plt.plot(train.Depth,train.Presence,'.')
train=train.dropna()

test=test.fillna(method='ffill') 
X=train.drop(['pointid','Presence'], axis=1)

Y=train.drop(['pointid','Salinity_today','Temperature_today','Substrate','Depth','Exposure'],axis=1)
cc = ClusterCentroids(random_state=0)

X_resampled, y_resampled = cc.fit_resample(X, Y)
clf = RandomForestClassifier(max_depth=20,criterion='gini')

# обучение модели

clf.fit(X_resampled, y_resampled)

# предсказание на тестовой выборке

y_pred = clf.predict(test.drop(['pointid'], axis=1))
sub = pd.read_csv("/kaggle/input/killer-shrimp-invasion/temperature_submission.csv")

sub["Presence"] = y_pred

sub.to_csv("submission.csv", index=False)

sub.head()