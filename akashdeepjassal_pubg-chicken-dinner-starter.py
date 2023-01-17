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
df_train=pd.read_csv('../input/train.csv')
df_train.head()
import matplotlib.pyplot as plt
print('Distribution of damage dealt')
print('{0:.4f}% players dealt zero damage'.format((df_train['damageDealt'] == 0).sum()/ df_train.shape[0]))
plt.hist(df_train['damageDealt'], bins=40);
print('Distribution of damage dealt')
print('{0:.4f}% players dealt zero damage'.format((df_train['DBNOs'] == 0).sum()/ df_train.shape[0]))
plt.hist(df_train['damageDealt'], bins=40);
df_train=df_train.drop(['groupId'],axis=1)
df_train.head()
y_train=df_train['winPlacePerc']
df_train=df_train.drop(['winPlacePerc'],axis=1)
df_train.shape
y_train.head()
#from sklearn.ensemble import RandomForestRegressor
import os

n=os.cpu_count()
#regr = RandomForestRegressor( n_jobs=n)
#regr.fit(df_train, y_train)
from sklearn.model_selection import train_test_split
X_train, X_val, y_Train, y_val = train_test_split(df_train, y_train, test_size=0.3)
del df_train,y_train
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)

X_val = scaler.transform(X_val)

df_test=pd.read_csv('../input/test.csv')
df_test.head()
