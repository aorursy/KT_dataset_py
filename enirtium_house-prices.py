# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn import preprocessing

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import classification_report
df_train = pd.read_csv('./../input/train.csv')

df_test = pd.read_csv('./../input/test.csv')
'''

Plot a distribution of Saleprice on the training dataset

'''
sns.distplot(df_train['SalePrice'], kde=False)
totals = df_train.isnull().sum()

percent = (df_train.isnull().sum()/df_train.shape[0])*100



missing = pd.concat([totals,percent], axis=1, keys=['Totals','Percent'])

missing = missing.sort_values(['Totals'],ascending=False)

print(missing)
drop_cols = missing[missing['Totals']>1]

df_train = df_train.drop(drop_cols.index.values, 1)

df_test = df_test.drop(drop_cols.index.values, 1)



print(df_train.columns.values)

print(df_train.shape)

print(df_test.shape)
#drop records with null in training set

df_train = df_train.dropna()
nn = df_train.select_dtypes(include=['object'])

print (nn.columns.values)

df_train = df_train.drop(nn.columns.values,axis=1)

df_test = df_test.drop(nn.columns.values,axis=1)
df_test = df_test.fillna(value=-99)
corr = df_train.corr()

sns.heatmap(corr)
#seperate Features and label

X_train = df_train.iloc[:, 1:-1]

y_train = df_train.iloc[:, -1]



X_test = df_test.iloc[:,1:]



clf = RandomForestRegressor(n_estimators=100)

clf.fit(X_train,y_train)

predictions = clf.predict(X_test) 
sns.distplot(predictions, kde=False)