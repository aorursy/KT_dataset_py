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
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import pandas as pd
bottle = pd.read_csv('../input/calcofi/bottle.csv')
bottle.head()
bottle.info()
bottle.describe()
sns.heatmap(bottle.corr(),annot= True)
bottle.columns
bottle = bottle[['Depthm', 'T_degC', 'Salnty']]

bottle = bottle[:][:1000]
sns.lmplot(x='Salnty',y='T_degC',data = bottle)
sns.lmplot(x='Salnty',y='T_degC',data = bottle)
sns.lmplot(x='Depthm',y='T_degC',data = bottle)
bottle.fillna(method='ffill', inplace=True)

bottle.dropna(inplace=True)
X = bottle[['Depthm','Salnty','T_degC']]

y = bottle['Salnty']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train.shape,y_train.shape)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
accuracy = lm.score(X_test, y_test)

print(accuracy*100)
y_pred = lm.predict(X_test)

for i in range(10):

    print('Actual value: {:.3f} Predicted Value: {:.3f}'.format(y_test.values[i],y_pred[i]))