# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import mean_squared_error

from sklearn.feature_selection import VarianceThreshold

import scipy as sp

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('/kaggle/input/bits-f464-l1/train.csv') 

df.head()
X=df.iloc[:,0:103]

y=df.iloc[:,103]
scaler = StandardScaler()

scaler.fit(X)

X=scaler.transform(X)

print(X)
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))

X = sel.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=69)

X_train=X

y_train=y
reg=RandomForestRegressor(max_depth=18, random_state=0)

reg.fit(X_train,y_train)
y_test.values.reshape(-1,1)

ypr=reg.predict(X_test).reshape(-1,1)

yact=y_test.values.reshape(-1,1)
mean_squared_error(yact, ypr)
dftest=pd.read_csv('/kaggle/input/bits-f464-l1/test.csv')

Xtest=scaler.transform(dftest)

Xtest=sel.transform(Xtest)

ytest=reg.predict(Xtest).reshape(-1,1)

(pd.DataFrame(ytest)).to_csv('opforest.csv',index=True)