# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/HousePrices_HalfMil.csv')

y = data.iloc[ : , -1]

X = data.iloc[ : , : -1]

X_train , X_test , y_train , y_test = train_test_split(X,y , random_state = 20)

clf = RandomForestRegressor(min_samples_leaf = 5)

print(clf)

clf.fit(X_train , y_train)

pred = clf.predict(X_test)

error = mean_absolute_error(y_test , pred)

print('Error : ' , error)

sc = clf.score(X_train , y_train)

sc1 = clf.score(X_test , y_test)

print('Training Score : {}\nTesting Score : {}'.format(sc , sc1))