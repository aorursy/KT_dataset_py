# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

production = pd.read_csv("../input/production.csv")
import datetime as dt

production['year'] = pd.to_datetime(production['year'])

production['year']= production['year'].map(dt.datetime.toordinal)



X = production.iloc[:, :-1].values

y = production.iloc[:, 3].values



print(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor_fit = regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

print(y_pred, y_test)

print(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

print(df)
print(regressor_fit.score(X_test, y_test))