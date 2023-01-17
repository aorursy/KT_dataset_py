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

import pandas_profiling as pp

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures



data = pd.read_csv('/kaggle/input/nasa-airfoil-self-noise/NASA_airfoil_self_noise.csv')
data.head()
pp.ProfileReport(data)
Y = data['Sound']

X = data.drop(['Sound'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
polynomial_features= PolynomialFeatures(degree=4)

X_train_p = polynomial_features.fit_transform(X_train)

X_test_p = polynomial_features.fit_transform(X_test)

model = LinearRegression().fit(X_train_p, y_train)

y_pred = model.predict(X_test_p)

print('r2 score:',r2_score(y_test ,y_pred))