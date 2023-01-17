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
train_da = pd.read_csv('/kaggle/input/random-linear-regression/train.csv')
train_da = train_da.fillna(0)
x=train_da.iloc[:, :1].values
y=train_da.iloc[:, 1:2].values
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train , y_test = train_test_split(x, y, test_size = 2/5, random_state = 0)
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
y_pred = linear_reg.predict(X_test)
import matplotlib.pyplot as plt 
#plt.scatter(X_test, y_test, color='Red')
plt.plot(X_test, linear_reg.predict(X_test), color='Blue')
from sklearn.metrics import mean_squared_error
from math import sqrt
score1 = sqrt(mean_squared_error(y_test,y_pred))/100
score1
