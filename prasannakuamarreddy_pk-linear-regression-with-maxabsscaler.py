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
dataset = pd.read_csv('/kaggle/input/random-linear-regression/train.csv')

dataset.head()
dataset.isnull().sum().sum()

x = dataset['x']

y = dataset['y']
x
y
x.isnull().sum().sum()
y.isnull().sum().sum()
df2 = dataset.fillna(dataset.mean())

X = df2['x']

Y = df2['y']
df2.isna().sum().sum()

break
X=X.values.reshape(1,-1)

Y=Y.values.reshape(1,-1)

X.shape

Y.shape
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()  

model = regressor.fit(X.transpose(),Y.transpose())
df_test = pd.read_csv('/kaggle/input/random-linear-regression/test.csv')

x_test= df_test['x']

y_test= df_test['y']

x_t = x_test.values.reshape(1,-1)
Y_Predict = model.predict(x_t.transpose())
import matplotlib.pyplot as plt



plt.plot(x_t.transpose(),Y_Predict)

plt.scatter(x_t,y_test)

plt.show()
from sklearn import preprocessing

from sklearn.preprocessing import MaxAbsScaler



MaxAbsScaler = preprocessing.MaxAbsScaler()

X_train_MaxAbsScaler = MaxAbsScaler.fit_transform(X.transpose())



X_test_MaxAbsScaler = MaxAbsScaler.transform(x_t)
Y_train_MaxAbsScaler = MaxAbsScaler.fit_transform(Y.transpose())



y_pred_MaxAbsScaler = MaxAbsScaler.transform(Y_Predict)

import matplotlib.pyplot as plt



plt.scatter(X_test_MaxAbsScaler.transpose(),y_pred_MaxAbsScaler)

plt.show()