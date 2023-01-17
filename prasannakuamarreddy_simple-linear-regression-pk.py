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
df = pd.read_csv('/kaggle/input/random-linear-regression/train.csv')

df_test = pd.read_csv('/kaggle/input/random-linear-regression/test.csv')
xi = df['x']

yi = df['y']
yi.size
xi.isna().sum()

yi.isna().sum()
yi = yi.fillna(yi.mean())

yi
xo = df_test['x']

yo = df_test['y']
xi = xi.values.reshape(1,-1)
xi.shape
yi = yi.values.reshape(1,-1)
xi.shape
yi.shape
xi = xi.reshape(1,-1)
xi.shape
xi = xi.transpose()
xi
xi.shape
yi.shape
yi = yi.transpose()
yi.shape
yo.isna().sum()
xo = xo.values.reshape(1,-1)

yo = yo.values.reshape(1,-1)
xo.shape
xo= xo.transpose()
xo.shape
yo.shape
yo = yo.transpose()
yo.shape
from sklearn.linear_model import LinearRegression



regressor = LinearRegression()

model = regressor.fit(xi,yi)
import matplotlib.pyplot as plt



plt.scatter(xi,yi)

plt.show()
YP = model.predict(xo)
plt.scatter(xo,yo)

plt.plot(xo,YP,color = 'red')

plt.show()
Comparison = pd.DataFrame({'Actual': yo.flatten(), 'Predicted': YP.flatten()})

Comparison
import sklearn.metrics as mtr



print('Mean Absolute Error:', mtr.mean_absolute_error(yo,YP))  

print('Mean Squared Error:', mtr.mean_squared_error(yo,YP))  

print('Root Mean Squared Error:', np.sqrt(mtr.mean_squared_error(yo,YP)))
from sklearn import preprocessing

from sklearn.linear_model import LinearRegression



mm_scaler = preprocessing.MinMaxScaler()

xi_minmax = mm_scaler.fit_transform(xi)

yi_minmax = mm_scaler.fit_transform(yi)

xo_minmax = mm_scaler.transform(xo)

yo_minmax = mm_scaler.transform(yo)



reg = LinearRegression()

modelMinMax = reg.fit(xi_minmax,yi_minmax)



YP_minmax = modelMinMax.predict(xo_minmax)
Comparison_MinMax = pd.DataFrame({'Actual':yo_minmax.flatten(),'Predict':YP_minmax.flatten()})
import sklearn.metrics as mtr



print('Mean Absolute Error:', mtr.mean_absolute_error(yo_minmax,YP_minmax))  

print('Mean Squared Error:', mtr.mean_squared_error(yo_minmax,YP_minmax))  

print('Root Mean Squared Error:', np.sqrt(mtr.mean_squared_error(yo_minmax,YP_minmax)))
import matplotlib.pyplot as plt



plt.scatter(xo_minmax,yo_minmax)

plt.plot(xo_minmax,YP_minmax)

plt.show()