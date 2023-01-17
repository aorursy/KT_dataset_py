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
from pylab import rcParams

from plotly import tools

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.figure_factory as ff

import statsmodels.api as sm

from numpy.random import normal, seed

from scipy.stats import norm

from statsmodels.tsa.arima_model import ARMA

from statsmodels.tsa.stattools import adfuller

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.arima_process import ArmaProcess

from statsmodels.tsa.arima_model import ARIMA

import math

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

df = pd.read_csv("../input/dow-jones-project/dow_jones_index.csv",index_col='date', parse_dates=True, squeeze=True)

df = df[['close']]

df.head()
df.isnull().sum()
df1 = df.replace(to_replace=[r'^\s*$', r'[?]', r'\'\s*\'', 'N/A', 'None'],value=np.nan, regex=True)

df1.head()
df1 = df.replace(to_replace=['\$'],value='', regex=True)

df1.head()
df1.isnull().any()
df1.dtypes
df1 = df1.iloc[1:]

df1 = df1.fillna(method='ffill')

df1.head()
df1.dtypes
df1 = df1.astype({'close':float})

df1.dtypes
df1.shape
# check white noise

# Plotting white noise

rcParams['figure.figsize'] = 16, 6

white_noise = np.random.normal(loc=0, scale=1, size=1000)

# loc is mean, scale is variance

plt.plot(white_noise)
# Plotting autocorrelation of white noise

plot_acf(white_noise,lags=20)

plt.show()
# Augmented Dickey-Fuller test on close 

adf = adfuller(df1["close"])

print("p-value of close: {}".format(float(adf[1])))
# Prediction using ARIMA model

rcParams['figure.figsize'] = 16, 6

model = ARIMA(df1["close"].diff().iloc[1:].values, order=(2,1,0))

result = model.fit()

print(result.summary())

result.plot_predict(start=600, end=700)

plt.show()
rmse = math.sqrt(mean_squared_error(df1["close"].diff().iloc[600:701].values, result.predict(start=600,end=700)))

print("The root mean squared error is {}.".format(rmse))
# Forecasting and predicting close ARMA Model

model = ARMA(df1["close"].diff().iloc[1:].values, order=(3,3))

result = model.fit()

print(result.summary())

print("μ={}, ϕ={}, θ={}".format(result.params[0],result.params[1],result.params[2]))

result.plot_predict(start=600, end=700)

plt.show()
rmse = math.sqrt(mean_squared_error(df1["close"].diff().iloc[600:701].values, result.predict(start=600,end=700)))

print("The root mean squared error is {}.".format(rmse))
# Predicting closing price SARIMA models'

train_sample = df1["close"].diff().iloc[1:].values

model = sm.tsa.SARIMAX(train_sample,order=(4,0,4),trend='c')

result = model.fit(maxiter=1000,disp=False)

print(result.summary())

predicted_result = result.predict(start=0, end=700)

result.plot_diagnostics()

# calculating error

rmse = math.sqrt(mean_squared_error(train_sample[1:702], predicted_result))

print("The root mean squared error is {}.".format(rmse))