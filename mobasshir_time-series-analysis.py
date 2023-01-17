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
from dateutil.parser import parse
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', parse_dates=['date'])

df.head()
df2 = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', parse_dates=['date'],index_col='date')

df2.head()
df2.index
ts=df2['value']

ts.head()
#import as series object

data = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', parse_dates=['date'], index_col='date',squeeze= True)

data.head()
print(type(data))
data['1991-07-01']
from datetime import datetime

data[datetime(1991,7,1)]
ts['1991-07-01':'1991-12-01']
ts[:'1991-12-01']
ts['1992']
ts1=ts.sort_index()

ts1.head()
# dataset source: https://github.com/rouseguy

df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/MarketArrivals.csv')

df = df.loc[df.market=='MUMBAI', :]

df.head()
from dateutil.parser import parse

import pandas as pd

 

#import as dataframe

data = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', parse_dates=['date'],index_col='date')

 

print(data.size)

print(type(data))

print (data.isnull().sum())
print(data.describe())
# Time series data source: fpp pacakge in R.

import matplotlib.pyplot as plt

 

# Draw Plot

def plot_data(data, x, y, title="", xlabel='Date', ylabel='Value', dpi=100):

    plt.figure(figsize=(16,5), dpi=dpi)

    plt.plot(x, y, color='tab:red')

    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)

    plt.show()

 

plot_data(data, x=data.index, y=data, title='Monthly anti-diabetic drug sales in Australia from 1992 to 2008.') 

plt.show()
# Import data

import numpy as np

 

x = data.index

y1 = data.value

 

# Plot

fig, ax = plt.subplots(1, 1, figsize=(16,5), dpi= 120)

plt.fill_between(x, y1=y1, y2=-y1, alpha=0.5, linewidth=2, color='seagreen')

plt.ylim(-35, 35)

plt.title('Monthly anti-diabetic drug sales in Australia from 1992 to 2008 (Two Side View)', fontsize=16)

plt.hlines(y=0, xmin=np.min(data.index), xmax=np.max(data.index), linewidth=.5)

plt.show()
import matplotlib as mpl

X=data

 

# Prepare data

X['year'] = [d.year for d in X.index]

X['month'] = [d.strftime('%b') for d in X.index]

years = X['year'].unique()

 

# Prep Colors

np.random.seed(100)

mycolors = np.random.choice(list(mpl.colors.XKCD_COLORS.keys()), len(years), replace=False)

 

# Draw Plot

plt.figure(figsize=(16,12), dpi= 80)

for i, y in enumerate(years):

    if i > 0:        

        plt.plot('month', 'value', data=X.loc[X.year==y, :], color=mycolors[i], label=y)

        plt.text(X.loc[X.year==y, :].shape[0]-.9, X.loc[X.year==y, 'value'][-1:].values[0], y, fontsize=12, color=mycolors[i])

 

# Decoration

plt.gca().set(xlim=(-0.3, 11), ylim=(2, 30), ylabel='$Drug Sales$', xlabel='$Month$')

plt.yticks(fontsize=12, alpha=.7)

plt.title("Seasonal Plot of Drug Sales Time Series", fontsize=20)

plt.show()
import matplotlib 

print('matplotlib: {}'.format(matplotlib.__version__))
# Import Data

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

 

# Draw Plot

fig, axes = plt.subplots(1, 2, figsize=(20,7), dpi= 80)

sns.boxplot(x='year', y='value', data=X, ax=axes[0])

sns.boxplot(x='month', y='value', data=X.loc[~X.year.isin([1991, 2008]), :])

 

# Set Title

axes[0].set_title('Year-wise Box Plot\n(The Trend)', fontsize=18); 

axes[1].set_title('Month-wise Box Plot\n(The Seasonality)', fontsize=18)

plt.show()
# Import Data

from matplotlib import pyplot as plt

import pandas as pd

fig, axes = plt.subplots(1,3, figsize=(20,4), dpi=100)

pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/guinearice.csv', parse_dates=['date'], index_col='date').plot(title='Trend Only', legend=False, ax=axes[0])

 

pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/sunspotarea.csv', parse_dates=['date'], index_col='date').plot(title='Seasonality Only', legend=False, ax=axes[1])

 

pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/AirPassengers.csv', parse_dates=['date'], index_col='date').plot(title='Trend and Seasonality', legend=False, ax=axes[2])

plt.show()
from statsmodels.tsa.seasonal import seasonal_decompose

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

 

data = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', parse_dates=['date'],index_col='date')

 

#additive decomposition

result_add = seasonal_decompose(data, model='additive',extrapolate_trend='freq')

 

#multiplicative decomposition

result_mul = seasonal_decompose(data, model='multiplicative',extrapolate_trend='freq')

 

#plot

result_mul.plot().suptitle('Multiplicative Decompose')

result_add.plot().suptitle('Additive Decompose')

plt.show()
# Extract the Components ----

reconst= pd.concat([result_mul.seasonal, result_mul.trend, result_mul.resid, result_mul.observed], axis=1)

reconst.columns = ['seasonal', 'trend', 'residual', 'actual_values']

reconst.head()
data.hist()

plt.show()
split = int(len(data) / 2)

X1, X2 = data[0:split], data[split:]

mean1, mean2 = X1.mean(), X2.mean()

var1, var2 = X1.var(), X2.var()

print('mean1=%f, mean2=%f' % (mean1, mean2))

print('variance1=%f, variance2=%f' % (var1, var2))
from statsmodels.tsa.stattools import adfuller

X = data.value

result = adfuller(X)

print('ADF Statistic: %f' % result[0])

print('p-value: %f' % result[1])
from statsmodels.tsa.stattools import kpss

result = kpss(X, regression='c')

print('\nKPSS Statistic: %f' % result[0])

print('p-value: %f' % result[1])
from scipy import signal

detrended = signal.detrend(data.value)

plt.plot(detrended)

plt.title('Drug Sales detrended', fontsize=22)

plt.show()
# Using statmodels: Subtracting the Trend Component.

result_mul = seasonal_decompose(data.value, model='multiplicative', extrapolate_trend='freq')

detrended = data.value - result_mul.trend

plt.plot(detrended)

plt.title('Drug Sales detrended by subtracting the trend component', fontsize=12)

plt.show()
from statsmodels.tsa.seasonal import seasonal_decompose

stationarized = result_mul.resid

plt.plot(stationarized)

plt.title('Drug Sales stationarized', fontsize=16)

plt.show()
from numpy import log

result = adfuller(data.value.dropna())

print('p-value: %f' % result[1])

data_log=log(data.value)

 

#After 1st difference

data_diff=data_log.diff()

result_diff = adfuller(data_diff.dropna())

print('p-value after 1st difference: %f' % result_diff[1])
from pandas.plotting import autocorrelation_plot

 

# Draw Plot

autocorrelation_plot(data.value.tolist())
from statsmodels.graphics.tsaplots import plot_pacf

plot_pacf(data_diff.dropna())

 

plt.show()
from statsmodels.graphics.tsaplots import plot_acf

plot_acf(data_diff.dropna())

 

plt.show()
from statsmodels.tsa.arima_model import ARIMA

 

# 1,1,2 ARIMA Model

model = ARIMA(data_log, order=(1,1,2))

model_fit = model.fit(disp=0)

print(model_fit.summary())
from statsmodels.tsa.arima_model import ARIMA

 

# 2,1,2 ARIMA Model

model = ARIMA(data_log, order=(2,1,2))

model_fit = model.fit(disp=0)

print(model_fit.summary())
# Plot residual errors

residuals = pd.DataFrame(model_fit.resid)

fig, ax = plt.subplots(1,2)

residuals.plot(title="Residuals", ax=ax[0])

residuals.plot(kind='kde', title='Density', ax=ax[1])

plt.show()
from statsmodels.tsa.arima_model import ARIMA

 

# AR-2 Model

model = ARIMA(data_log, order=(2,1,0))

model_fit = model.fit(disp=0)

print(model_fit.summary())
from statsmodels.tsa.arima_model import ARIMA

 

# MA-2 Model

model = ARIMA(data, order=(0,1,2))

model_fit = model.fit(disp=0)

print(model_fit.summary())
from statsmodels.tsa.arima_model import ARIMA

 

# ARMA Model

model = ARIMA(data_diff.dropna(), order=(2,0,2))

model_fit = model.fit(disp=0)

print(model_fit.summary())