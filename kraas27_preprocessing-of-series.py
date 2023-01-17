import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
data_1 = pd.read_csv('../input/monthly-us-auto-registration-tho.csv', sep=';')

data_2 = pd.read_csv('../input/weekly-closings-of-the-dowjones-.csv')
data_1.columns = ['Month', 'Quantity']

series_1 = data_1['Quantity']

data_2.columns = ['Week', 'Index']

series_2 = data_2['Index']
def plot_ts_and_points(ts, start_point, step):

    new_series = [None for i in range(len(ts))]

    for i in range(len(ts)):

        pos = start_point + step * i

        if pos >= len(ts):

            break

        new_series[pos] = ts[pos]

    new_series = pd.Series(new_series)

    

    with plt.style.context('bmh'):

        plt.figure(figsize=(16, 8))

        ts_ax = plt.axes()

        ts.plot(ax=ts_ax, color='blue')

        new_series.plot(ax=ts_ax, style='ro')
plot_ts_and_points(series_1, 2, 4)
plot_ts_and_points(series_2, 6, 12)
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):

    print('Results of Dickey-Fuller Test:')

    dftest = adfuller(timeseries, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])

    for [key, value] in dftest[4].items():

        dfoutput['Critical Value (%s)' % key] = value

    print(dfoutput)
test_stationarity(series_1)
test_stationarity(series_1)
from scipy.stats import boxcox
series_1 = boxcox(series_1, 0)

series_2 = boxcox(series_2, 0)
test_stationarity(series_1)
test_stationarity(series_1)
import statsmodels.api as sm

import statsmodels.tsa.api as smt



def tsplot(y, lags=None, figsize=(14, 8), style='bmh'):

    if not isinstance(y, pd.Series):

        y = pd.Series(y)

    with plt.style.context(style):

        plt.figure(figsize=figsize)

        layout = (4, 1)

        ts_ax = plt.subplot2grid(layout, (0, 0), rowspan=2)

        acf_ax = plt.subplot2grid(layout, (2, 0))

        pacf_ax = plt.subplot2grid(layout, (3, 0))



        y.plot(ax=ts_ax, color='blue', label='Or')

        ts_ax.set_title('Original')



        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.05)

        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.05)



        plt.tight_layout()

    return
import numpy as np
series_1_diff = np.diff(series_1, 2)
tsplot(series_1_diff)
test_stationarity(series_1_diff)
series_1_final = series_1_diff[12:] - series_1_diff[:-12]
tsplot(series_1_final)
series_2_diff = np.diff(series_2, 1)
tsplot(series_2_diff)
test_stationarity(series_2_diff)