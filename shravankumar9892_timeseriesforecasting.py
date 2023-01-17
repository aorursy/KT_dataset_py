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
file = open('/kaggle/input/demand-forecasting-kernels-only/train.csv', 'r')

print(file.read())
import pandas as pd



data = pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/train.csv')
# Nth item -> 2013-01-01 to 2017-12-31

# Total time period: 5 years



# Start forecasting for 1 store and 1 item

sales1 = data[(data['store'] == 1) & (data['item'] == 1)]

sales1.drop(['store', 'item'], axis=1, inplace=True)
sales1 = sales1.set_index('date')
sales1.index = pd.to_datetime(sales1.index)

y = sales1['sales'].resample('MS').mean()
import itertools



p = d = q = range(0, 2)

pdq = list(itertools.product(p, d, q))

seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]
import statsmodels.api as sm



# Selecting optimum parameters

for param in pdq:

    for param_seasonal in seasonal_pdq:

        try:

            mod = sm.tsa.statespace.SARIMAX(y,

                                            order=param,

                                            seasonal_order=param_seasonal,

                                            enforce_stationarity=False,

                                            enforce_invertibility=False)

            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))

        except:

            continue
# ARIMA(1, 1, 1)x(1, 1, 0, 12)12 - AIC:112.96784307172449

# Optimum combination



# Fitting the model with pdq and seasonla_pdq of lowest AIC value from previous result

mod = sm.tsa.statespace.SARIMAX(y,

                                order=(1, 1, 1),              # pdq

                                seasonal_order=(1, 1, 0, 12), # seasonal_pdq

                                enforce_stationarity=False,

                                enforce_invertibility=False)

results = mod.fit()

print(results.summary().tables[1])
import matplotlib.pyplot as plt

import matplotlib

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

from pylab import rcParams



matplotlib.rcParams['axes.labelsize'] = 10

matplotlib.rcParams['xtick.labelsize'] = 13

matplotlib.rcParams['ytick.labelsize'] = 13

matplotlib.rcParams['text.color'] = 'k'

matplotlib.rcParams['axes.titlesize'] = 20

matplotlib.rcParams['legend.fontsize'] = 20

rcParams['figure.figsize'] = 18, 13



# Model diagnostics

results.plot_diagnostics(figsize=(16, 8))

plt.show()
rcParams['figure.figsize'] = 25, 20



decomposition = sm.tsa.seasonal_decompose(y["2016":], model='additive')

fig = decomposition.plot()

plt.show()
matplotlib.rcParams['axes.labelsize'] = 20

matplotlib.rcParams['xtick.labelsize'] = 20

matplotlib.rcParams['ytick.labelsize'] = 20

matplotlib.rcParams['text.color'] = 'k'

matplotlib.rcParams['axes.titlesize'] = 20

matplotlib.rcParams['legend.fontsize'] = 25

rcParams['figure.figsize'] = 18, 13



pred = results.get_prediction(start=pd.to_datetime('2017-01-01'), dynamic=False)

pred_ci = pred.conf_int()

ax = y['2013':].plot(label='observed')

pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(25, 13))



# Coloring the area for range forecasting

ax.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color='k', alpha=.2)



# Labels

ax.set_xlabel('Date')

ax.set_ylabel('Sales')

plt.legend()

plt.show()