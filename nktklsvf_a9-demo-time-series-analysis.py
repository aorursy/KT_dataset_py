import warnings

warnings.filterwarnings('ignore')

import numpy as np

import pandas as pd

import os



from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from plotly import graph_objs as go

import requests

import pandas as pd

from numpy.linalg import LinAlgError



print(__version__) # need 1.9.0 or greater

init_notebook_mode(connected = True)





def plotly_df(df, title = ''):

    data = []

    

    for column in df.columns:

        trace = go.Scatter(

            x = df.index,

            y = df[column],

            mode = 'lines',

            name = column

        )

        data.append(trace)

    

    layout = dict(title = title)

    fig = dict(data = data, layout = layout)

    iplot(fig, show_link=False)
df = pd.read_csv('../input/wiki_machine_learning.csv', sep = ' ')

df = df[df['count'] != 0]

df.head()
df.shape
df.date = pd.to_datetime(df.date)
plotly_df(df.set_index('date')[['count']])
from fbprophet import Prophet
predictions = 30



df = df[['date', 'count']]

df.columns = ['ds', 'y']

df.tail()
# You code here

train_df = df[:-predictions] 

m = Prophet()

m.fit(train_df)

future = m.make_future_dataframe(periods=predictions)

forecast = m.predict(future)

m.plot(forecast)
forecast[forecast['ds'] == '2016-01-20']['yhat']
# You code here



from sklearn.metrics import mean_absolute_error



def mean_absolute_percentage_error(y_true, y_pred): 

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



y_true = np.array(df['y'][-predictions:])

y_pred = np.array(forecast['yhat'][-predictions:])
print('MAE:', mean_absolute_error(y_true, y_pred))

print('MAPE:', mean_absolute_percentage_error(y_true, y_pred))
%matplotlib inline

import matplotlib.pyplot as plt

from scipy import stats

import statsmodels.api as sm

plt.rcParams['figure.figsize'] = (15, 10)
# You code here

sm.tsa.stattools.adfuller(train_df['y'])[1]
plt.plot(train_df['ds'], train_df['y'])
import warnings

warnings.filterwarnings('ignore')
from itertools import product



best_aic = float("inf")



best_p, best_d, best_q, best_P, best_D, best_Q = 0, 0, 0, 0, 0, 0



prange = range(4)



for p in prange:

    for d in prange:

        for q in prange:

            for P in prange:

                for D in prange:

                    for Q in prange:

                        try:

                            model = sm.tsa.statespace.SARIMAX(train_df['y'], order=(p, d, q), 

                                        seasonal_order=(P, D, Q, 7)).fit()

                            aic = model.aic

                            if aic < best_aic:

                                best_aic = aic

                                best_p, best_d, best_q, best_P, best_D, best_Q = p, d, q, P, D, Q

                        except (ValueError, LinAlgError):

                            pass
print(best_aic)

print('D = {}, d = {}, Q = {}, q = {}, P = {}, p = {}'.format(best_D, best_d, best_Q, best_q, best_P, best_p)) 