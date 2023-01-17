# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import matplotlib
import matplotlib.pylab as plt
import matplotlib.finance as mpf
matplotlib.style.use('seaborn')
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 5
from plotly.graph_objs import *
from plotly.offline import init_notebook_mode, iplot, iplot_mpl
init_notebook_mode()
from tqdm import tqdm
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('../input/Data/Stocks/goog.us.txt')
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')
df['Close_diff'] = df['Close']-df.shift()['Close']
df['Close_diff_log'] = np.log1p(df['Close'])-np.log1p(df.shift()['Close'])
df = df[['Close', 'Close_diff', 'Close_diff_log']]
df = df.dropna()
train = df['2015-01-01':'2017-01-01']
test = df['2017-01-01':]
len(train), len(test)
iplot([Scatter(x=train.index, y=train['Close_diff'])])
iplot([Histogram(x=train['Close_diff'])])
plt = sm.graphics.tsa.plot_acf(train['Close_diff'], lags=40)
plt.show()
plt = sm.graphics.tsa.plot_pacf(train['Close_diff'], lags=40)
plt.show()
res = sm.tsa.arma_order_select_ic(train['Close_diff'], ic='aic', trend='nc')
res
from statsmodels.tsa.arima_model import ARIMA

arima_3_1_0 = ARIMA(train['Close'].as_matrix(), order=(3, 1, 0)).fit(dist=False)
arima_3_1_0.params
plt = sm.graphics.tsa.plot_acf(arima_3_1_0.resid, lags=40)
plt.show()
plt = sm.graphics.tsa.plot_pacf(arima_3_1_0.resid, lags=40)
plt.show()
ts = train['Close'].as_matrix()
predictions = np.empty((0), dtype=np.float32)
n_pre = 100
for i in tqdm(range(n_pre)):
    arima_3_1_0 = ARIMA(ts, order=(3, 1, 0)).fit(dist=False)
    predict = arima_3_1_0.forecast()[0]
    predictions = np.hstack([predictions, predict])
    ts = np.hstack([ts, predict])
nans = np.zeros(len(train))
nans[:] = np.nan
orgs = pd.concat([train['Close'], test[:n_pre]['Close']])
orgs = pd.DataFrame({
    'Date': orgs.index,
    'Original': orgs.as_matrix(),
    'Prediction': np.hstack([nans, predictions])
})
orgs = orgs.set_index('Date')
orgs.plot(color=['blue', 'red'])
plt.show()
def search_param(path, start='2015-01-01', end='2017-01-01'):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    df['Close_diff'] = df['Close']-df.shift()['Close']
    df = df[['Close', 'Close_diff']]
    df = df.dropna()
    train = df[start:end]
    test = df[end:]
    res = sm.tsa.arma_order_select_ic(train['Close_diff'], ic='aic', trend='nc')
    print(res)
    return train, test
def show_predict(train, test, n_pre, p, d, q):
    arima_init = ARIMA(train['Close'].as_matrix(), order=(p, d, q)).fit(dist=False)
    ts = train['Close'].as_matrix()
    predictions = np.empty((0), dtype=np.float32)
    for i in tqdm(range(n_pre)):
        arima = ARIMA(ts, order=(p, d, q)).fit(dist=False)
        predict = arima.forecast()[0]
        predictions = np.hstack([predictions, predict])
        ts = np.hstack([ts, predict])
    nans = np.zeros(len(train))
    nans[:] = np.nan
    orgs = pd.concat([train['Close'], test[:n_pre]['Close']])
    orgs = pd.DataFrame({
        'Date': orgs.index,
        'Original': orgs.as_matrix(),
        'Prediction': np.hstack([nans, predictions])
    })
    orgs = orgs.set_index('Date')
    orgs.plot(color=['blue', 'red'])
    plt.show()
    return arima_init
train, test = search_param('../input/Data/Stocks/aapl.us.txt')
plt = sm.graphics.tsa.plot_acf(train['Close_diff'], lags=40)
plt.show()
plt = sm.graphics.tsa.plot_pacf(train['Close_diff'], lags=40)
plt.show()
arima_0_1_1 = show_predict(train, test, 100, 0, 1, 1)
print(arima_0_1_1.params)
plt = sm.graphics.tsa.plot_acf(arima_0_1_1.resid, lags=40)
plt.show()
plt = sm.graphics.tsa.plot_pacf(arima_0_1_1.resid, lags=40)
plt.show()
train, test = search_param('../input/Data/Stocks/fb.us.txt')
arima_4_1_1 = show_predict(train, test, 100, 4, 1, 1)
plt = sm.graphics.tsa.plot_acf(arima_4_1_1.resid, lags=40)
plt.show()
plt = sm.graphics.tsa.plot_pacf(arima_4_1_1.resid, lags=40)
plt.show()