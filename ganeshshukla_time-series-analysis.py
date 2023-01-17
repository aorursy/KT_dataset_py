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
import warnings                                  # do not disturbe mode

warnings.filterwarnings('ignore')



# Load packages

import numpy as np                               # vectors and matrices

import pandas as pd                              # tables and data manipulations

import matplotlib.pyplot as plt                  # plots

import seaborn as sns                            # more plots



from dateutil.relativedelta import relativedelta # working with dates with style

from scipy.optimize import minimize              # for function minimization



import statsmodels.formula.api as smf            # statistics and econometrics

import statsmodels.tsa.api as smt

import statsmodels.api as sm

import scipy.stats as scs



from itertools import product                    # some useful functions

from tqdm import tqdm_notebook



# Importing everything from forecasting quality metrics

from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error

from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error
import numpy as np

import pandas as pd

import scipy

import statsmodels.api as sm

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn

import statsmodels.graphics.tsaplots as sgt

import statsmodels.tsa.stattools as sts

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.statespace.sarimax import SARIMAX

#from pmdarima.arima import auto_arima

#from arch import arch_model

import warnings

warnings.filterwarnings("ignore")

sns.set()
# MAPE

def mean_absolute_percentage_error(y_true, y_pred): 

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    

def tsplot(y, lags=None, figsize=(12, 7), style='bmh'):

    """

        Plot time series, its ACF and PACF, calculate Dickey–Fuller test

        

        y - timeseries

        lags - how many lags to include in ACF, PACF calculation

    """

    

    if not isinstance(y, pd.Series):

        y = pd.Series(y)

        

    with plt.style.context(style):    

        fig = plt.figure(figsize=figsize)

        layout = (2, 2)

        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)

        acf_ax = plt.subplot2grid(layout, (1, 0))

        pacf_ax = plt.subplot2grid(layout, (1, 1))

        

        y.plot(ax=ts_ax)

        p_value = sm.tsa.stattools.adfuller(y)[1]

        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))

        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)

        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)

        plt.tight_layout()
newdata=pd.read_csv('/kaggle/input/stockdata/GLOBAL DATA/CL1.csv',index_col=['Dates'], parse_dates=['Dates'])

cldata=newdata.copy()

cldata
rng=pd.date_range(start="9/8/2019 3:35",end="4/1/2020 20:05",freq='5T')

rng
plt.figure(figsize=(18, 6))

plt.plot(cldata.Close)

plt.title('Closing Price')

plt.grid(True)

plt.show()
cldata['open_value']=cldata.Open

cldata['close_value']=cldata.Close

cldata['high_value']=cldata.High

cldata['Low_value']=cldata.Low

cldata['Volume_value']=cldata.Volume



del cldata['Open']

del cldata['High']

del cldata['Low']

del cldata['Close']

del cldata['Volume']
cldata['ret_open_value'] = cldata.open_value.pct_change(1).mul(100)

cldata['ret_close_value'] = cldata.close_value.pct_change(1).mul(100)

cldata['ret_high_value'] = cldata.high_value.pct_change(1).mul(100)

cldata['ret_Low_value'] = cldata.Low_value.pct_change(1).mul(100)

cldata['ret_volume_value'] =cldata.Volume_value.pct_change(1).mul(100)
cldata
del cldata['open_value']

del cldata['high_value']

del cldata['Low_value']

del cldata['close_value']

del cldata['Volume_value']
del cldata['ret_open_value']

del cldata['ret_Low_value']

del cldata['ret_high_value']

del cldata['ret_volume_value']
cldata
cldata.isna().sum()
cldata=cldata.fillna(method='bfill')
cldata
tsplot(cldata.ret_close_value, lags=60)
cldata_diff = cldata.ret_close_value - cldata.ret_close_value.shift(5)

tsplot(cldata_diff[5:], lags=60)


# setting initial values and some bounds for them

ps = range(2, 5)

d=1 

qs = range(2, 5)

Ps = range(0, 2)

D=1 

Qs = range(0, 2)

s = 5 # season length is still 24



# creating list with all the possible combinations of parameters

parameters = product(ps, qs, Ps, Qs)

parameters_list = list(parameters)

len(parameters_list)
def optimizeSARIMA(parameters_list, d, D, s):

    """Return dataframe with parameters and corresponding AIC

        

        parameters_list - list with (p, q, P, Q) tuples

        d - integration order in ARIMA model

        D - seasonal integration order 

        s - length of season

    """

    

    results = []

    best_aic = float("inf")



    for param in tqdm_notebook(parameters_list):

        # we need try-except because on some combinations model fails to converge

        try:

            model=sm.tsa.statespace.SARIMAX(cldata.ret_close_value, order=(param[0], d, param[1]), 

                                            seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)

        except:

            continue

        aic = model.aic

        # saving best model, AIC and parameters

        if aic < best_aic:

            best_model = model

            best_aic = aic

            best_param = param

        results.append([param, model.aic])



    result_table = pd.DataFrame(results)

    result_table.columns = ['parameters', 'aic']

    # sorting in ascending order, the lower AIC is - the better

    result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)

    

    return result_table
%%time

warnings.filterwarnings("ignore") 

result_table = optimizeSARIMA(parameters_list, d, D, s)
result_table.head()
# set the parameters that give the lowest AIC

p, q, P, Q = result_table.parameters[0]



best_model=sm.tsa.statespace.SARIMAX(cldata.ret_close_value, order=(p, d, q), 

                                        seasonal_order=(P, D, Q, s)).fit(disp=-1)

print(best_model.summary())
tsplot(best_model.resid[24+1:], lags=60)
def plotSARIMA(series, model, n_steps):

    """Plots model vs predicted values

        

        series - dataset with timeseries

        model - fitted SARIMA model

        n_steps - number of steps to predict in the future    

    """

    

    # adding model values

    data = series.copy()

    data.columns = ['ret_close_value']

    data['sarima_model'] = model.fittedvalues

    # making a shift on s+d steps, because these values were unobserved by the model

    # due to the differentiating

    data['sarima_model'][:s+d] = np.NaN

    

    # forecasting on n_steps forward 

    forecast = model.predict(start = data.shape[0], end = data.shape[0]+n_steps)

    forecast = data.sarima_model.append(forecast)

    # calculate error, again having shifted on s+d steps from the beginning

    error = mean_absolute_percentage_error(data['ret_close_value'][s+d:], data['sarima_model'][s+d:])



    plt.figure(figsize=(15, 7))

    plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error))

    plt.plot(forecast, color='r', label="model")

    plt.axvspan(data.index[-1], forecast.index[-1], alpha=0.5, color='lightgrey')

    plt.plot(data.ret_close_value, label="Closing")

    plt.legend()

    plt.grid(True)
plotSARIMA(cldata, best_model, 50)
ctr=  pd.cldata(cldata.iloc[-5:,:].values)

ctr.to_csv(r'/kaggle/input.csv' ,index=False)