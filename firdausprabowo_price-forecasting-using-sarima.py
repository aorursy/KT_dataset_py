# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import matplotlib.pyplot as plt



print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
calendar = pd.read_csv('../input/calendar.csv')

calendar.head()
calendar['date']=pd.to_datetime(calendar['date'])

calendar.head()
calendar_available = calendar[calendar['available']=='t']

calendar_available['price_cleaned'] = calendar_available['price'].str.replace('$','').apply(pd.to_numeric, errors='coerce')

calendar_available.index = calendar_available['date']

calendar_available = calendar_available.drop(columns=['date','available','price'])
mean_price = calendar_available.groupby(calendar_available.index).mean().drop(columns='listing_id')

mean_price.plot()

ax = plt.ylabel('mean price ($)')
from dateutil.relativedelta import relativedelta

import statsmodels.api as sm  

from statsmodels.tsa.stattools import acf  

from statsmodels.tsa.stattools import pacf

from statsmodels.tsa.seasonal import seasonal_decompose



decomposition = seasonal_decompose(mean_price) 

decomposition.plot()

plt.show()
from statsmodels.tsa.stattools import adfuller



def test_stationarity(timeseries):

    

    #Determing rolling statistics

    rolmean = timeseries.rolling(window=7).mean()

    rolstd = timeseries.rolling(window=7).std()



    #Plot rolling statistics:

    fig = plt.figure(figsize=(12, 8))

    orig = plt.plot(timeseries, color='blue',label='Original')

    mean = plt.plot(rolmean, color='red', label='Rolling Mean')

    std = plt.plot(rolstd, color='black', label = 'Rolling Std')

    plt.legend(loc='best')

    plt.title('Rolling Mean & Standard Deviation')

    plt.show()

    

    #Perform Dickey-Fuller test:

    print('Results of Dickey-Fuller Test:')

    dftest = adfuller(timeseries, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    print(dfoutput)
test_stationarity(mean_price.price_cleaned)
price_diff = mean_price.diff().dropna()

test_stationarity(price_diff.price_cleaned)
import itertools

import warnings



def parameter_search_sarimax(timeseries,s):

    lowest_aic = None

    lowest_parm = None

    lowest_param_seasonal = None

    

    p = D = q = range(0, 3)

    d = [1, 2]

    pdq = list(itertools.product(p, d, q))

    seasonal_pdq = [(x[0], x[1], x[2], s) for x in list(itertools.product(p, D, q))]



    warnings.filterwarnings('ignore') # specify to ignore warning messages



    for param in pdq:

        for param_seasonal in seasonal_pdq:

            try:

                mod = sm.tsa.statespace.SARIMAX(timeseries,

                                                order=param,

                                                seasonal_order=param_seasonal,

                                                enforce_stationarity=False,

                                                enforce_invertibility=False)



                results = mod.fit()

                

                # Store results

                current_aic = results.aic

                # Set baseline for aic

                if (lowest_aic == None):

                    lowest_aic = results.aic

                # Compare results

                if (current_aic <= lowest_aic):

                    lowest_aic = current_aic

                    lowest_parm = param

                    lowest_param_seasonal = param_seasonal



                print('SARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))

            except:

                continue

            

    print('The best model is: SARIMA{}x{} - AIC:{}'.format(lowest_parm, lowest_param_seasonal, lowest_aic))
parameter_search_sarimax(mean_price,12)
def model_run_validate(parameter):



    mod = sm.tsa.statespace.SARIMAX(mean_price,

                                    order=(parameter[0], parameter[1], parameter[2]),

                                    seasonal_order=(parameter[3], parameter[4], parameter[5], parameter[6]),

                                    enforce_stationarity=False,

                                    enforce_invertibility=False)



    results = mod.fit(maxiter=200)



    print(results.summary().tables[1])



    results.plot_diagnostics(figsize=(15, 12))



    pred_start_date = '2016-12-20'

    pred_dynamic = results.get_prediction(start=pd.to_datetime(pred_start_date), dynamic=True, full_results=True)

    pred_dynamic_ci = pred_dynamic.conf_int()



    ax = mean_price.plot(label='Observed', figsize=(20, 15))

    pred_dynamic.predicted_mean.plot(label='Dynamic Forecast', ax=ax)



    ax.fill_between(pred_dynamic_ci.index,

                    pred_dynamic_ci.iloc[:, 0],

                    pred_dynamic_ci.iloc[:, 1], color='k', alpha=.25)



    ax.fill_betweenx(ax.get_ylim(), pd.to_datetime(pred_start_date), mean_price.index[-1],

                     alpha=.1, zorder=-1)



    ax.set_xlabel('Date')

    ax.set_ylabel('Mean Price ($)')

    ax.legend()
parameter = [2,1,2,2,1,1,12]

model_run_validate(parameter)
parameter_search_sarimax(mean_price,7)
parameter = [0,1,2,2,1,2,7]

model_run_validate(parameter)
parameter = [2,1,1,2,1,2,7]

model_run_validate(parameter)
def run_model_forecast(parameter):

    

    mod = sm.tsa.statespace.SARIMAX(mean_price,

                                    order=(parameter[0], parameter[1], parameter[2]),

                                    seasonal_order=(parameter[3], parameter[4], parameter[5], parameter[6]),

                                    enforce_stationarity=False,

                                    enforce_invertibility=False)



    results = mod.fit(maxiter=200)

    

    pred_uc = results.get_forecast(steps=60)

    pred_ci = pred_uc.conf_int()

    

    ax = mean_price.price_cleaned.plot(label='Observed', figsize=(20, 15))

    pred_uc.predicted_mean.plot(ax=ax, label='Forecast')

    ax.fill_between(pred_ci.index,

                    pred_ci.iloc[:, 0],

                    pred_ci.iloc[:, 1], color='k', alpha=.25)

    ax.set_xlabel('Date')

    ax.set_ylabel('Mean Price ($)')

    ax.legend()
parameter = [2,1,1,2,1,2,7]

run_model_forecast(parameter)
parameter = [0,1,2,2,1,2,7]

run_model_forecast(parameter)
parameter = [2,1,2,1,1,2,28]

model_run_validate(parameter)
run_model_forecast(parameter)
parameter = [2,1,0,2,2,0,84]

model_run_validate(parameter)
run_model_forecast(parameter)