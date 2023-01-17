import pandas as pd 

import matplotlib.pyplot as plt

from datetime import date

#loading important libraries

import numpy as np 

%matplotlib inline

from statsmodels.graphics.tsaplots import plot_acf

import statsmodels.api as sm

import warnings

import itertools

import numpy as np

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

plt.style.use('fivethirtyeight')

import pandas as pd

import statsmodels.api as sm

import matplotlib

import pandas as pd

data = pd.read_csv("../input/tesla-data/TSLA.csv")





data.shape

from statsmodels.graphics import tsaplots

plt.style.use('fivethirtyeight')

from sklearn.metrics import mean_squared_error



#preprocessing



fig = tsaplots.plot_acf(data["Close"], lags=24)



# Show plot

plt.show()











data["Close"].plot()
data["Low"].plot()
data["High"].plot()
#define function for ADF test

from statsmodels.tsa.stattools import adfuller

def adf_test(timeseries):

    #Perform Dickey-Fuller test:

    print ('Results of Dickey-Fuller Test:')

    dftest = adfuller(timeseries, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

       dfoutput['Critical Value (%s)'%key] = value

    print (dfoutput)



#apply adf test on the series

adf_test(data["Close"])

data["Close_log"] = np.log(data["Close"])

data["Close_log"] = data["Close_log"] - data["Close_log"].shift(1)

data = data[["Date","Close_log"]].set_index('Date')

data.head()

train = data.Close_log[1:1812]

test = data.Close_log[1812:2416]





from statsmodels.tsa.arima_model import ARIMA

from pandas import DataFrame
model = ARIMA(train, order=(5,1,0))

model_fit = model.fit(disp=0)

print(model_fit.summary())

# plot residual errors

residuals = DataFrame(model_fit.resid)

residuals.plot()

plt.show()

residuals.plot(kind='kde')

plt.show()

print(residuals.describe())
test.head()




# Forecast

fc, se, conf = model_fit.forecast(604, alpha=0.05)  # 95% conf



# Make as pandas series

fc_series = pd.Series(fc, index=test.index)

lower_series = pd.Series(conf[:, 0], index=test.index)

upper_series = pd.Series(conf[:, 1], index=test.index)



# Plot

plt.figure(figsize=(12,5), dpi=100)

plt.plot(train, label='training')

plt.plot(test, label='actual')

plt.plot(fc_series, label='forecast')

plt.fill_between(lower_series.index, lower_series, upper_series, 

                 color='k', alpha=.15)

plt.title('Forecast vs Actuals')

plt.legend(loc='upper left', fontsize=8)

plt.show()

def forecast_accuracy(forecast, actual):

    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE

    me = np.mean(forecast - actual)             # ME

    mae = np.mean(np.abs(forecast - actual))    # MAE

    mpe = np.mean((forecast - actual)/actual)   # MPE

    rmse = np.mean((forecast - actual)**2)**.5  # RMSE

    

   

    return({'mape':mape, 'me':me, 'mae': mae, 

            'mpe': mpe, 'rmse':rmse})



forecast_accuracy(fc, test.values)







