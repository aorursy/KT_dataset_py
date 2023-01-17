import pandas as pd
trade = pd.read_csv('../input/tradein/TradeInventories.csv',index_col=0,parse_dates=True)

trade.index.freq='MS'
trade.head()
%matplotlib inline
trade['Inventories'].plot(figsize=(12,5))
from statsmodels.tsa.statespace.tools import diff
trade['d1'] = diff(trade['Inventories'],k_diff=1)
trade['d1'].plot(figsize=(12,5))
from statsmodels.tsa.stattools import adfuller



def adf_test(series,title=''):

    """

    Pass in a time series and an optional title, returns an ADF report

    """

    print(f'Augmented Dickey-Fuller Test: {title}')

    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data

    

    labels = ['ADF test statistic','p-value','# lags used','# observations']

    out = pd.Series(result[0:4],index=labels)



    for key,val in result[4].items():

        out[f'critical value ({key})']=val

        

    print(out.to_string())          # .to_string() removes the line "dtype: float64"

    

    if result[1] <= 0.05:

        print("Strong evidence against the null hypothesis")

        print("Reject the null hypothesis")

        print("Data has no unit root and is stationary")

    else:

        print("Weak evidence against the null hypothesis")

        print("Fail to reject the null hypothesis")

        print("Data has a unit root and is non-stationary")
adf_test(trade['Inventories'],title='DF for orignal data')
adf_test(trade['d1'],title='DF for 1 diff data')
import numpy as np

from statsmodels.tsa.arima_model import ARMA,ARMAResults,ARIMA,ARIMAResults

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

import warnings

warnings.filterwarnings("ignore")
df2 = pd.read_csv('../input/tradein/TradeInventories.csv',index_col=0,parse_dates=True)

df2.index.freq='MS'
import matplotlib.ticker as ticker

formatter = ticker.StrMethodFormatter('{x:,.0f}')



title = 'Real Manufacturing and Trade Inventories'

ylabel='Chained 2012 Dollars'

xlabel='' # we don't really need a label here



ax = df2['Inventories'].plot(figsize=(12,5),title=title)

ax.autoscale(axis='x',tight=True)

ax.set(xlabel=xlabel, ylabel=ylabel)

ax.yaxis.set_major_formatter(formatter);
from statsmodels.tsa.statespace.tools import diff

df2['d1'] = diff(df2['Inventories'],k_diff=1)



adf_test(df2['d1'],'Real Manufacturing and Trade Inventories')
title = 'Autocorrelation: Real Manufacturing and Trade Inventories'

lags = 40

plot_acf(df2['Inventories'],title=title,lags=lags);
title = 'Partial Autocorrelation: Real Manufacturing and Trade Inventories'

lags = 40

plot_pacf(df2['Inventories'],title=title,lags=lags);
len(df2)
# Set one year for testing

train = df2.iloc[:252]

test = df2.iloc[252:]
model = ARIMA(train['Inventories'],order=(1,1,0))

results = model.fit()

results.summary() 
model = ARIMA(train['Inventories'],order=(1,1,1))

results = model.fit()

results.summary()
start=len(train)

end=len(train)+len(test)-1

predictions = results.predict(start=start, end=end, dynamic=False, typ='levels').rename('ARIMA(1,1,1) Predictions')
for i in range(len(predictions)):

    print(f"predicted={predictions[i]:<11.10}, expected={test['Inventories'][i]}")
title = 'Real Manufacturing and Trade Inventories'

ylabel='Chained 2012 Dollars'

xlabel=''



ax = test['Inventories'].plot(legend=True,figsize=(12,6),title=title)

predictions.plot(legend=True)

ax.autoscale(axis='x',tight=True)

ax.set(xlabel=xlabel, ylabel=ylabel)

ax.yaxis.set_major_formatter(formatter);
from statsmodels.tools.eval_measures import rmse



error = rmse(test['Inventories'], predictions)

print(f'ARIMA(1,1,1) RMSE Error: {error:11.10}')
model = ARIMA(df2['Inventories'],order=(1,1,1))

results = model.fit()

fcast = results.predict(len(df2),len(df2)+11,typ='levels').rename('ARIMA(1,1,1) Forecast')
title = 'Real Manufacturing and Trade Inventories'

ylabel='Chained 2012 Dollars'

xlabel=''



ax = df2['Inventories'].plot(legend=True,figsize=(12,6),title=title)

fcast.plot(legend=True)

ax.autoscale(axis='x',tight=True)

ax.set(xlabel=xlabel, ylabel=ylabel)

ax.yaxis.set_major_formatter(formatter);
from statsmodels.tsa.seasonal import seasonal_decompose



result = seasonal_decompose(df2['Inventories'], model='additive')  # model='add' also works

result.plot();
from statsmodels.graphics.tsaplots import month_plot,quarter_plot
df = pd.read_csv('../input/airline1/airline_passengers.csv',index_col=0, parse_dates=True)

df.index.freq='MS'
df.head()
month_plot(df['Thousands of Passengers']);
# !!pip install pyramid-arima

from pmdrima import auto_arima
stepwise_fit = auto_arima(df2['Inventories'], start_p=0, start_q=0,

                          max_p=2, max_q=2, m=12,

                          seasonal=False,

                          d=None, trace=True,

                          error_action='ignore',   # we don't want to know if an order does not work

                          suppress_warnings=True,  # we don't want convergence warnings

                          stepwise=True)           # set to stepwise



stepwise_fit.summary()
stepwise_fit = auto_arima(df2['Inventories'], start_p=0, start_q=0,

                          max_p=2, max_q=2, m=4,

                          seasonal=True,

                          d=None, trace=True,

                          error_action='ignore',   # we don't want to know if an order does not work

                          suppress_warnings=True,  # we don't want convergence warnings

                          stepwise=True)           # set to stepwise



stepwise_fit.summary()
stepwise_fit = auto_arima(df['Thousands of Passengers'], start_p=0, start_q=0,

                          max_p=2, max_q=2, m=12,

                          seasonal=True,

                          d=None, trace=True,

                          error_action='ignore',   # we don't want to know if an order does not work

                          suppress_warnings=True,  # we don't want convergence warnings

                          stepwise=True)           # set to stepwise



stepwise_fit.summary()