import pandas as pd

import matplotlib.pyplot as plt

plt.style.use('ggplot')
df = pd.read_csv('../input/daily-total-female-births-in-california-1959/daily-total-female-births-CA.csv')

df.head()
x , y = df.date , df.births

fig = plt.figure(figsize=(12,4))

plt.plot(x, y);

plt.xlabel('Date')

plt.ylabel('Number of Births')

plt.title('Daily total female births in california, 1959')

plt.show()
from statsmodels.stats.diagnostic import acorr_ljungbox

import numpy as np



# lags is suggested to be the log(len(y))

acorr_ljungbox(y, lags=6, return_df =True)

from statsmodels.tsa.statespace.tools import diff



fig = plt.figure(figsize=(12,4))

plt.plot(diff(y),color='b');

plt.xlabel('Date')

plt.ylabel('Number of Births')

plt.title('Differenced series')

plt.show()
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf



plt.figure(figsize=(16,4))

ax1 = plt.subplot(1,2,1)

plot_acf(diff(y), title='ACF of differenced time series',ax=ax1 )

plt.xlabel('Lag')

plt.ylabel('ACF')

ax2 = plt.subplot(1,2,2)

plot_pacf(diff(y), title='PACF of differenced time series',ax=ax2)

plt.xlabel('Lag')

plt.ylabel('Partial ACF')

plt.show()
from statsmodels.tsa.arima_model import ARIMA



model1 = ARIMA(y, order=(0,1,1))

model1_fit = model1.fit(disp=0)

print(model1_fit.summary())



residuals = pd.DataFrame(model1_fit.resid)



model1_sse = sum((residuals**2).values)

model1_aic = model1_fit.aic

model2 = ARIMA(y, order=(0,1,2))

model2_fit = model2.fit(disp=0)

print(model2_fit.summary())



residuals = pd.DataFrame(model2_fit.resid)



model2_sse = sum((residuals**2).values)

model2_aic = model2_fit.aic
model3 = ARIMA(y, order=(7,1,1))

model3_fit = model3.fit(disp=0)

print(model3_fit.summary())



residuals = pd.DataFrame(model3_fit.resid)



model3_sse = sum((residuals**2).values)

model3_aic = model3_fit.aic
model4 = ARIMA(y, order=(7,1,2))

model4_fit = model4.fit(disp=0)

print(model4_fit.summary())



residuals = pd.DataFrame(model4_fit.resid)



model4_sse = sum((residuals**2).values)

model4_aic = model4_fit.aic
results_df = pd.DataFrame({

    'Arima(0,1,1)': [model1_aic, "{:.3f}".format(model1_sse[0])], 

    'Arima(0,1,2)': [model2_aic, "{:.3f}".format(model2_sse[0])],

    'Arima(7,1,1)': [model3_aic, "{:.3f}".format(model3_sse[0])],

    'Arima(7,1,2)': [model4_aic, "{:.3f}".format(model4_sse[0])]

}, index=['AIC', 'SSE'])



results_df.head()
df_comp = df.copy()

df_comp.head()
df_comp.set_index('date', inplace=True)

df_comp = df_comp.asfreq('d')

df_comp.head()
# Import

from statsmodels.tsa.seasonal import seasonal_decompose# Decompose time series into daily trend, seasonal, and residual components.



decomp = seasonal_decompose(df_comp, model = 'additive')# Plot the decomposed time series to interpret.



fig, (ax0, ax1,ax2,ax3) = plt.subplots(4,1, figsize=(15,8));

decomp.observed.plot(ax=ax0, title='Our time series');

decomp.trend.plot(ax=ax1, title='Trend');

decomp.resid.plot(ax=ax2, title='Residual');

decomp.seasonal.plot(ax=ax3, title='Seasonality');
decomp = seasonal_decompose(df_comp, model = 'multiplicative')# Plot the decomposed time series to interpret.



fig, (ax0, ax1,ax2,ax3) = plt.subplots(4,1, figsize=(15,8));

decomp.observed.plot(ax=ax0, title='Our time series');

decomp.trend.plot(ax=ax1, title='Trend');

decomp.resid.plot(ax=ax2, title='Residual');

decomp.seasonal.plot(ax=ax3, title='Seasonality');
import itertools



q = [1,2]

p = [1,7]

d = [0,1]

pdq = list(itertools.product(p, d, q))

seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Examples of parameter for SARIMA...')

print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))

print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))

print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))

print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
from statsmodels.tsa.statespace.sarimax import SARIMAX



rest_dict = {}



for param in pdq:

    for param_seasonal in seasonal_pdq:

        try:

            mod = SARIMAX(diff(y),order=param,seasonal_order=param_seasonal)

            results = mod.fit(maxiter=100, method='powell')

#             print('ARIMA{}x{}12 - AIC:{}'.format(param,param_seasonal,results.aic)) 

            rest_dict[param] = {param_seasonal: results.aic}

        except: 

            continue

print(rest_dict)
mod =  SARIMAX(diff(y),

               order=(1, 0, 1),

               seasonal_order=(7, 1, 2, 12))



results = mod.fit(maxiter=100, method='powell')

print(results.summary().tables[1])
results.plot_diagnostics(figsize=(18, 8))

plt.show()