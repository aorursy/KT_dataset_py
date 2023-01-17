%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pylab

import matplotlib.pyplot as plt

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 20, 16
import warnings

import itertools

warnings.filterwarnings("ignore") # specify to ignore warning messages
df = pd.read_csv("../input/MER_T12_06.csv")

df.head()
df.info()
dateparse = lambda x: pd.to_datetime(x, format='%Y%m', errors = 'coerce')

df = pd.read_csv("../input/MER_T12_06.csv", parse_dates=['YYYYMM'], index_col='YYYYMM', date_parser=dateparse) 

df.head()
df.head(15)
ts = df[pd.Series(pd.to_datetime(df.index, errors='coerce')).notnull().values]

ts.head(15)
ts.dtypes
#ss = ts.copy(deep=True)

ts['Value'] = pd.to_numeric(ts['Value'] , errors='coerce')

ts.head()
ts.info()
ts.dropna(inplace = True)
Energy_sources = ts.groupby('Description')

Energy_sources.head()
fig, ax = plt.subplots()

for desc, group in Energy_sources:

    group.plot(x = group.index, y='Value', label=desc,ax = ax, title='Carbon Emissions per Energy Source', fontsize = 20)

    ax.set_xlabel('Time(Monthly)')

    ax.set_ylabel('Carbon Emissions in MMT')

    ax.xaxis.label.set_size(20)

    ax.yaxis.label.set_size(20)

    ax.legend(fontsize = 16)
fig, axes = plt.subplots(3,3, figsize = (30, 20))

for (desc, group), ax in zip(Energy_sources, axes.flatten()):

    group.plot(x = group.index, y='Value',ax = ax, title=desc, fontsize = 18)

    ax.set_xlabel('Time(Monthly)')

    ax.set_ylabel('Carbon Emissions in MMT')

    ax.xaxis.label.set_size(18)

    ax.yaxis.label.set_size(18)
CO2_per_source = ts.groupby('Description')['Value'].sum().sort_values()
# I want to use shorter descriptions for the energy sources

CO2_per_source.index
cols = ['Geothermal Energy', 'Non-Biomass Waste', 'Petroleum Coke','Distillate Fuel ',

        'Residual Fuel Oil', 'Petroleum', 'Natural Gas', 'Coal', 'Total Emissions']
fig = plt.figure(figsize = (16,9))

x_label = cols

x_tick = np.arange(len(cols))

plt.bar(x_tick, CO2_per_source, align = 'center', alpha = 0.5)

fig.suptitle("CO2 Emissions by Electric Power Sector", fontsize= 25)

plt.xticks(x_tick, x_label, rotation = 70, fontsize = 20)

plt.yticks(fontsize = 20)

plt.xlabel('Carbon Emissions in MMT', fontsize = 20)

plt.show()
Emissions = ts.iloc[:,1:]   # Monthly total emissions (mte)

Emissions= Emissions.groupby(['Description', pd.TimeGrouper('M')])['Value'].sum().unstack(level = 0)

mte = Emissions['Natural Gas Electric Power Sector CO2 Emissions'] # monthly total emissions (mte)

mte.head()
mte.tail()
import statsmodels

import statsmodels.api as sm

from statsmodels.tsa.stattools import coint, adfuller
plt.plot(mte)
def TestStationaryPlot(ts):

    rol_mean = ts.rolling(window = 12, center = False).mean()

    rol_std = ts.rolling(window = 12, center = False).std()

    

    plt.plot(ts, color = 'blue',label = 'Original Data')

    plt.plot(rol_mean, color = 'red', label = 'Rolling Mean')

    plt.plot(rol_std, color ='black', label = 'Rolling Std')

    plt.xticks(fontsize = 25)

    plt.yticks(fontsize = 25)

    

    plt.xlabel('Time in Years', fontsize = 25)

    plt.ylabel('Total Emissions', fontsize = 25)

    plt.legend(loc='best', fontsize = 25)

    plt.title('Rolling Mean & Standard Deviation', fontsize = 25)

    plt.show(block= True)
def TestStationaryAdfuller(ts, cutoff = 0.01):

    ts_test = adfuller(ts, autolag = 'AIC')

    ts_test_output = pd.Series(ts_test[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    

    for key,value in ts_test[4].items():

        ts_test_output['Critical Value (%s)'%key] = value

    print(ts_test_output)

    

    if ts_test[1] <= cutoff:

        print("Strong evidence against the null hypothesis, reject the null hypothesis. Data has no unit root, hence it is stationary")

    else:

        print("Weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")

        
TestStationaryPlot(mte)
TestStationaryAdfuller(mte)
moving_avg = mte.rolling(12).mean()

plt.plot(mte)

plt.plot(moving_avg, color='red')

plt.xticks(fontsize = 25)

plt.yticks(fontsize = 25)

plt.xlabel('Time (years)', fontsize = 25)

plt.ylabel('CO2 Emission (MMT)', fontsize = 25)

plt.title('CO2 emission from electric power generation', fontsize = 25)

plt.show()
mte_moving_avg_diff = mte - moving_avg

mte_moving_avg_diff.head(13)
mte_moving_avg_diff.dropna(inplace=True)

TestStationaryPlot(mte_moving_avg_diff)
TestStationaryAdfuller(mte_moving_avg_diff)
mte_exp_wighted_avg = pd.ewma(mte, halflife=12)

plt.plot(mte)

plt.plot(mte_exp_wighted_avg, color='red')

plt.xticks(fontsize = 25)

plt.yticks(fontsize = 25)

plt.xlabel('Time (years)', fontsize = 25)

plt.ylabel('CO2 Emission (MMT)', fontsize = 25)

plt.title('CO2 emission from electric power generation', fontsize = 25)

plt.show()
mte_ewma_diff = mte - mte_exp_wighted_avg

TestStationaryPlot(mte_ewma_diff)
TestStationaryAdfuller(mte_ewma_diff)
mte_first_difference = mte - mte.shift(1)  

TestStationaryPlot(mte_first_difference.dropna(inplace=False))
TestStationaryAdfuller(mte_first_difference.dropna(inplace=False))
mte_seasonal_difference = mte - mte.shift(12)  

TestStationaryPlot(mte_seasonal_difference.dropna(inplace=False))

TestStationaryAdfuller(mte_seasonal_difference.dropna(inplace=False))
mte_seasonal_first_difference = mte_first_difference - mte_first_difference.shift(12)  

TestStationaryPlot(mte_seasonal_first_difference.dropna(inplace=False))
TestStationaryAdfuller(mte_seasonal_first_difference.dropna(inplace=False))
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(mte)



trend = decomposition.trend

seasonal = decomposition.seasonal

residual = decomposition.resid



plt.subplot(411)

plt.plot(mte, label='Original')

plt.legend(loc='best')

plt.subplot(412)

plt.plot(trend, label='Trend')

plt.legend(loc='best')

plt.subplot(413)

plt.plot(seasonal,label='Seasonality')

plt.legend(loc='best')

plt.subplot(414)

plt.plot(residual, label='Residuals')

plt.legend(loc='best')

plt.tight_layout()
mte_decompose = residual

mte_decompose.dropna(inplace=True)

TestStationaryPlot(mte_decompose)

TestStationaryAdfuller(mte_decompose)
fig = plt.figure(figsize=(12,8))

ax1 = fig.add_subplot(211)

fig = sm.graphics.tsa.plot_acf(mte_seasonal_first_difference.iloc[13:], lags=40, ax=ax1)

ax2 = fig.add_subplot(212)

fig = sm.graphics.tsa.plot_pacf(mte_seasonal_first_difference.iloc[13:], lags=40, ax=ax2)
p = d = q = range(0, 2) # Define the p, d and q parameters to take any value between 0 and 2

pdq = list(itertools.product(p, d, q)) # Generate all different combinations of p, q and q triplets

pdq_x_QDQs = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))] # Generate all different combinations of seasonal p, q and q triplets

print('Examples of Seasonal ARIMA parameter combinations for Seasonal ARIMA...')

print('SARIMAX: {} x {}'.format(pdq[1], pdq_x_QDQs[1]))

print('SARIMAX: {} x {}'.format(pdq[2], pdq_x_QDQs[2]))
for param in pdq:

    for seasonal_param in pdq_x_QDQs:

        try:

            mod = sm.tsa.statespace.SARIMAX(mte,

                                            order=param,

                                            seasonal_order=seasonal_param,

                                            enforce_stationarity=False,

                                            enforce_invertibility=False)

            results = mod.fit()

            print('ARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))

        except:

            continue

mod = sm.tsa.statespace.SARIMAX(mte, 

                                order=(1,1,1), 

                                seasonal_order=(0,1,1,12),   

                                enforce_stationarity=False,

                                enforce_invertibility=False)

results = mod.fit()

print(results.summary())
results.resid.plot()
print(results.resid.describe())
results.resid.plot(kind='kde')
results.plot_diagnostics(figsize=(15, 12))

plt.show()
pred = results.get_prediction(start = 480, end = 523, dynamic=False)

pred_ci = pred.conf_int()

pred_ci.head()
ax = mte['1973':].plot(label='observed')

pred.predicted_mean.plot(ax=ax, label='One-step ahead forecast', alpha=.7)



ax.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color='r', alpha=.5)



ax.set_xlabel('Time (years)')

ax.set_ylabel('NG CO2 Emissions')

plt.legend()



plt.show()
mte_forecast = pred.predicted_mean

mte_truth = mte['2013-01-31':]



# Compute the mean square error

mse = ((mte_forecast - mte_truth) ** 2).mean()

print('The Mean Squared Error (MSE) of the forecast is {}'.format(round(mse, 2)))

print('The Root Mean Square Error (RMSE) of the forcast: {:.4f}'

      .format(np.sqrt(sum((mte_forecast-mte_truth)**2)/len(mte_forecast))))
mte_pred_concat = pd.concat([mte_truth, mte_forecast])
pred_dynamic = results.get_prediction(start=pd.to_datetime('2013-01-31'), dynamic=True, full_results=True)

pred_dynamic_ci = pred_dynamic.conf_int()
ax = mte['1973':].plot(label='observed', figsize=(20, 15))

pred_dynamic.predicted_mean.plot(label='Dynamic Forecast', ax=ax)



ax.fill_between(pred_dynamic_ci.index,

                pred_dynamic_ci.iloc[:, 0],

                pred_dynamic_ci.iloc[:, 1], 

                color='r', 

                alpha=.3)



ax.fill_betweenx(ax.get_ylim(), 

                 pd.to_datetime('2013-01-31'), 

                 mte.index[-1],

                 alpha=.1, zorder=-1)



ax.set_xlabel('Time (years)')

ax.set_ylabel('CO2 Emissions')



plt.legend()

plt.show()
# Extract the predicted and true values of our time series

mte_forecast = pred_dynamic.predicted_mean

mte_orginal = mte['2013-01-31':]



# Compute the mean square error

mse = ((mte_forecast - mte_orginal) ** 2).mean()

print('The Mean Squared Error (MSE) of the forecast is {}'.format(round(mse, 2)))

print('The Root Mean Square Error (RMSE) of the forcast: {:.4f}'

      .format(np.sqrt(sum((mte_forecast-mte_orginal)**2)/len(mte_forecast))))
# Get forecast of 10 years or 120 months steps ahead in future

forecast = results.get_forecast(steps= 120)

# Get confidence intervals of forecasts

forecast_ci = forecast.conf_int()

forecast_ci.head()
ax = mte.plot(label='observed', figsize=(20, 15))

forecast.predicted_mean.plot(ax=ax, label='Forecast')

ax.fill_between(forecast_ci.index,

                forecast_ci.iloc[:, 0],

                forecast_ci.iloc[:, 1], color='g', alpha=.4)

ax.set_xlabel('Time (year)')

ax.set_ylabel('NG CO2 Emission level')



plt.legend()

plt.show()