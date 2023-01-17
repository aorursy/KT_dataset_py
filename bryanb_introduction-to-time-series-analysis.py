# Regular libraries for data manipulation

import pprint

import numpy as np

import pandas as pd

from sklearn.impute import SimpleImputer



# Visualization

from pandas.plotting import lag_plot

import matplotlib.pyplot as plt

import seaborn as sns



# Statistical tools for time series analysis

from scipy import signal

import statsmodels.api as sm

!pip install pmdarima

from pmdarima.arima import auto_arima

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.statespace.tools import diff

from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing

from statsmodels.tsa.stattools import adfuller, kpss, acf, grangercausalitytests

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf,month_plot,quarter_plot

from sklearn.metrics import mean_squared_error, mean_absolute_error



# Ignore warnings

import warnings

warnings.filterwarnings('ignore')



%matplotlib inline



sns.set_style("whitegrid")

plt.rc('xtick', labelsize=15) 

plt.rc('ytick', labelsize=15)



from pylab import rcParams
data = pd.read_csv('../input/cac40-stocks-dataset/preprocessed_CAC40.csv', usecols = ['Name','Date','Open','Closing_Price','Daily_High','Daily_Low','Volume'])
# Filter the dataframe on Air Liquide stocks

air_liquide = data[data['Name'] == 'Air Liquide'].copy()
# Converting 'Date' to datetime object

air_liquide['Date'] = pd.to_datetime(air_liquide['Date'])

air_liquide['Year'] = air_liquide['Date'].dt.year

air_liquide['Month'] = air_liquide['Date'].dt.month
print(f'air_liquide shape: {air_liquide.shape[0]} rows, {air_liquide.shape[1]} columns')
air_liquide.head()
# Line plot

fig, ax = plt.subplots(figsize=(15, 6))

sns.lineplot(air_liquide['Date'], air_liquide['Open'] )



# Formatting

ax.set_title('Open Price', fontsize = 20, loc='center', fontdict=dict(weight='bold'))

ax.set_xlabel('Year', fontsize = 16, fontdict=dict(weight='bold'))

ax.set_ylabel('Price', fontsize = 16, fontdict=dict(weight='bold'))

plt.tick_params(axis='y', which='major', labelsize=16)

plt.tick_params(axis='x', which='major', labelsize=16)
# Let's check NaN 

air_liquide.isna().sum()
# Get index of the missing value in Open column

index_open_missing = air_liquide[pd.isnull(air_liquide['Open'])].index

print(list(index_open_missing)[0])
# I replace the missing value by the Open price value of the previous day (backward fill)

air_liquide['Open'] = air_liquide['Open'].fillna(method='bfill')
print(f"Percentage of missing values in Volume: {round(sum(pd.isnull(air_liquide['Volume']))/air_liquide.shape[0],2)}\n")

print(air_liquide[air_liquide['Volume']==0])
# Imputation

imputer = SimpleImputer(strategy='constant', fill_value=0)

air_liquide_plus = imputer.fit_transform(air_liquide)



# Imputation removed column names; put them back

imputed_air_liquide = pd.DataFrame(air_liquide_plus)

imputed_air_liquide.columns = air_liquide.columns
# Replace , by . so Volume can be converted as float

imputed_air_liquide['Volume'] = imputed_air_liquide['Volume'].apply(lambda x : str(x))

imputed_air_liquide['Volume'] = pd.to_numeric(imputed_air_liquide['Volume'].apply(lambda x : x.replace(',','.',1)))



# Convert object to numeric 

imputed_air_liquide['Open'] = pd.to_numeric(imputed_air_liquide['Open'])

imputed_air_liquide['Closing_Price'] = pd.to_numeric(imputed_air_liquide['Closing_Price'])

imputed_air_liquide['Daily_High'] = pd.to_numeric(imputed_air_liquide['Daily_High'])

imputed_air_liquide['Daily_Low'] = pd.to_numeric(imputed_air_liquide['Daily_Low'])
# Plot Daily Volume Lineplot

fig, ax = plt.subplots(figsize=(15, 6))

sns.lineplot(imputed_air_liquide['Date'], imputed_air_liquide['Volume'] )



ax.set_title('Daily Volume', fontsize = 20, loc='center', fontdict=dict(weight='bold'))

ax.set_xlabel('Year', fontsize = 16, fontdict=dict(weight='bold'))

ax.set_ylabel('Price', fontsize = 16, fontdict=dict(weight='bold'))

plt.tick_params(axis='y', which='major', labelsize=16)

plt.tick_params(axis='x', which='major', labelsize=16)
# Aggregating the Time Series to a monthly scaled index

y = imputed_air_liquide[['Date','Volume']].copy()

y.set_index('Date', inplace=True)

y.index = pd.to_datetime(y.index)

y = y.resample('1M').mean()

y['Date'] = y.index



# Plot the Monthly Volume Lineplot

fig, ax = plt.subplots(figsize=(15, 6))

sns.lineplot(y['Date'], y['Volume'] )



ax.set_title('Monthly Volume', fontsize = 20, loc='center', fontdict=dict(weight='bold'))

ax.set_xlabel('Year', fontsize = 16, fontdict=dict(weight='bold'))

plt.tick_params(axis='y', which='major', labelsize=16)

plt.tick_params(axis='x', which='major', labelsize=16)
imputed_air_liquide['Year'] = imputed_air_liquide['Date'].dt.year

imputed_air_liquide['Month'] = imputed_air_liquide['Date'].dt.month
imputed_air_liquide['Year'].unique()
variable = 'Open'

fig, ax = plt.subplots(figsize=(15, 6))



sns.lineplot(imputed_air_liquide['Month'], imputed_air_liquide[variable], hue = imputed_air_liquide['Year'])

ax.set_title('Seasonal plot of Price', fontsize = 20, loc='center', fontdict=dict(weight='bold'))

ax.set_xlabel('Month', fontsize = 16, fontdict=dict(weight='bold'))

ax.set_ylabel('Price', fontsize = 16, fontdict=dict(weight='bold'))

ax.legend(labels = [str(2010+i) for i in range(11)], bbox_to_anchor=(1.1, 1.05))



fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))



sns.boxplot(imputed_air_liquide['Year'], imputed_air_liquide[variable], ax=ax[0])

ax[0].set_title('Year-wise Box Plot\n(The Trend)', fontsize = 20, loc='center', fontdict=dict(weight='bold'))

ax[0].set_xlabel('Year', fontsize = 16, fontdict=dict(weight='bold'))

ax[0].set_ylabel('Price', fontsize = 16, fontdict=dict(weight='bold'))



sns.boxplot(imputed_air_liquide['Month'], imputed_air_liquide[variable], ax=ax[1])

ax[1].set_title('Month-wise Box Plot\n(The Seasonality)', fontsize = 20, loc='center', fontdict=dict(weight='bold'))

ax[1].set_xlabel('Month', fontsize = 16, fontdict=dict(weight='bold'))

ax[1].set_ylabel('Price', fontsize = 16, fontdict=dict(weight='bold'))



fig.autofmt_xdate()
# Aggregating the Time Series to a monthly scaled index

y = imputed_air_liquide[['Date','Open','Closing_Price']].copy()

y.set_index('Date', inplace=True)

y.index = pd.to_datetime(y.index)

y = y.resample('1M').mean()



# The magic

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(16, 10))



month_plot(y['Open'], ax=ax[0]);

ax[0].set_ylabel('Open', fontsize = 16, fontdict=dict(weight='bold'))



month_plot(y['Closing_Price'], ax=ax[1]);

ax[1].set_ylabel('Closing_Price', fontsize = 16, fontdict=dict(weight='bold'))
# Aggregating the Time Series to a monthly scaled index

y = imputed_air_liquide[['Date','Open']].copy()

y.set_index('Date', inplace=True)

y.index = pd.to_datetime(y.index)

y = y.resample('1M').mean()



# Setting rcparams

rcParams['figure.figsize'] = 15, 12

rcParams['axes.labelsize'] = 20

rcParams['ytick.labelsize'] = 16

rcParams['xtick.labelsize'] = 16



# Using statistical tools of statsmodel library

decomposition = sm.tsa.seasonal_decompose(y, model='multiplicative', freq = 12)

decomp = decomposition.plot()

decomp.suptitle('Open decomposition', fontsize=22)
# check for stationarity

def adf_test(series, title=''):

    """

    Pass in a time series and an optional title, returns an ADF report

    """

    print('Augmented Dickey-Fuller Test: {}'.format(title))

    # .dropna() handles differenced data

    result = adfuller(series.dropna(),autolag='AIC') 

    

    labels = ['ADF test statistic','p-value','# lags used','# observations']

    out = pd.Series(result[0:4],index=labels)



    for key,val in result[4].items():

        out['critical value ({})'.format(key)]=val

        

    # .to_string() removes the line "dtype: float64"

    print(out.to_string())          

    

    if result[1] <= 0.05:

        print("Strong evidence against the null hypothesis")

        print("Reject the null hypothesis")

        print("Data has no unit root and is stationary")

    else:

        print("Weak evidence against the null hypothesis")

        print("Fail to reject the null hypothesis")

        print("Data has a unit root and is non-stationary")

        

# Aggregating the Time Series to a monthly scaled index

y = imputed_air_liquide[['Date','Open']].copy()

y.set_index('Date', inplace=True)

y.index = pd.to_datetime(y.index)

y = y.resample('1M').mean()

        

adf_test(y['Open'],title='') 
fig, ax = plt.subplots(nrows=2, ncols=2,figsize=(15, 11))



y['OpenDiff1'] = diff(y['Open'],k_diff=1)

y['OpenDiff2'] = diff(y['Open'],k_diff=2)

y['OpenDiff3'] = diff(y['Open'],k_diff=3)



y['Open'].plot(title="Initial Data",ax=ax[0][0]).autoscale(axis='x',tight=True);

y['OpenDiff1'].plot(title="First Difference Data",ax=ax[0][1]).autoscale(axis='x',tight=True);

y['OpenDiff2'].plot(title="Second Difference Data",ax=ax[1][0]).autoscale(axis='x',tight=True);

y['OpenDiff3'].plot(title="Third Difference Data",ax=ax[1][1]).autoscale(axis='x',tight=True);



fig.autofmt_xdate()
fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(15, 6))

autocorr = acf(imputed_air_liquide['Open'], nlags=60, fft=False)

print(autocorr)



plot_acf(imputed_air_liquide['Open'].tolist(), lags=60, ax=ax[0], fft=False);

plot_pacf(imputed_air_liquide['Open'].tolist(), lags=60, ax=ax[1]);
lag_plot(y['Open']);
# Aggregating the Time Series to a monthly scaled index

y = imputed_air_liquide[['Date','Open']].copy()

y.set_index('Date', inplace=True)

y.index = pd.to_datetime(y.index)

y = y.resample('1M').mean()



y['MA3'] = y.rolling(window=3).mean() 

y.plot(figsize=(15,6));
# Setting parameters value

span = 3

# Weights of observations

alpha = 2/(span+1)



# Plot Simple exponential smoothing

y['ES3'] = SimpleExpSmoothing(y['Open']).fit(smoothing_level = alpha, optimized = False).fittedvalues.shift(-1)

y[['Open','ES3']].plot(figsize=(15,6));
# Plot Double and Triple exponential smoothing

y['DESmul3'] = ExponentialSmoothing(y['Open'], trend = 'add').fit().fittedvalues.shift(-1)

y['TESmul3'] = ExponentialSmoothing(y['Open'], trend = 'add', seasonal = 'add', seasonal_periods = 12).fit().fittedvalues.shift(-1)

y[['Open', 'TESmul3', 'DESmul3']].plot(figsize = (15,6));
# Reverse index so the dataframe is from oldest to newest values

imputed_air_liquide = imputed_air_liquide.reindex(index=imputed_air_liquide.index[::-1])
# Split data into train and validation set 90/10

air_liquide_train, air_liquide_val = imputed_air_liquide[:int(len(imputed_air_liquide)*0.9)], imputed_air_liquide[int(len(imputed_air_liquide)*0.9):]



# Index disappeared, put them back

air_liquide_val = air_liquide_val.set_index('Date', drop=False)

air_liquide_train = air_liquide_train.set_index('Date', drop=False)



# Line plot

fig, ax = plt.subplots(figsize=(15, 6))

sns.lineplot(air_liquide_train['Date'], air_liquide_train['Open'], color = 'black')

sns.lineplot(air_liquide_val['Date'], air_liquide_val['Open'], color = 'blue')



# Formatting

ax.set_title('Open Price', fontsize = 20, loc='center', fontdict=dict(weight='bold'))

ax.set_xlabel('Year', fontsize = 16, fontdict=dict(weight='bold'))

ax.set_ylabel('Price', fontsize = 16, fontdict=dict(weight='bold'))

plt.tick_params(axis='y', which='major', labelsize=16)

plt.tick_params(axis='x', which='major', labelsize=16)

plt.legend(loc='upper right' ,labels = ('train', 'test'))
%%time

model_autoARIMA = auto_arima(air_liquide_train['Open'])



""", 

                             start_p = 0, 

                             start_q = 0,

                             test = 'adf', # use adftest to find optimal 'd'

                             max_p = 3,

                             max_q = 3, # maximum p and q

                             m = 7, # frequency of series

                             seasonal = False,

                             start_P = 0, 

                             D = 0, 

                             trace = True,

                             error_action = 'ignore',  

                             stepwise = True

"""



print(model_autoARIMA.summary())
model = ARIMA(air_liquide_train['Open'], order = (1, 1, 2))

# disp=-1: no output

fitted = model.fit(disp = -1)

print(fitted.summary())
# Forecast 260 next observations 

fc, se, conf = fitted.forecast(260, alpha=0.05)  # 95% confidence

fc_series = pd.Series(fc, index=air_liquide_val.index)

lower_series = pd.Series(conf[:, 0], index=air_liquide_val.index)

upper_series = pd.Series(conf[:, 1], index=air_liquide_val.index)



plt.figure(figsize=(12,5), dpi=100)

plt.plot(air_liquide_train['Open'], label='training')

plt.plot(air_liquide_val['Open'], color = 'blue', label='Actual Stock Price')

plt.plot(fc_series, color = 'orange',label='Predicted Stock Price')

plt.fill_between(lower_series.index, lower_series, upper_series, 

                 color='k', alpha=.10)

plt.title('Air Liquide Stock Price Prediction')

plt.xlabel('Date')

plt.ylabel('Actual Stock Price')

plt.legend(loc='upper left', fontsize=8)

plt.show()
import math
# Report performance

mse = mean_squared_error(air_liquide_val['Open'], fc)

print('MSE: '+str(mse))

mae = mean_absolute_error(air_liquide_val['Open'], fc)

print('MAE: '+str(mae))

rmse = math.sqrt(mean_squared_error(air_liquide_val['Open'], fc))

print('RMSE: '+str(rmse))

mape = np.mean(np.abs(fc - air_liquide_val['Open'])/np.abs(air_liquide_val['Open']))

print('MAPE: '+str(mape))