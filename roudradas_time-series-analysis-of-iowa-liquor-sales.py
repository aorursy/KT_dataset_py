# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

import itertools

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

plt.style.use('fivethirtyeight')

import statsmodels.api as sm

import matplotlib

import datetime

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
## Data Loading

file_path = '../input/iowa-liquor-sales/Iowa_Liquor_Sales.csv'

df = pd.read_csv(file_path)
print(df.shape)

print(df.columns)

df.head()
df['Sale (Dollars)'] = df['Sale (Dollars)'].str.replace('$', '')

df['Sale (Dollars)'] = df['Sale (Dollars)'].astype('float')

df['Date'] = pd.to_datetime(df['Date'])

df['Date'].head(), df['Sale (Dollars)'].head()
## Data Preprocessing

JBB = df.loc[df['Vendor Name'] == 'Jim Beam Brands']

cols = ['Invoice/Item Number', 'Store Number', 'Store Name', 'Address', 'City', 'Zip Code',

       'Store Location', 'County Number', 'County', 'Category', 'Category Name', 'Vendor Number',

       'Vendor Name', 'Item Number', 'Item Description', 'Pack', 'Bottle Volume (ml)', 'State Bottle Cost',

        'State Bottle Retail', 'Bottles Sold', 'Volume Sold (Liters)', 'Volume Sold (Gallons)']

JBB.drop(cols, axis = 1, inplace = True)

JBB.isnull().sum()



JBB = JBB.groupby('Date')['Sale (Dollars)'].sum().reset_index()

JBB = JBB.set_index('Date')

JBB.index

## Time Series Analysis

JBB.head()

y = JBB['Sale (Dollars)'].resample('MS').mean()

y['2016':].describe()

y.plot()
from pylab import rcParams

rcParams['figure.figsize'] = 18,8

decomposition = sm.tsa.seasonal_decompose(y, model = 'additive')

fig = decomposition.plot()

# Parameter Tuning

p = d = q = range(0, 2)

pdq = list(itertools.product(p, d, q))

seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]





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

# Fitting Model

mod = sm.tsa.statespace.SARIMAX(y,

                                order = (0,1,1),

                                seasonal_order = (0,1,1,12),

                                enforce_stationarity = False,

                                enforce_invertibility = False)



results = mod.fit()

print(results.summary().tables[1])



results.plot_diagnostics(figsize = (16,8))
# Validation

pred = results.get_prediction(start=pd.to_datetime('2016-11-01'), dynamic=False)

pred_ci = pred.conf_int()

ax = y['2014':].plot(label='observed')

pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))

ax.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color='k', alpha=.2)





ax.set_xlabel('Date')

ax.set_ylabel('Sales')

plt.legend()
y_forecasted = pred.predicted_mean

y_truth = y['2014-01-01':]

mse = ((y_forecasted - y_truth) ** 2).mean()

print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))
# Forecasting

pred_uc = results.get_forecast(steps=30)

pred_ci = pred_uc.conf_int()

ax = y.plot(label='observed', figsize=(14, 7))

pred_uc.predicted_mean.plot(ax=ax, label='Forecast')

ax.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color='k', alpha=.25)



ax.set_xlabel('Date')

ax.set_ylabel('Sales ($)')

ax.set_title('Jim Beam')

plt.legend()
## Comparing Vendor/Brands

df['Vendor Name'].drop_duplicates().values

WDL = df.loc[df['Vendor Name'] == 'Wilson Daniels Ltd.']

JBB = df.loc[df['Vendor Name'] == 'Jim Beam Brands']



JBB.drop(cols, axis = 1, inplace = True)

WDL.drop(cols, axis = 1, inplace = True)

JBB.isnull().sum()

WDL.isnull().sum()



JBB = JBB.groupby('Date')['Sale (Dollars)'].sum().reset_index()

WDL = WDL.groupby('Date')['Sale (Dollars)'].sum().reset_index()

JBB = JBB.set_index('Date')

WDL = WDL.set_index('Date')

JBB.index

WDL.index
y_JBB = JBB['Sale (Dollars)'].resample('MS').mean()

y_WDL = WDL['Sale (Dollars)'].resample('MS').mean()

JBB = pd.DataFrame({'Date': y_JBB.index, 'Sale (Dollars)': y_JBB.values})

WDL = pd.DataFrame({'Date': y_WDL.index, 'Sale (Dollars)': y_WDL.values})



vendor = JBB.merge(WDL, how = 'inner', on = 'Date')

vendor.rename(columns = {'Sale (Dollars)_x': 'Jim Beam Sales', 'Sale (Dollars)_y': 'Wilson Daniels Ltd Sales'}, inplace = True)

vendor.head()
plt.figure(figsize=(20,8))

plt.plot(vendor['Date'], vendor['Jim Beam Sales'], 'b-', label = 'Jim Beam')

plt.plot(vendor['Date'], vendor['Wilson Daniels Ltd Sales'], 'r-', label = 'Wilson Daniels')

plt.xlabel('Date')

plt.ylabel('Sales')

plt.title('Sales of Jim Beam and Wilson Daniels')

plt.legend()
## Forecasting with Prophet

from fbprophet import Prophet



JBB = JBB.rename(columns = {'Date': 'ds', 'Sale (Dollars)': 'y'})

JBB_model = Prophet(interval_width = 0.95)

JBB_model.fit(JBB)



WDL = WDL.rename(columns = {'Date': 'ds', 'Sale (Dollars)': 'y'})

WDL_model = Prophet(interval_width = 0.95)

WDL_model.fit(WDL)



JBB_forecast = JBB_model.make_future_dataframe(periods=36, freq = 'MS')

JBB_forecast = JBB_model.predict(JBB_forecast)



WDL_forecast = WDL_model.make_future_dataframe(periods = 36, freq = 'MS')

WDL_forecast = WDL_model.predict(WDL_forecast)
plt.figure(figsize=(18,6))

JBB_model.plot(JBB_forecast, xlabel = 'Date', ylabel = 'Sales')

plt.title('Jim Beam Sales with Forecasts')
plt.figure(figsize=(18,6))

WDL_model.plot(WDL_forecast, xlabel = 'Date', ylabel = 'Sales')

plt.title('Wilson Daniels Sales with Forecasts')
JBB_names = ['JBB_%s' % column for column in JBB_forecast.columns]

WDL_names = ['WDL_%s' % column for column in WDL_forecast.columns]



merge_JBB_forecast = JBB_forecast.copy()

merge_WDL_forecast = WDL_forecast.copy()

merge_JBB_forecast.columns = JBB_names

merge_WDL_forecast.columns = WDL_names



forecast = pd.merge(merge_JBB_forecast, merge_WDL_forecast, how = 'inner', left_on = 'JBB_ds', right_on = 'WDL_ds')

forecast = forecast.rename(columns = {'JBB_ds': 'Date'}).drop('WDL_ds', axis = 1)

forecast.head()
plt.figure(figsize = (10,7))

plt.plot(forecast['Date'], forecast['JBB_trend'], 'b-')

plt.plot(forecast['Date'], forecast['WDL_trend'], 'r-')

plt.xlabel('Date')

plt.ylabel('Sales')

plt.title('Jim Beam vs Wilson Daniels Trend')

plt.legend()
plt.figure(figsize=(10,7))

plt.plot(forecast['Date'], forecast['JBB_yhat'], 'b-')

plt.plot(forecast['Date'], forecast['WDL_yhat'], 'r-')

plt.xlabel('Date')

plt.ylabel('Sales')

plt.title('Jim Beam vs Wilson Daniels Estimate')

plt.legend()
df1 =df.copy()

df1.drop(cols, axis = 1, inplace = True)

df1 = df1.groupby('Date')['Sale (Dollars)'].sum().reset_index()

df1 = df1.set_index('Date')

y = df1['Sale (Dollars)'].resample('MS').mean()



df1 = pd.DataFrame({'Date': y.index, 'Sale (Dollars)': y.values})

df1 = df1.rename(columns ={'Date': 'ds', 'Sale (Dollars)': 'y'} )

df1_model = Prophet(interval_width = 0.95)

df1_model.fit(df1)



df1_forecast = df1_model.make_future_dataframe(periods = 36, freq = 'MS')

df1_forecast = df1_model.predict(df1_forecast)
plt.figure(figsize = (18,6))

df1_model.plot(df1_forecast, xlabel = 'Date', ylabel = 'Sales ($)')

plt.title('Liquor Market')
