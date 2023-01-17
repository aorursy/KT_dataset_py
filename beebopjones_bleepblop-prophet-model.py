

# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import os
import numpy as np
import pandas as pd
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation

# plotting
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
%matplotlib inline
    
from statsmodels.tsa.seasonal import seasonal_decompose #decompose seasonality
from statsmodels.tsa.stattools import adfuller #test if series is stationary (then can perform ARIMA)

    
# set random seeds 
from numpy.random import seed
from tensorflow import set_random_seed

RANDOM_SEED = 2018
seed(RANDOM_SEED)
set_random_seed(RANDOM_SEED)
plt.rcParams["figure.figsize"] = [16,9]
def SMAPE (forecast, actual):
    """Returns the Symmetric Mean Absolute Percentage Error between two Series"""
    masked_arr = ~((forecast==0)&(actual==0))
    diff = abs(forecast[masked_arr] - actual[masked_arr])
    avg = (abs(forecast[masked_arr]) + abs(actual[masked_arr]))/2
    
    print('SMAPE Error Score: ' + str(round(sum(diff/avg)/len(forecast) * 100, 2)) + ' %')
def Fuller(TimeSeries):
    """Provides Fuller test results for TimeSeries"""
    stationary_test = adfuller(TimeSeries)
    print('ADF Statistic: %f' % stationary_test[0])
    print('p-value: %f' % stationary_test[1])
    print('Critical Values:')
    for key, value in stationary_test[4].items():
        print('\t%s: %.3f' % (key, value))
#Bring in the Data

train = pd.read_csv('../input/train.csv', parse_dates=['date'], index_col=['date'])
test = pd.read_csv('../input/test.csv', parse_dates=['date'], index_col=['date'])
train.index = pd.to_datetime(train.index)
train.shape, test.shape

train.shape, test.shape
train.head()
test.head()
stores = pd.DataFrame(train.groupby(['date','store']).sum()['sales']).unstack()
stores = stores.resample('7D',label='left').sum()
stores.sort_index(inplace = True)
stores.plot(figsize=(16,9), title='Weekly Store Sales', legend=True)
plt.show()
stores.head(10)
store_qtr = pd.DataFrame(stores.quantile([0.0,0.25,0.5,0.75,1.0],axis=1)).transpose()
store_qtr.sort_index(inplace = True)
store_qtr.columns = ['Min','25%','50%','75%','Max']
store_qtr.plot(figsize=(16,9), title='Weekly Quartile Sales')
plt.show()
seasonal = seasonal_decompose(pd.DataFrame(store_qtr['50%']).diff(1).iloc[1:,0],model='additive')
seasonal.plot()
plt.suptitle = 'Additive Seasonal Decomposition of Average Store Week-to-Week Sales'
plt.show()
Fuller(pd.DataFrame(store_qtr['50%']).diff(1).iloc[1:,0])
items = pd.DataFrame(train.groupby(['date','item']).sum()['sales']).unstack()
items = items.resample('7D',label='left').sum()
items.sort_index(inplace = True)

items.tail(13)
items.plot(figsize=(16,9), title='Weekly Item Sales', legend=None)
plt.show()
item_WK_qtr = pd.DataFrame(items.quantile([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],axis=1)).transpose()
item_WK_qtr.sort_index(inplace = True)
item_WK_qtr.columns = ['Min','10%','20%','30%','40%','50%','60%','70%','80%','90%','Max']
item_WK_qtr.plot(figsize=(16,9), title='Weekly Quartile Sales')
plt.show()
seasonal = seasonal_decompose(pd.DataFrame(item_WK_qtr['50%']).diff(1).iloc[1:,0],model='additive')
seasonal.plot()
plt.title = 'Additive Seasonal Decomposition of Average Item Week-to-Week Sales'
plt.show()
Fuller(pd.DataFrame(item_WK_qtr['50%']).diff(1).iloc[1:,0])
store_item = train.groupby(by=['item','store']).sum()['sales'].groupby(level=0).apply(
    lambda x: 100* x/ x.sum()).unstack()
sns.heatmap(store_item, cmap='Blues', linewidths=0.01, linecolor='gray').set_title(
    'Store % of Total Sales by Item')
plt.show()
item_store = train.groupby(by=['store','item']).sum()['sales'].groupby(level=0).apply(
    lambda x: 100* x/ x.sum()).unstack()
sns.heatmap(item_store , cmap='Blues', linewidths=0.01, linecolor='gray').set_title(
    'Item % of Total Sales by Store')
plt.show()
train['Day'] = train.index.weekday_name
train.head()
dow_store = train.groupby(['store','Day']).sum()['sales'].groupby(level=0).apply(
    lambda x: 100* x/ x.sum()).unstack().loc[:,['Monday',
                                                'Tuesday',
                                                'Wednesday',
                                                'Thursday',
                                                'Friday',
                                                'Saturday',
                                                'Sunday']]
sns.heatmap(dow_store, cmap='Blues', linewidths=0.01, linecolor='gray').set_title(
    'Day % of Total Sales by Store')
plt.show()
dow_store
dow_item = train.groupby(['item','Day']).sum()['sales'].groupby(level=0).apply(
    lambda x: 100* x/ x.sum()).unstack().loc[:,['Monday',
                                                'Tuesday',
                                                'Wednesday',
                                                'Thursday',
                                                'Friday',
                                                'Saturday',
                                                'Sunday']]
sns.heatmap(dow_item, cmap='Blues', linewidths=0.01, linecolor='gray').set_title(
    'Day % of Total Sales by Item')
plt.show()
dow = pd.DataFrame(train.groupby(['date','Day']).sum()['sales']).unstack()['sales'].loc[:,
                                                                                ['Monday',
                                                                               'Tuesday',
                                                                               'Wednesday',
                                                                               'Thursday',
                                                                               'Friday',
                                                                               'Saturday',
                                                                               'Sunday']]
dow = dow.resample('7D',label='left').sum()
dow.sort_index(inplace = True)
dow.plot(figsize=(16,9), title='Sales by Day of Week')
plt.show()
#Transforming the sales figures to limit the impact of outliers during modeling

train['sales'] = np.log1p(train['sales'])
proph_results = test.reset_index()
proph_results['sales'] = 0
proph_results.head()
#Setting Prophet Holiday Parameters

nfl_playoffs = ['2013-01-11','2013-01-12', '2013-01-19','2013-01-26','2013-02-02','2014-01-10', '2014-01-11',
            '2014-01-18', '2014-01-25','2014-02-01','2015-01-16','2015-01-17','2015-01-24','2015-01-31','2015-02-07', '2016-01-14',
            '2016-01-15', '2016-01-22','2016-01-29', '2016-02-05']
major_holidays = ['2013-01-01', '2013-12-25', '2014-01-01', '2014-12-25','2015-01-01', '2015-12-25','2016-01-01', '2016-12-25',
            '2017-01-01', '2017-12-25']

nfl_playoffs = pd.DataFrame({
    'holiday': 'nfl_playoffs',
    'ds': pd.to_datetime(nfl_playoffs),
    'lower_window': 0,
    'upper_window': 1,
})
major_holidays = pd.DataFrame({
    'holiday': 'major_holidays',
    'ds': pd.to_datetime(major_holidays),
    'lower_window': 0,
    'upper_window': 1,
})

holidays = pd.concat((nfl_playoffs, major_holidays))
#holidays
for s in proph_results['store'].unique():
    for i in proph_results['item'].unique():
        proph_train = train.loc[(train['store'] == s) & (train['item'] == i)].reset_index()
        proph_train.rename(columns={'date': 'ds', 'sales': 'y'}, inplace=True)
        
        m = Prophet(holidays=holidays, holidays_prior_scale=0.5,
            yearly_seasonality=4,  interval_width=0.95,
            changepoint_prior_scale=0.006, daily_seasonality=True)
        m.fit(proph_train[['ds', 'y']])
        future = m.make_future_dataframe(periods=len(test.index.unique()), include_history=False)
        fcst = m.predict(future)
        
        proph_results.loc[(proph_results['store'] == s) & (proph_results['item'] == i), 'sales'] = np.expm1(fcst['yhat']).values
proph_results.head(20)
proph_results.shape
proph_results.drop(['date', 'store', 'item'], axis=1, inplace=True)
proph_results.head()
proph_results.to_csv('proph_results.csv', index=False)
