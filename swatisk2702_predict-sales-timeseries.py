# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import statsmodels.tsa.api as smt
import scipy.stats as scs

# settings
import warnings
warnings.filterwarnings("ignore")
sale_df = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')
item_df = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')
sale_df.head()
print('Original dtypes', sale_df.info())
sale_df.date = sale_df.date.apply(lambda x: datetime.datetime.strptime(x,"%d.%m.%Y"))
print(sale_df.info())
sale_agg = sale_df.groupby(["date_block_num", "shop_id","item_id"])["item_cnt_day","item_price"].agg({'item_cnt_day' : 'sum', 'item_price':'mean'}) 
sale_agg
item_df.head()
# number of items per cat 
x=item_df.groupby(['item_category_id']).count()
x=x.sort_values(by='item_id',ascending=False)
x=x.iloc[0:10].reset_index()
x

# #plot
plt.figure(figsize=(8,4))
ax= sns.barplot(x.item_category_id, x.item_id, alpha=0.8)
plt.title("Items per Category")
plt.ylabel('# of items', fontsize=12)
plt.xlabel('Category', fontsize=12)
plt.show()
sales_pm = sale_df.groupby(["date_block_num"]).agg({'item_cnt_day':'sum'})
plt.title('Total sales of the company')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.plot(sales_pm)
rolling_window = sales_pm.rolling(12)
plt.figure(figsize= (16,6))
plt.plot(rolling_window.mean(), label = "Rolling mean")
plt.plot(rolling_window.std(), label = "Rolling standard deviation")
plt.legend()
result1 = sm.tsa.seasonal_decompose(sales_pm,model='additive', freq=12)
fig1 = result1.plot()
result2 = sm.tsa.seasonal_decompose(sales_pm, model ='multiplicative', freq=12)
fig2 = result2.plot()
def adf_test(timeseries):
    result = adfuller(timeseries, autolag='AIC')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    for key, value in result[4].items():
        print('Critial Values:')
        print(f'   {key}, {value}')
adf_test(sales_pm)
# to remove trend
from pandas import Series as Series
# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

# invert differenced forecast
def inverse_difference(last_ob, value):
    return value + last_ob
ts=sales_pm.groupby(["date_block_num"])["item_cnt_day"].sum()
ts.astype('float')
plt.figure(figsize=(16,16))
plt.subplot(311)
plt.title('Original')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.plot(ts)
plt.subplot(312)
plt.title('After De-trend')
plt.xlabel('Time')
plt.ylabel('Sales')
new_ts=difference(ts)
plt.plot(new_ts)
plt.plot()

plt.subplot(313)
plt.title('After De-seasonalization')
plt.xlabel('Time')
plt.ylabel('Sales')
new_ts=difference(ts,12)       # assuming the seasonality is 12 months long
plt.plot(new_ts)
plt.plot()
adf_test(new_ts)
def tsplot(y, lags = None, figsize = (10,8), style ='bmh', title=''):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        layout = (3,2)
        ts_ax=plt.subplot2grid(layout, (0,0), colspan = 2)
        acf_ax = plt.subplot2grid(layout, (1,0))
        pacf_ax = plt.subplot2grid(layout,(1,1))
        qq_ax = plt.subplot2grid(layout,(2,0))
        pp_ax = plt.subplot2grid(layout,(2,1))
        
        y.plot(ax = ts_ax)
        ts_ax.set_title(title)
        smt.graphics.plot_acf(y, lags = lags, ax= acf_ax, alpha = 0.5)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax,alpha = 0.5)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')
        scs.probplot(y, sparams =(y.mean(),y.std()), plot=pp_ax )
        plt.tight_layout()
    return
# Simulate an AR(1) process with alpha = 0.6
np.random.seed(1)
n_samples = 1000
alpha = 0.6
x = w = np.random.normal(size = n_samples)
for t in range(n_samples):
    x[t] = alpha * x[t-1] + w[t]
_ = tsplot(x, lags = 12,title = "AR(1)  process" )
# Simulate an AR(2) process
n = int(1000)
alphas = np.array([.444, .333])
betas = np.array([0.])

# Python requires us to specify the zero-lag value which is 1
# Also note that the alphas for the AR model must be negated
# We also set the betas for the MA equal to 0 for an AR(p) model
# For more information see the examples at statsmodels.org

ar = np.r_[1, -alphas]
ma = np.r_[1, betas]

ar2 = smt.arma_generate_sample(ar=ar, ma= ma, nsample = n)
_ = tsplot(ar2, lags=12, title = "AR(2) process")
# Simulate an MA(1) process
n = int(1000)
# set the AR(p) alphas equal to 0
alphas = np.array([0.])
betas = np.array([0.8])
# add zero-lag and negate alphas
ar = np.r_[1, -alphas]
ma = np.r_[1, betas]
ma1 = smt.arma_generate_sample(ar=ar, ma=ma, nsample=n) 
limit=12
_ = tsplot(ma1, lags=limit,title="MA(1) process")
# Simulate MA(2) process with betas 0.6, 0.4
n = int(1000)
alphas = np.array([0.])
betas = np.array([0.6, 0.4])
ar = np.r_[1, -alphas]
ma = np.r_[1, betas]

ma3 = smt.arma_generate_sample(ar=ar, ma=ma, nsample=n)
_ = tsplot(ma3, lags=12,title="MA(2) process")
# Simulate an ARMA(2, 2) model with alphas=[0.5,-0.25] and betas=[0.5,-0.3]
max_lag = 12

n = int(5000) # lots of samples to help estimates
burn = int(n/10) # number of samples to discard before fit

alphas = np.array([0.8, -0.65])
betas = np.array([0.5, -0.7])
ar = np.r_[1, -alphas]
ma = np.r_[1, betas]

arma22 = smt.arma_generate_sample(ar=ar, ma=ma, nsample=n, burnin=burn)
_ = tsplot(arma22, lags=max_lag,title="ARMA(2,2) process")
# pick best order by aic 
# smallest aic value wins
best_aic = np.inf 
best_order = None
best_mdl = None

rng = range(5)
for i in rng:
    for j in rng:
        try:
            tmp_mdl = smt.ARMA(arma22, order=(i, j)).fit(method='mle', trend='nc')
            tmp_aic = tmp_mdl.aic
            if tmp_aic < best_aic:
                best_aic = tmp_aic
                best_order = (i, j)
                best_mdl = tmp_mdl
        except: continue


print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))
# pick best order by aic 
# smallest aic value wins
best_aic = np.inf 
best_order = None
best_mdl = None

rng = range(5)
for i in rng:
    for j in rng:
        try:
            tmp_mdl = smt.ARMA(new_ts.values, order=(i, j)).fit(method='mle', trend='nc')
            # mle = most likelihood estimate, nc = no constant
            tmp_aic = tmp_mdl.aic
            if tmp_aic < best_aic:
                best_aic = tmp_aic
                best_order = (i, j)
                best_mdl = tmp_mdl
        except: continue


print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))
ts = sale_df.groupby(["date_block_num"])["item_cnt_day"].sum()
ts.index = pd.date_range(start = '2013-01-01', end = '2015-10-01', freq='MS')
ts = ts.reset_index()
ts.head()
from fbprophet import Prophet
#prophet reqiures a pandas df at the below config 
# ( date column named as DS and the value column as Y)
ts.columns=['ds','y']
model = Prophet( yearly_seasonality=True) #instantiate Prophet with only yearly seasonality as our data is monthly 
model.fit(ts) #fit the model with your dataframe
# predict for five months in the future and MS - month start is the frequency
future = model.make_future_dataframe(periods = 5, freq = 'MS')  
# now lets make the forecasts
forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig = model.plot(forecast)
fig2 = model.plot_components(forecast)
total_sales = sale_df.groupby(["date_block_num"])['item_cnt_day'].sum()
dates = pd.date_range(start= '2013-01-01', end = '2015-10-01', freq = 'MS')
total_sales.index = dates
total_sales.head()
# get the unique combinations of item-store from the sales data at monthly level
sale_pm_item = sale_df.groupby(['shop_id', 'item_id', 'date_block_num'])['item_cnt_day'].sum()

# arrange it conviniently to perform the hts 
sale_pm_item = sale_pm_item.unstack(level = -1).fillna(0)
sale_pm_item = sale_pm_item.T
dates = pd.date_range(start = '2013-01-01', end = '2015-10-01',freq = 'MS')
sale_pm_item.index = dates
sale_pm_item = sale_pm_item.reset_index()
sale_pm_item.head()
import time
start_time = time.time()

# Bottoms up
# Calculating the base forecasts using prophet
# From HTSprophet pachage -- https://github.com/CollinRooney12/htsprophet/blob/master/htsprophet/hts.py

forecastsDict = {}

for node in range(len(sale_pm_item[0])):
    nodeToForecast = pd.concat([sale_pm_item.iloc[:,0], sale_pm_item.iloc[:,node+1]], axis = 1)
    # rename for prophet compatability
    nodeToForecast.columns = ["ds", "y"]
    growth = 'linear'
    model = Prophet(growth, yearly_seasonality = True)
    model.fit(nodeToForecast)
    future = model.make_future_dataframe(periods = 1, freq = 'MS')
    forecastsDict[node] = model.predict(future)
    if (node== 10):
        end_time=time.time()
        print("forecasting for ",node,"th node and took",end_time-start_time,"s")
        break
    
    
sale_pm_shop = sale_df.groupby(["date_block_num", "shop_id"])["item_cnt_day"].sum()
# get the shops to the columns
sale_pm_shop = sale_pm_shop.unstack(level = 1)
sale_pm_shop = sale_pm_shop.fillna(0)
sale_pm_shop.index = dates 
sale_pm_shop = sale_pm_shop.reset_index()
sale_pm_shop.head()
start_time = time.time()

# Bottoms up
# Calculating the base forecasts using prophet
# From HTSprophet package -- https://github.com/CollinRooney12/htsprophet/blob/master/htsprophet/hts.py

forecastsDict = {}

for node in range(len(sale_pm_shop[0])):
    nodeToForecast = pd.concat([sale_pm_shop.iloc[:,0], sale_pm_shop.iloc[:,node+1]], axis = 1)
    # rename for prophet compatability
    nodeToForecast.columns = ["ds", "y"]
    growth = 'linear'
    model = Prophet(growth, yearly_seasonality = True)
    model.fit(nodeToForecast)
    future = model.make_future_dataframe(periods = 1, freq = 'MS')
    forecastsDict[node] = model.predict(future)
    
for key in range(len(forecastsDict.keys())):
    
    f1 = np.array(forecastsDict[key].yhat)
    f2 = f1[:,np.newaxis]
    if key == 0:
        predictions = f2.copy()
    else:
        predictions = np.concatenate((predictions, f2), axis = 1)
predictions[-1]