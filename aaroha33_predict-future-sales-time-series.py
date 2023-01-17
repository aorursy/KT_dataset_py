import numpy as np 

import pandas as pd 

import random as rd

import datetime

import time

import sys

import gc

import pickle

sys.version_info



from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt 

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 12, 4

import seaborn as sns 





import warnings

warnings.filterwarnings("ignore")



from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.dummy import DummyRegressor



from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.statespace.sarimax import SARIMAX

from pandas.plotting import autocorrelation_plot

from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic

import statsmodels.formula.api as smf

import statsmodels.tsa.api as smt

import statsmodels.api as sm

import scipy.stats as scs



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")

item_categories = pd.read_csv("../input/competitive-data-science-predict-future-sales/item_categories.csv")

items = pd.read_csv("../input/competitive-data-science-predict-future-sales/items.csv")

shops = pd.read_csv("../input/competitive-data-science-predict-future-sales/shops.csv")

test = pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")

submission = pd.read_csv("../input/competitive-data-science-predict-future-sales/sample_submission.csv")
print(sales_train.info())

print('----------------------------------')

print(items.info())

print('----------------------------------')

print(item_categories.info())

print('----------------------------------')

print(shops.info())

print('----------------------------------')

print(test.info())

print('----------------------------------')

print(submission.info())
print(train.isnull().sum())

print('----------------------------------')

print(items.isnull().sum())

print('----------------------------------')

print(item_categories.isnull().sum())

print('----------------------------------')

print(shops.isnull().sum())

print('----------------------------------')

print(test.isnull().sum())
plt.figure(figsize=(10,4))

plt.xlim(-100, 3000)

sns.boxplot(x=train.item_cnt_day)



plt.figure(figsize=(10,4))

plt.xlim(train.item_price.min(), train.item_price.max()*1.1)

sns.boxplot(x=train.item_price)



print(len(train[train.item_cnt_day>999]))

print(len(train[train.item_cnt_day>500]))

print(len(train[train.item_cnt_day<501]))



train = train[train.item_price<100000]

train = train[train.item_cnt_day<1000]
plt.figure(figsize=(10,4))

plt.xlim(-100, 1000)

sns.boxplot(x=train.item_cnt_day)



plt.figure(figsize=(10,4))

plt.xlim(0, 100000)

#plt.xlim(train.item_price.min(), train.item_price.max()*1.1)

sns.boxplot(x=train.item_price)
train = train[train.item_price > 0].reset_index(drop=True)

train[train.item_cnt_day <= 0].item_cnt_day.unique()

train.loc[train.item_cnt_day < 1, 'item_cnt_day'] = 0
plt.figure(figsize=(10,4))

plt.xlim(-100, 1000)

sns.boxplot(x=train.item_cnt_day)



plt.figure(figsize=(10,4))

#plt.xlim(0, 100000)

plt.xlim(train.item_price.min(), train.item_price.max()*1.1)

sns.boxplot(x=train.item_price)
shops.shop_name.unique()
shops.groupby(['category']).sum()
monthly_sales=sales_train.groupby(["date_block_num","shop_id","item_id"])[

    "date","item_price","item_cnt_day"].agg({"date":["min",'max'],"item_price":"mean","item_cnt_day":"sum"})
monthly_sales.head(10)
x=items.groupby(['item_category_id']).count()

x=x.sort_values(by='item_id',ascending=False)

x=x.iloc[0:10].reset_index()



plt.figure(figsize=(8,4))

ax= sns.barplot(x.item_category_id, x.item_id, alpha=0.8)

plt.title("Items per Category")

plt.ylabel('# of items', fontsize=12)

plt.xlabel('Category', fontsize=12)

plt.show()
ts=sales_train.groupby(["date_block_num"])["item_cnt_day"].sum()

ts.astype('float')

plt.figure(figsize=(16,8))

plt.title('Total Sales of the company')

plt.xlabel('Time')

plt.ylabel('Sales')

plt.plot(ts);
import statsmodels.api as sm

res = sm.tsa.seasonal_decompose(ts.values,freq=12,model="multiplicative")

plt.figure(figsize=(16,12))

fig = res.plot()

fig.show()
res = sm.tsa.seasonal_decompose(ts.values,freq=12,model="additive")

plt.figure(figsize=(16,12))

fig = res.plot()

fig.show()
# Stationarity tests

def test_stationarity(timeseries):

    

    #Perform Dickey-Fuller test:

    print('Results of Dickey-Fuller Test:')

    dftest = adfuller(timeseries, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    print (dfoutput)



test_stationarity(ts)
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
ts=sales_train.groupby(["date_block_num"])["item_cnt_day"].sum()

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
# now testing the stationarity again after de-seasonality

test_stationarity(new_ts)
def tsplot(y, lags=None, figsize=(10, 8), style='bmh',title=''):

    if not isinstance(y, pd.Series):

        y = pd.Series(y)

    with plt.style.context(style):    

        fig = plt.figure(figsize=figsize)

        #mpl.rcParams['font.family'] = 'Ubuntu Mono'

        layout = (3, 2)

        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)

        acf_ax = plt.subplot2grid(layout, (1, 0))

        pacf_ax = plt.subplot2grid(layout, (1, 1))

        qq_ax = plt.subplot2grid(layout, (2, 0))

        pp_ax = plt.subplot2grid(layout, (2, 1))

        

        y.plot(ax=ts_ax)

        ts_ax.set_title(title)

        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)

        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)

        sm.qqplot(y, line='s', ax=qq_ax)

        qq_ax.set_title('QQ Plot')        

        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)



        plt.tight_layout()

    return 
# Simulate an AR(1) process with alpha = 0.6

np.random.seed(1)

n_samples = int(1000)

a = 0.6

x = w = np.random.normal(size=n_samples)



for t in range(n_samples):

    x[t] = a*x[t-1] + w[t]

limit=12    

_ = tsplot(x, lags=limit,title="AR(1)process")
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



ar2 = smt.arma_generate_sample(ar=ar, ma=ma, nsample=n) 

_ = tsplot(ar2, lags=12,title="AR(2) process")

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

best_aic = np.inf 

best_order = None

best_mdl = None



rng = range(5)

for i in rng:

    for j in rng:

        try:

            tmp_mdl = smt.ARMA(new_ts.values, order=(i, j)).fit(method='mle', trend='nc')

            tmp_aic = tmp_mdl.aic

            if tmp_aic < best_aic:

                best_aic = tmp_aic

                best_order = (i, j)

                best_mdl = tmp_mdl

        except: continue





print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))
# adding the dates to the Time-series as index

ts=sales_train.groupby(["date_block_num"])["item_cnt_day"].sum()

ts.index=pd.date_range(start = '2013-01-01',end='2015-10-01', freq = 'MS')

ts=ts.reset_index()

ts.head()
from fbprophet import Prophet

#prophet reqiures a pandas df at the below config 

# ( date column named as DS and the value column as Y)

ts.columns=['ds','y']

model = Prophet( yearly_seasonality=True) #instantiate Prophet with only yearly seasonality as our data is monthly 

model.fit(ts) #fit the model with your dataframe
# predict for five months in the furure and MS - month start is the frequency

future = model.make_future_dataframe(periods = 5, freq = 'MS')  

# now lets make the forecasts

forecast = model.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
model.plot(forecast)
model.plot_components(forecast)
total_sales=sales_train.groupby(['date_block_num'])["item_cnt_day"].sum()

dates=pd.date_range(start = '2013-01-01',end='2015-10-01', freq = 'MS')



total_sales.index=dates

total_sales.head()
# get the unique combinations of item-store from the sales data at monthly level

monthly_sales=sales_train.groupby(["shop_id","item_id","date_block_num"])["item_cnt_day"].sum()

# arrange it conviniently to perform the hts 

monthly_sales=monthly_sales.unstack(level=-1).fillna(0)

monthly_sales=monthly_sales.T

dates=pd.date_range(start = '2013-01-01',end='2015-10-01', freq = 'MS')

monthly_sales.index=dates

monthly_sales=monthly_sales.reset_index()

monthly_sales.head()
import time

start_time=time.time()



# Bottoms up

# Calculating the base forecasts using prophet

# From HTSprophet pachage -- https://github.com/CollinRooney12/htsprophet/blob/master/htsprophet/hts.py

forecastsDict = {}

for node in range(len(monthly_sales)):

    # take the date-column and the col to be forecasted

    nodeToForecast = pd.concat([monthly_sales.iloc[:,0], monthly_sales.iloc[:, node+1]], axis = 1)

#     print(nodeToForecast.head())  # just to check

# rename for prophet compatability

    nodeToForecast = nodeToForecast.rename(columns = {nodeToForecast.columns[0] : 'ds'})

    nodeToForecast = nodeToForecast.rename(columns = {nodeToForecast.columns[1] : 'y'})

    growth = 'linear'

    m = Prophet(growth, yearly_seasonality=True)

    m.fit(nodeToForecast)

    future = m.make_future_dataframe(periods = 1, freq = 'MS')

    forecastsDict[node] = m.predict(future)

    if (node== 10):

        end_time=time.time()

        print("forecasting for ",node,"th node and took",end_time-start_time,"s")

        break
monthly_shop_sales=sales_train.groupby(["date_block_num","shop_id"])["item_cnt_day"].sum()

# get the shops to the columns

monthly_shop_sales=monthly_shop_sales.unstack(level=1)

monthly_shop_sales=monthly_shop_sales.fillna(0)

monthly_shop_sales.index=dates

monthly_shop_sales=monthly_shop_sales.reset_index()

monthly_shop_sales.head()
start_time=time.time()



# Calculating the base forecasts using prophet

# From HTSprophet pachage -- https://github.com/CollinRooney12/htsprophet/blob/master/htsprophet/hts.py

forecastsDict = {}

for node in range(len(monthly_shop_sales)):

    # take the date-column and the col to be forecasted

    nodeToForecast = pd.concat([monthly_shop_sales.iloc[:,0], monthly_shop_sales.iloc[:, node+1]], axis = 1)

#     print(nodeToForecast.head())  # just to check

# rename for prophet compatability

    nodeToForecast = nodeToForecast.rename(columns = {nodeToForecast.columns[0] : 'ds'})

    nodeToForecast = nodeToForecast.rename(columns = {nodeToForecast.columns[1] : 'y'})

    growth = 'linear'

    m = Prophet(growth, yearly_seasonality=True)

    m.fit(nodeToForecast)

    future = m.make_future_dataframe(periods = 1, freq = 'MS')

    forecastsDict[node] = m.predict(future)
#predictions = np.zeros([len(forecastsDict[0].yhat),1]) 

nCols = len(list(forecastsDict.keys()))+1

for key in range(0, nCols-1):

    f1 = np.array(forecastsDict[key].yhat)

    f2 = f1[:, np.newaxis]

    if key==0:

        predictions=f2.copy()

       # print(predictions.shape)

    else:

       predictions = np.concatenate((predictions, f2), axis = 1)
predictions_unknown=predictions[-1]

predictions_unknown
test.tail()
gp = sales_train.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_cnt_day': ['sum']})
X = np.array(list(map(list, gp.index.values)))

y_train = gp.values
test['date_block_num'] = train['date_block_num'].max() + 1

X_test = test[['date_block_num', 'shop_id', 'item_id']].values
samp = np.random.permutation(np.arange(len(y_train)))[:int(len(y_train)*.2)]

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[samp,1], X[samp,2], y_train[samp].ravel(), c=X[samp,0], marker='.')
oh0 = OneHotEncoder(categories='auto').fit(X[:,0].reshape(-1, 1))

x0 = oh0.transform(X[:,0].reshape(-1, 1))
oh1 = OneHotEncoder(categories='auto').fit(X[:,1].reshape(-1, 1))

x1 = oh1.transform(X[:,1].reshape(-1, 1))

x1_t = oh1.transform(X_test[:,1].reshape(-1, 1))
print(X[:, :1].shape, x1.toarray().shape, X[:, 2:].shape)

X_train = np.concatenate((X[:, :1], x1.toarray(), X[:, 2:]), axis=1)

X_test = np.concatenate((X_test[:, :1], x1_t.toarray(), X_test[:, 2:]), axis=1)
dmy = DummyRegressor().fit(X_train, y_train)



reg = LinearRegression().fit(X_train, y_train)



rfr = RandomForestRegressor().fit(X_train, y_train.ravel())
rmse_dmy = np.sqrt(mean_squared_error(y_train, dmy.predict(X_train)))

print('Dummy RMSE: %.4f' % rmse_dmy)

rmse_reg = np.sqrt(mean_squared_error(y_train, reg.predict(X_train)))

print('LR RMSE: %.4f' % rmse_reg)

rmse_rfr = np.sqrt(mean_squared_error(y_train, rfr.predict(X_train)))

print('RFR RMSE: %.4f' % rmse_rfr)
y_test = rfr.predict(X_test)
sample_submission['item_cnt_month'] = y_test
sample_submission.to_csv('xgb_submission.csv', index=False)