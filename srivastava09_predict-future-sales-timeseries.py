import numpy as np
import pandas as pd
import random as rd
import datetime

#Viz
import matplotlib.pyplot as plt
import seaborn as sns

#Time Series
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller, acf, pacf, arma_order_select_ic
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs

#Ignore Warning
import warnings
warnings.filterwarnings("ignore")
sales = pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")
item_cat = pd.read_csv("../input/competitive-data-science-predict-future-sales/item_categories.csv")
items = pd.read_csv("../input/competitive-data-science-predict-future-sales/items.csv")
shops = pd.read_csv("../input/competitive-data-science-predict-future-sales/shops.csv")
test = pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")
#Convertingg date to datetime
sales['date'] = pd.to_datetime(sales['date'])
monthly_sales = sales.groupby(['date_block_num','shop_id','item_id'])['date','item_price','item_cnt_day'].agg({"date":["min","max"],"item_price":"mean","item_cnt_day":"sum"})
x = items.groupby(['item_category_id']).count()
x = x.rename(columns = {'item_id':'total_item_count'})
x = x.sort_values(by='total_item_count',ascending=False)
x = x.iloc[0:10].reset_index()

plt.figure(figsize=(8,4))
sns.barplot(x.item_category_id,x.total_item_count, alpha = 0.8)
plt.title("Item category to Item Count")
plt.xlabel("Item Category")
plt.ylabel("Item Count")
plt.show()
ts = sales.groupby(['date_block_num'])['item_cnt_day'].sum()
plt.figure(figsize=(16,8))
plt.title('Total sales of company')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.plot(ts)
plt.scatter(ts[ts > 140000].index,ts[ts > 140000],color='red')
plt.figure(figsize=(16,6))
plt.plot(ts.rolling(window=12,center=False).mean(),label="Rolling Mean")
plt.plot(ts.rolling(window=12,center=False).std(), label='Rolling std')
plt.legend()
import statsmodels.api as sm
res = sm.tsa.seasonal_decompose(ts.values, freq=12,model='multiplicative')
fig = res.plot()
res = sm.tsa.seasonal_decompose(ts.values, freq=12,model='additive')
fig = res.plot()
def test_stationarity(timeseries):
    print("Results of Dicky-Fuller Test:")
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput  = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value',
                                              'Lags Used', 'Number of Observation Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
test_stationarity(ts)
from pandas import Series as Series

def difference(dataset, interval = 1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

def inverse_difference(last_ob, value):
    return value + last_ob
ts = sales.groupby(['date_block_num'])['item_cnt_day'].sum()
plt.figure(figsize=(16,16))
plt.subplot(311)
plt.title('Original')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.plot(ts)
plt.plot(ts.rolling(window=12,center=False).mean())
plt.subplot(312)
plt.title('After De-Trend')
plt.xlabel('Time')
plt.ylabel('Sales')
new_ts = difference(ts)
plt.plot(new_ts)
plt.plot(new_ts.rolling(window=12,center=False).mean())
plt.plot()

plt.subplot(313)
plt.title("After De-Seasonlization")
plt.xlabel('Time')
plt.ylabel('Sales')
new_ts  = difference(ts,12)
plt.plot(new_ts)
plt.plot(new_ts.rolling(window=12,center=False).mean())
plt.plot()
test_stationarity(difference(ts))
def tstplot(y, lags=None, figsize=(10,8),style='bmh',title=''):
    if not isinstance(y,pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):
        fig = plt.figure(figsize = figsize)
        layout = (3,2)
        ts_ax = plt.subplot2grid(layout, (0,0), colspan=2)
        acf_ax = plt.subplot2grid(layout,(1,0))
        pacf_ax = plt.subplot2grid(layout,(1,1))
        qq_ax = plt.subplot2grid(layout, (2,0))
        pp_ax = plt.subplot2grid(layout, (2,1))
        
        y.plot(ax = ts_ax)
        ts_ax.set_title(title)
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax,alpha=0.5)
        sm.qqplot(y,line='s',ax=qq_ax)
        qq_ax.set_title("QQ plot")
        scs.probplot(y, sparams=(y.mean(),y.std()),plot=pp_ax)
        plt.tight_layout()
        return
np.random.seed(1)
n_samples = int(1000)
a=0.6
x = w = np.random.normal(size=n_samples)

for t in range(n_samples):
    x[t] = a * x[t-1] + w[t]

limit=12
_ = tstplot(x,lags = limit, title="AR(1)process")
n = int(1000)
alphas = np.array([.444,.333])
betas = np.array([0.0])

ar = np.r_[1,-alphas]
ma = np.r_[1,betas]

ar2 = smt.arma_generate_sample(ar=ar,ma = ma,nsample=n)
_ = tstplot(ar2,lags=12,title="AR(2) process")
n = int(1000)
alphas = np.array([0.])
betas = np.array([0.8])
ar = np.r_[1,-alphas]
ma = np.r_[1,betas]
ma1 = smt.arma_generate_sample(ar=ar, ma=ma, nsample=n)
limit=12
_ = tstplot(ma1, lags=limit, title="MA(1) process")
n = int(1000)
alphas = np.array([0.])
betas = np.array([0.6,0.4])
ar = np.r_[1,-alphas]
ma = np.r_[1,betas]
ma3 = smt.arma_generate_sample(ar=ar,ma=ma,nsample=n)
_ = tstplot(ma3,lags=12,title="MA(2) process")
max_lag = 12
n = int(1000)
burn = n // 10
alphas = np.array([0.8,-0.65])
betas = np.array([0.5,-0.7])
ar = np.r_[1,-alphas]
ma = np.r_[1,betas]
arma22 = smt.arma_generate_sample(ar=ar,ma=ma,nsample=n,burnin=burn)
_ = tstplot(arma22,lags=max_lag,title="ARMA(2,2) process")
best_aic = np.inf
best_order = None
best_mdl = None
rng = range(5)
for i in rng:
    for j in rng:
        try:
            tmp_mdl = smt.ARMA(new_ts.values,order=(i,j)).fit(method='mle',trend='nc')
            tmp_aic = tmp_mdl.aic
            if tmp_aic < best_aic:
                best_aic = tmp_aic
                best_order = (i,j)
                best_mdl = tmp_mdl
        except Exception as e:
            print(e)

print('aic: {:6.5f} | order:{}'.format(best_aic,best_order))
ts = sales.groupby(['date_block_num'])['item_cnt_day'].sum()
ts.index = pd.date_range(start='2013-01-01',end='2015-10-01',freq='MS')
ts = ts.reset_index()
ts.head()
from fbprophet import Prophet
ts.columns = ['ds','y']
model = Prophet(yearly_seasonality = True)
model.fit(ts)
future = model.make_future_dataframe(periods = 5,freq='MS')
forecast = model.predict(future)
forecast[['ds','yhat',
         'yhat_lower','yhat_upper']].tail()
_ = model.plot(forecast)
_ = model.plot_components(forecast)
total_sales = sales.groupby(['date_block_num'])['item_cnt_day'].sum()
dates = pd.date_range(start='2013-01-01',end='2015-10-01',freq='MS')
total_sales.index = dates
total_sales.head()
monthly_sales = sales.groupby(['shop_id','item_id','date_block_num'])['item_cnt_day'].sum()
monthly_sales = monthly_sales.unstack(-1).fillna(0)
monthly_sales = monthly_sales.T
dates = pd.date_range(start='2013-01-01',end='2015-10-01',freq='MS')
monthly_sales.index = dates
monthly_sales = monthly_sales.reset_index()
monthly_sales.head()
import time
start_time = time.time()

forecastsDict = {}
for node in range(len(monthly_sales)):
    nodeToForecast = pd.concat([monthly_sales.iloc[:,0], monthly_sales.iloc[:,node+1]],axis=1)
    nodeToForecast.columns = ['ds', 'y']
    growth = 'linear'
    m = Prophet(growth, yearly_seasonality = True)
    m.fit(nodeToForecast)
    future = m.make_future_dataframe(periods=1,freq='MS')
    forecastsDict[node] = m.predict(future)
    if node == 10:
        end_time = time.time()
        print('forecasting for ',node,'th node and took',end_time - start_time,"s")
        break
monthly_shop_sales = sales.groupby(['date_block_num','shop_id'])['item_cnt_day'].sum()
monthly_shop_sales = monthly_shop_sales.unstack(level=1)
monthly_shop_sales = monthly_shop_sales.fillna(0)
monthly_shop_sales.index = dates
monthly_shop_sales = monthly_shop_sales.reset_index()
monthly_shop_sales.head()
start_time = time.time()

forecastDict = {}
for node in range(len(monthly_shop_sales)):
    nodeToForecast = pd.concat([monthly_shop_sales.iloc[:,0], monthly_shop_sales.iloc[:,node+1]],axis=1)
    nodeToForecast.columns = ['ds', 'y']
    growth = 'linear'
    m = Prophet(growth, yearly_seasonality=True)
    m.fit(nodeToForecast)
    future = m.make_future_dataframe(periods = 1,freq='MS')
    forecastDict[node] = m.predict(future)
nCols = len(list(forecastDict.keys())) + 1
for key in range(0,nCols-1):
    f1 = np.array(forecastDict[key].yhat)
    f2 = f1[:,np.newaxis]
    if key == 0:
        predictions = f2.copy()
    else:
        predictions = np.concatenate((predictions,f2),axis=1)
predictions_unknown = predictions[-1]
predictions_unknown
start_time = time.time()
forecastDict = {}
for node in range(0,len(monthly_shop_sales.columns)-1):
    nodeToForecast = pd.concat([monthly_shop_sales.iloc[:,0], monthly_shop_sales.iloc[:,node+1]],axis=1)
    nodeToForecast.columns = ['ds', 'y']
    growth = 'linear'
    m = Prophet(growth, yearly_seasonality=True)
    m.fit(nodeToForecast)
    future = m.make_future_dataframe(periods = 1,freq='MS')
    forecastDict[node] = m.predict(future)
    print('Processed Node : ',node,'Time : ',(time.time() - start_time))
sales_shop = sales.groupby(['shop_id'])['item_cnt_day'].sum()
sales_shop = sales_shop.fillna(0)
sales_shop = sales_shop.reset_index()
sales_shop.head()
sales_shop_item = sales.groupby(['shop_id','item_id'])['item_cnt_day'].sum()
sales_shop_item = sales_shop_item.fillna(0)
sales_shop_item = sales_shop_item.reset_index()
sales_shop_item.head()
merged_salesc = pd.merge(sales,items,on='item_id')
merged_salesc
merged_salesc_category = merged_salesc.groupby(['shop_id','item_category_id'])['item_cnt_day'].sum()
merged_salesc_category = merged_salesc_category.fillna(0)
merged_salesc_category = merged_salesc_category.reset_index()
merged_salesc_category.head()
merged_salesc_category_item = merged_salesc.groupby(['item_category_id'])['item_cnt_day'].sum()
merged_salesc_category_item = merged_salesc_category_item.fillna(0)
merged_salesc_category_item = merged_salesc_category_item.reset_index()
merged_salesc_category_item.tail()
merged_salesc_category_item_id = merged_salesc.groupby(['item_category_id','item_id'])['item_cnt_day'].sum()
merged_salesc_category_item_id = merged_salesc_category_item_id.fillna(0)
merged_salesc_category_item_id = merged_salesc_category_item_id.reset_index()
merged_salesc_category_item_id.tail(50)
import pdb
def get_results(x):
    shop_id = x['shop_id'].item()
    item_id = x['item_id'].item()
    item_category_id = items.loc[items.item_id == item_id]['item_category_id'].item()
    total_sale = sales_shop.loc[sales_shop.shop_id == shop_id]['item_cnt_day'].item()
    try:
        item_sale = merged_salesc_category.loc[(merged_salesc_category.shop_id == shop_id) & (merged_salesc_category.item_category_id == item_category_id)]['item_cnt_day'].item()
    except Exception as e1:
        item_sale = 1
    total_sale = (item_sale / total_sale) * forecastDict[shop_id]['yhat'].tail(1).item()
    try:
        item_sale_id_total = merged_salesc_category_item.loc[merged_salesc_category_item.item_category_id == item_category_id]['item_cnt_day'].item()
        cat_item_sale = merged_salesc_category_item_id.loc[(merged_salesc_category_item_id.item_category_id == item_category_id) & (merged_salesc_category_item_id.item_id == item_id)]['item_cnt_day'].item()
    except Exception as e:
        cat_item_sale = 1
    
    total_sale = (cat_item_sale / item_sale_id_total) * total_sale
    #if total_sale > 1:
    #    pdb.set_trace()
    return total_sale
    