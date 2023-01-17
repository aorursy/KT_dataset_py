import pandas as pd

import seaborn as sns

import numpy as np

from matplotlib import pyplot as plt

import itertools

from IPython.display import clear_output

import time



#



from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.statespace.sarimax import SARIMAX

from pandas.plotting import autocorrelation_plot

from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic

import statsmodels.formula.api as smf

import statsmodels.tsa.api as smt

import statsmodels.api as sm

import scipy.stats as scs

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import warnings

warnings.filterwarnings('ignore')
dir = '/kaggle/input/competitive-data-science-predict-future-sales'



items_df = pd.read_csv(os.path.join(dir, 'items.csv'))

item_cat_df = pd.read_csv(os.path.join(dir, 'item_categories.csv'))

shops_df = pd.read_csv(os.path.join(dir, 'shops.csv'))

test_df =  pd.read_csv(os.path.join(dir, 'test.csv'))



sales_df = pd.read_csv(os.path.join(dir, 'sales_train.csv'), parse_dates=['date']).set_index('date')

sales_df.sort_values(by = ["shop_id", "date"], ascending = True, inplace = True)

print(sales_df.info())

display(sales_df.head())
# Shop names are unique

shops_df.shape[0] == len(shops_df["shop_name"].unique())
# Shop IDs are unique

shops_df.groupby("shop_id").size()[shops_df.groupby("shop_id").size() > 1]
# No missing values in the shops df

shops_df.isnull().sum().sum()
# We don't have any duplicates in the item_category_name field

item_cat_df.shape[0] == len(item_cat_df["item_category_name"].unique())
# No missing values in the items_category df

item_cat_df.isnull().sum().sum()
# No missing values in the items

items_df.isnull().sum().sum()
# No null values in the sales df

sales_df.isnull().sum().sum()
grouped_sales = pd.read_csv(os.path.join(dir, 'sales_train.csv'), parse_dates=['date']).groupby(["date_block_num","shop_id","item_id"])[

        ["date","item_price","item_cnt_day"]].aggregate({"date":["min",'max'],"item_price":"mean","item_cnt_day":"sum"})

grouped_sales.head(10)
# number of items per cat 

count = items_df.groupby(['item_category_id']).count().sort_values(by='item_id', ascending=False)

y = count['item_id'] # sets all columns to the number of items with that id

x = count.index



plt.figure(figsize=(8,4))

ax= sns.barplot(x[:10], y[:10], alpha=0.8)

plt.title("Items per Category")

plt.ylabel('# of items', fontsize=12)

plt.xlabel('Category', fontsize=12)

plt.show()
# relative sales 

cat_sales = items_df.merge(sales_df, on='item_id').groupby('item_category_id').sum()['item_cnt_day']

y = cat_sales.sort_values(ascending=False)[:20]



plt.figure(figsize=(8,4))

ax= sns.barplot(y.index, y, alpha=0.8)

plt.title("Items sold per Category")

plt.ylabel('# of sales ', fontsize=12)

plt.xlabel('Category', fontsize=12)

plt.show()
def plot_seasonality(ts, resampling='', w=1):

    plt.figure(figsize=(16,8))

    plt.title(f'Total Sales of the company, {resampling} resampling')

    plt.xlabel('Time')

    plt.ylabel('Sales')

    plt.plot(ts, label='Original');

    plt.plot(ts.rolling(window=w,center=False).mean(),label='Rolling Mean');

    plt.plot(ts.rolling(window=w,center=False).std(),label='Rolling sd');

    plt.legend();
import datetime as dt

start = dt.datetime(2015, 7, 1)
# resampling and cutting off two months for a validation set

ts_m = sales_df["item_cnt_day"].resample("M").sum()

ts_m = ts_m[:ts_m.index.searchsorted(start)]

ts_w = sales_df["item_cnt_day"].resample("W").sum()

ts_w = ts_w[:ts_w.index.searchsorted(start)]

ts_d = sales_df["item_cnt_day"].resample("D").sum()

ts_d = ts_d[:ts_d.index.searchsorted(start)]



ts_m.name = 'Month sum series'

ts_w.name = 'Week sum series'

ts_d.name = 'Day sum series'



plot_seasonality(ts_m, 'Month ', 12)

plot_seasonality(ts_w, 'Week', 56)

plot_seasonality(ts_d, 'Day', 365)

def plot_decompose(ts, model='multiplicative', period=None):

    # additive

    res = sm.tsa.seasonal_decompose(ts,period=period,model=model)

    fig = res.plot()

    fig.set_figheight(8)

    fig.set_figwidth(16)

    fig.suptitle(f'Seasonal decomposition of {ts.name}')



try:

    plot_decompose(ts_m)

except ValueError:

    plot_decompose(ts_m, 'additive')

    

try:

    plot_decompose(ts_w)

except ValueError:

    plot_decompose(ts_w, 'additive')

    

try:

    plot_decompose(ts_d)

except ValueError:

    plot_decompose(ts_d, 'additive')
def test_stationarity(timeseries):

    

    #Perform Dickey-Fuller test:

    print('Results of Dickey-Fuller Test:')

    dftest = adfuller(timeseries, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    print (dfoutput)
test_stationarity(ts_w)
# to remove trend

from pandas import Series as Series

# create a differenced series

def difference(dataset, interval=1):

    diff = list()

    for i in range(interval, len(dataset)):

        value = dataset.iloc[i] - dataset.iloc[i - interval]

        diff.append(value)

    return Series(diff, index=dataset.index[interval:]).astype('float64')



def make_stationary(ts, period):

    #function that removes trend and seasonality, 

    #returns no trend series, no season series and a function to invert seasonality

    no_trend=difference(ts)

    no_season=difference(ts,period)

    

    # invert differenced forecast

    def invert_seasonality(serie):

        res = list()

        for i in range(len(serie)):

            try:

                add = ts.iloc[i]

            except:

                add = serie.iloc[i-period]

            value = serie.iloc[i] + add

            res.append(value)

        return Series(res, index=serie.index).astype('float64')

    

    return no_trend, no_season, invert_seasonality
plt.figure(figsize=(16,16))

plt.subplot(311)

plt.title('Original')

plt.xlabel('Time')

plt.ylabel('Sales')

plt.plot(ts_w)



no_trend,new_ts, invert = make_stationary(ts_w, 52) 



plt.subplot(312)

plt.title('After De-trend')

plt.xlabel('Time')

plt.ylabel('Sales')

plt.plot(no_trend)

plt.plot()



plt.subplot(313)

plt.title('After De-seasonalization')

plt.xlabel('Time')

plt.ylabel('Sales')

plt.plot(new_ts)

plt.plot()



validate_ts = sales_df["item_cnt_day"].resample("W").sum()

validate_ts = make_stationary(validate_ts, 52)[1]



n_predictions = 20

pred_x = pd.date_range(start=new_ts.index[-1], periods=n_predictions+1, freq='W')[1:]

# now testing the stationarity again after de-seasonality

test_stationarity(new_ts)
# function that makes some interesting plots with the timeseries

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
tsplot(new_ts)
# ar model

ar_mdl = smt.AR(new_ts).fit(method='cmle', trend='nc')



# arma model

# pick best order by aic 

# smallest aic value wins

best_aic = np.inf 

best_order = None

arma_mdl = None



rng = range(5)

for i in rng:

    for j in rng:

        try:

            tmp_mdl = smt.ARMA(new_ts, order=(i, j)).fit(method='mle', trend='nc')

            tmp_aic = tmp_mdl.aic

            if tmp_aic < best_aic:

                best_aic = tmp_aic

                best_order = (i, j)

                arma_mdl = tmp_mdl

        except: continue

            

print(f'aic: {best_aic:6.5f} | order: {best_order}')\



# simple exponential model

sex_mdl = SimpleExpSmoothing(new_ts).fit()
# complete series

plt.figure(figsize=(16,8))

plt.title('Models and original series')

plt.plot(validate_ts, label="Original")



plt.plot(ar_mdl.predict(), label="AR model", color='m')

plt.plot(ar_mdl.predict(start=pred_x[0], end=pred_x[-1]), label="AR model", color='m', ls='--')



plt.plot(arma_mdl.predict(), label="ARMA model", color='green')

plt.plot(arma_mdl.predict(start=pred_x[0], end=pred_x[-1]), label="ARMA model", color='green', ls='--')



sex_mdl.fittedvalues.plot(color='red', label='Simple Exponential Smoothing model')

sex_mdl.forecast(n_predictions).plot(color='red', style='--', label='Simple Exponential Smoothing model')

plt.legend()





#zoomed on predictions

plt.figure(figsize=(16,8))

plt.title('Predictions')

plt.plot(validate_ts[pred_x], label="Original")



plt.plot(ar_mdl.predict(start=pred_x[0], end=pred_x[-1]), label="AR model", color='m')



plt.plot(arma_mdl.predict(start=pred_x[0], end=pred_x[-1]), label="ARMA model", color='green')



sex_mdl.forecast(n_predictions).plot(color='red', label='Simple Exponential Smoothing model')

plt.legend()
# We could try using higher ranges of order, but the model already takes a long time as is, due to high seasonal order



p = range(0, 4)

d = (1,)

q = range(0, 4)

P = range(0, 4)

D = (1,)

Q = range(0, 4)

s = (52,) 

parameters = itertools.product(p,d,q,P,D,Q,s)

parameters_list = list(parameters)

len(parameters_list)
def optimize_SARIMA(ts, parameters_list):    

    results = []

    best_aic = float('inf')

    

    for i, param in enumerate(parameters_list):

        clear_output(wait=True)        

        print(i)

        try: 

            model = sm.tsa.statespace.SARIMAX(ts, order=param[:3],

                                               seasonal_order=param[3:]).fit(disp=False)

        except:

            continue

            

        aic = model.aic

        

        #Save best model, AIC and parameters

        if aic < best_aic:

            best_model = model

            best_aic = aic

            best_param = param

        results.append([param, model.aic])

        

    result_table = pd.DataFrame(results)

    result_table.columns = ['parameters', 'aic']

    #Sort in ascending order, lower AIC is better

    result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)

    

    return result_table
# TAKES TOO LONG



# start_time = time.time()

# res = optimize_SARIMA(ts_w, parameters_list)

# print(f'Time taken: {time.time()-start_time:.2f} seconds')

# res
plt.figure(figsize=(16,8))

plt.title('Total sales, inverting the seasonal differentiation')

plt.plot(invert(ar_mdl.predict(start=pred_x[0], end=pred_x[-1])), label="AR model", color='m')

plt.plot(invert(arma_mdl.predict(start=pred_x[0], end=pred_x[-1])), label="ARMA model", color='green')

plt.plot(invert(sex_mdl.forecast(n_predictions)), label="ARMA model", color='red')



plt.plot(sales_df["item_cnt_day"].resample("W").sum()[pred_x], label='Original')

plt.legend()
from fbprophet import Prophet

# Prophet needs this dataframe 

p_df = pd.DataFrame({'ds': ts_w.index,'y': ts_w.values})



model = Prophet( yearly_seasonality=True) #instantiate Prophet with only yearly seasonality as our data is monthly 

model.fit(p_df) #fit the model with your dataframe
# predict for five months in the furure and MS - month start is the frequency

future = model.make_future_dataframe(periods = n_predictions, freq = 'W')  

# now lets make the forecasts

forecast = model.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
from fbprophet.plot import plot as prophet_plot

fig = prophet_plot(model, forecast, figsize=(16,8))

plt.plot(sales_df["item_cnt_day"].resample("W").sum(), label='Original', color='red')