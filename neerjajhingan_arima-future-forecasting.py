# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random as rd # generating random numbers

import datetime # manipulating date formats

# Viz

import matplotlib.pyplot as plt # basic plotting

import seaborn as sns # for prettier plots





# TIME SERIES

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.statespace.sarimax import SARIMAX

from pandas.plotting import autocorrelation_plot

from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic

import statsmodels.formula.api as smf

import statsmodels.tsa.api as smt

import statsmodels.api as sm

import scipy.stats as scs





# settings

import warnings

warnings.filterwarnings("ignore")
# Import all of them 

sales=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv")



# settings

import warnings

warnings.filterwarnings("ignore")



item_cat=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv")

item=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/items.csv")

sub=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv")

shops=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/shops.csv")

test=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/test.csv")
print(sales.info())
#formatting the date column correctly or string datetime to datetime object

#sales.date=sales.date.apply(lambda x:datetime.datetime.strptime(x, '%d.%m.%Y'))

# check

#sales.info()
# Aggregate to monthly level the required metrics



monthly_sales=sales.groupby(["date_block_num","shop_id","item_id"]).agg({"date":["min",'max'],"item_price":"mean","item_cnt_day":"sum"})



## Lets break down the line of code here:

# aggregate by date-block(month),shop_id and item_id

# select the columns date,item_price and item_cnt(sales)

# Provide a dictionary which says what aggregation to perform on which column

# min and max on the date

# average of the item_price

# sum of the sales
# Print first 20 Values

monthly_sales.head(20)
# number of items per cat 

x=item.groupby(['item_category_id']).count()

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
ts=sales.groupby(["date_block_num"])["item_cnt_day"].sum()

plt.figure(figsize=(16,8))

plt.title('Total Sales of the company')

plt.xlabel('Time')

plt.ylabel('Sales')

plt.plot(ts);
plt.figure(figsize=(16,6))

plt.plot(ts.rolling(window=12,center=False).mean(),label='Rolling Mean');

plt.plot(ts.rolling(window=12,center=False).std(),label='Rolling sd');

plt.legend();
#Statsmodels is a Python package that provides a complement to scipy for statistical computations including descriptive statistics and estimation and inference for statistical models.

import statsmodels.api as sm

# multiplicative

res = sm.tsa.seasonal_decompose(ts.values,freq=12,model="multiplicative")

#plt.figure(figsize=(16,12))

fig = res.plot()

#fig.show()
# Additive model

res = sm.tsa.seasonal_decompose(ts.values,freq=12,model="additive")

#plt.figure(figsize=(16,12))

fig = res.plot()

#fig.show()
# Stationarity tests

def test_stationarity(timeseries):

    

    #Perform Dickey-Fuller test:

    print('Results of Dickey-Fuller Test:')

    dftest = adfuller(timeseries, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    print(dfoutput)



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
ts=sales.groupby(["date_block_num"])["item_cnt_day"].sum()

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

            tmp_aic = tmp_mdl.aic

            if tmp_aic < best_aic:

                best_aic = tmp_aic

                best_order = (i, j)

                best_mdl = tmp_mdl

        except: continue





print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))

# adding the dates to the Time-series as index

ts=sales.groupby(["date_block_num"])["item_cnt_day"].sum()

ts.index=pd.date_range(start = '2013-01-01',end='2015-10-01', freq = 'MS')

ts=ts.reset_index()

ts.head(35)
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