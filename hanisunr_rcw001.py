# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!ls ../input/*
# Basic packages

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

sales=pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")



# settings

import warnings

warnings.filterwarnings("ignore")



item_cat=pd.read_csv("../input/competitive-data-science-predict-future-sales/item_categories.csv")

item=pd.read_csv("../input/competitive-data-science-predict-future-sales/items.csv")

sub=pd.read_csv("../input/competitive-data-science-predict-future-sales/sample_submission.csv")

shops=pd.read_csv("../input/competitive-data-science-predict-future-sales/shops.csv")

test=pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")
#first, we need to format the date column

sales.date=sales.date.apply(lambda x:datetime.datetime.strptime(x, '%d.%m.%Y'))
#let's take a look at the information of sales' data

print(sales.info())
#inspect data on month-level

monthly_sales=sales.groupby(["date_block_num","shop_id","item_id"])[

    "date","item_price","item_cnt_day"].agg({"date":["min",'max'],"item_price":"mean","item_cnt_day":"sum"})
#let's see how it looks

monthly_sales.head(25)
#well nothing special for monthly level analysis, so we want to know the number of items for each catagory

#first we groupby the item according to its id

x=item.groupby(['item_category_id']).count()
#then we sort the values by descending

x=x.sort_values(by='item_id',ascending=False)

#then we make the item id index become a column cause we need it

x=x.iloc[0:10].reset_index()

#good, lets plot the data

plt.figure(figsize=(8,4))

ax= sns.barplot(x.item_category_id, x.item_id, alpha=0.8)

plt.title("Items per Category")

plt.ylabel('# of items', fontsize=12)

plt.xlabel('Category', fontsize=12)

plt.show()
#from the plot, no very useful information could be found

#since the datasets contain lots of date information,let's do time seires analysis

#first, let's see the relationship between time and sales

ts=sales.groupby(["date_block_num"])["item_cnt_day"].sum()

ts.astype('float')

plt.figure(figsize=(16,8))

plt.xlabel("Time")

plt.ylabel("Sales")

plt.plot(ts)
#apparently, the sales went down with time, but there are two peaks in this plot, which implies there are some seasonal repetation, maybe?

#like stock market, we also want to care about the rolling mean number and rolling std number,which could indicate what happened in essence

plt.figure(figsize=(16,6))

plt.plot(ts.rolling(window=12,center=False).mean(),label='Rolling Mean');

plt.plot(ts.rolling(window=12,center=False).std(),label='Rolling sd');

plt.legend();
#nothing useful could observed from the plot

#let's dig more into the seasonal peak

import statsmodels.api as sm

res = sm.tsa.seasonal_decompose(ts.values,freq=12,model="multiplicative")

fig = res.plot()
#ok, then clearly, we confirm our guess, the sales changes seasonally

#you can test the additive model, basically, the results is the similar with the multiplicative model

#there is one important clue we should notice, the plot also implies that the data is stationary, which means the series is time-invariance.

#let's test it

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

#Ok, the results shows that the p-value is not within the threshold 5%, hence, we could conclude that the series is not stationary

#now we need to remove the trend

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
#now it's time to show our work

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
#let's do the test again

test_stationarity(new_ts)
#Perfect, the p-value is within 5%

#Now let's observe, whether our data suits AR model, MA model or ARMA model?

#first let's define the plot

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
#Ok, let's check if the lag is 2

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
#How about MA process

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
#ok, now ARMA model process

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
#ok, let's find how to combine the two models, that is, the order of AR and MA

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

#

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
#let's predict the next values

# adding the dates to the Time-series as index

ts=sales.groupby(["date_block_num"])["item_cnt_day"].sum()

ts.index=pd.date_range(start = '2013-01-01',end='2015-10-01', freq = 'MS')

ts=ts.reset_index()

ts.head()
#now we can use prophet to make the prediction

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
#let's plot it

model.plot(forecast)

model.plot_components(forecast)
