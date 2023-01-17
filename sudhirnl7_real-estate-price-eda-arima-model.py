import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno

import squarify



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

from sklearn.metrics import mean_squared_error



%matplotlib inline

plt.style.use('ggplot')
#path ='file/'

path = '../input/'

state_ts = pd.read_csv(path+'State_time_series.csv',parse_dates=['Date'])

print('Number of rows and columns in state ts:',state_ts.shape)
msno.bar(state_ts,color='r')
# Analysis

print('Date range:{} to {}'.format(state_ts['Date'].min(),state_ts['Date'].max()))

print('Number of States',state_ts['RegionName'].nunique())
state_month = state_ts.resample('M',on='Date').mean()

state_month = state_month.reset_index()

state_month.shape
# Sample data by region name

state_vise = state_ts.groupby(['RegionName']).mean()

state_vise.shape
data = [go.Scatter(x=state_month['Date'],y = state_month['DaysOnZillow_AllHomes'])]

#layout = {'title': 'Days On Zillow All Homes', 'font': dict(size=16),'xaxis':{'range':['2010-01-01','2017-09-01']}}

layout = dict(

    title='Days On Zillow All Homes',

    xaxis=dict(

        range=['2010-01-01','2017-09-01'],

        rangeselector=dict(

            buttons=list([

                dict(count=1,

                     label='1m',

                     step='month',

                     stepmode='backward'),

                dict(count=6,

                     label='6m',

                     step='month',

                     stepmode='backward'),

                dict(count=1,

                    label='YTD',

                    step='year',

                    stepmode='todate'),

                dict(count=1,

                    label='1y',

                    step='year',

                    stepmode='backward'),

                #dict(step='all')

            ])

        ),

        rangeslider=dict(range=['2010-01-01','2017-09-01']),

        type='date'

    )

)

py.iplot({'data':data,

         'layout': layout})
plt.figure(figsize=(15,15))

k = state_vise['DaysOnZillow_AllHomes'].dropna()

k = k.sort_values(ascending=False)

squarify.plot(sizes = k, label=k.index, color=sns.color_palette('viridis'))

k = state_vise['DaysOnZillow_AllHomes']

k = k[k.isnull()]

k.reset_index().T
data = [go.Scatter(x = state_month['Date'], y = state_month['HomesSoldAsForeclosuresRatio_AllHomes'],name = 'Sold')]

#layout = {'title': 'Home Sold As Foreclosure Ratio of All Homes', 'font': dict(size=16)}

layout = dict(

    title='Home Sold As Foreclosure Ratio of All Homes',

    font= dict(size=16),

    xaxis=dict(

        rangeselector=dict(

            buttons=list([

                dict(count=1,

                     label='1m',

                     step='month',

                     stepmode='backward'),

                dict(count=6,

                     label='6m',

                     step='month',

                     stepmode='backward'),

                dict(count=1,

                    label='YTD',

                    step='year',

                    stepmode='todate'),

                dict(count=1,

                    label='1y',

                    step='year',

                    stepmode='backward'),

                dict(step='all')

            ])

        ),

        rangeslider=dict(),

        type='date'

    )

)

py.iplot({'data':data,'layout': layout})
plt.figure(figsize=(15,15))

k = state_vise['HomesSoldAsForeclosuresRatio_AllHomes'].dropna()

k = k.sort_values(ascending=False)

squarify.plot(sizes=k, label=k.index, color=sns.color_palette('Set1'))

# Missing value

k = state_vise['HomesSoldAsForeclosuresRatio_AllHomes']

k = k[k.isnull()]

k.reset_index().T
data = [go.Scatter(x = state_month['Date'], y = state_month['MedianSoldPrice_AllHomes'], name = 'Sold Price All Home')]

layout = {'title': 'Median Sold Price' }



py.iplot({'data':data,'layout':layout})
plt.figure(figsize=(15,15))

k = state_vise['MedianSoldPrice_AllHomes'].dropna()

k = k.sort_values(ascending=False)

squarify.plot(sizes=k, label=k.index, color=sns.color_palette('viridis'),)

# Missing value

k = state_vise['MedianSoldPrice_AllHomes']

k = k[k.isnull()]

k.reset_index().T
data = [go.Scatter(x = state_month['Date'],y = state_month['PriceToRentRatio_AllHomes'])]

layout = {'title':'Price/Rent All homes','xaxis':{'range':['2010-01-01','2017-12-01']}}

py.iplot({'data':data,'layout':layout})
plt.figure(figsize=(15,15))

k = state_vise['PriceToRentRatio_AllHomes'].dropna()

k = k.sort_values(ascending=False)

squarify.plot(sizes=k, label=k.index, color=sns.color_palette('viridis'),)

# Missing value

k = state_vise['PriceToRentRatio_AllHomes']

k = k[k.isnull()]

k.reset_index().T
data = [go.Scatter(x = state_month['Date'], y = state_month['PctOfHomesDecreasingInValues_AllHomes'],name = 'Decreasing'),

        go.Scatter(x = state_month['Date'], y = state_month['PctOfHomesIncreasingInValues_AllHomes'], name = 'Increasing'),

       ]

       

layout = {'title': 'Percentage Increse vs Decressing Value Of Homes', 'font': dict(size=16),}



py.iplot({'data':data,'layout': layout})
data = [go.Scatter(x = state_month['Date'], y = state_month['PctOfHomesSellingForGain_AllHomes'], name = 'Selling Gain'),

        go.Scatter(x = state_month['Date'], y = state_month['PctOfHomesSellingForLoss_AllHomes'], name = 'Selling Loss'),

       ]

       

layout = {'title': 'Percentage Gain vs Loss Sold of Home ', 'font': dict(size=16),}



py.iplot({'data':data,'layout': layout})
data = [go.Scatter(x = state_month['Date'], y = state_month['MedianListingPricePerSqft_1Bedroom'],name = '1 Bedroom'),

        go.Scatter(x = state_month['Date'], y = state_month['MedianListingPricePerSqft_2Bedroom'], name = '2 Bedroom'),

        go.Scatter(x = state_month['Date'], y = state_month['MedianListingPricePerSqft_3Bedroom'], name = '3 Bedroom'),

        go.Scatter(x = state_month['Date'], y = state_month['MedianListingPricePerSqft_4Bedroom'], name = '4 Bedroom'),

        go.Scatter(x = state_month['Date'], y = state_month['MedianListingPricePerSqft_5BedroomOrMore'], name = '5 or more Bedroom'),

        go.Scatter(x = state_month['Date'], y = state_month['MedianListingPricePerSqft_CondoCoop'], name = 'Condo Coop'),

        go.Scatter(x = state_month['Date'], y = state_month['MedianListingPricePerSqft_DuplexTriplex'], name = 'Duplex Triplex'),

        go.Scatter(x = state_month['Date'], y = state_month['MedianListingPricePerSqft_SingleFamilyResidence'], name = 'Single Family'),

       ]

       

layout = {'title': 'Median Listing Price$/sqft', 'font': dict(size=16),'xaxis':{'range':['2009-01-01','2017-10-01']}}



py.iplot({'data':data,'layout': layout})
data = [go.Scatter(x = state_month['Date'], y = state_month['ZRI_AllHomes'], name = 'All Home')   

       ]

layout = {'title': 'ZRI', 'font':{'size':16},'xaxis':{'range':['2010-01-01','2017-10-01']}}

py.iplot({'data':data,'layout':layout})
data = [go.Scatter(x = state_month['Date'], y = state_month['MedianRentalPricePerSqft_1Bedroom'],name = '1 Bedroom'),

        go.Scatter(x = state_month['Date'], y = state_month['MedianRentalPricePerSqft_2Bedroom'], name = '2 Bedroom'),

        go.Scatter(x = state_month['Date'], y = state_month['MedianRentalPricePerSqft_3Bedroom'], name = '3 Bedroom'),

        go.Scatter(x = state_month['Date'], y = state_month['MedianRentalPricePerSqft_4Bedroom'], name = '4 Bedroom'),

        go.Scatter(x = state_month['Date'], y = state_month['MedianRentalPricePerSqft_5BedroomOrMore'], name = '5 or more Bedroom'),

        go.Scatter(x = state_month['Date'], y = state_month['MedianRentalPricePerSqft_CondoCoop'], name = 'Condo Coop'),

        go.Scatter(x = state_month['Date'], y = state_month['MedianRentalPricePerSqft_DuplexTriplex'], name = 'Duplex Triplex'),

        go.Scatter(x = state_month['Date'], y = state_month['MedianRentalPricePerSqft_SingleFamilyResidence'], name = 'Single Family'),

        go.Scatter(x = state_month['Date'], y = state_month['MedianRentalPricePerSqft_Studio'], name = 'Studio'),

       ]

       

layout = {'title': 'Rental Price Home per Square foot', 'font': dict(size=16),'xaxis':{'range':['2009-01-01','2017-10-01']}}



py.iplot({'data':data,'layout': layout})
data = [go.Scatter(x = state_month['Date'],y = state_month['ZHVIPerSqft_AllHomes'])]

layout = {'title':'Median of the value of all homes per square foot',}

py.iplot({'data':data,'layout':layout})
from statsmodels.tsa.stattools import adfuller,acf,pacf

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.seasonal import seasonal_decompose

from pandas.plotting import autocorrelation_plot
#state_ts['Date'] = pd.datetime(state_ts['Date'])

state_ts = state_ts.set_index('Date')

ts = state_ts['MedianSoldPrice_AllHomes']

ts.head()
plt.figure(figsize=(14,4))

ts.plot()
# Resample data by monthly

plt.figure(figsize=(14,4))

ts = ts.resample('M').mean()

ts.plot()
# forward fill for nan values

ts = ts.ffill()
def test_stationarity(timeseries):

    

    #rolling statics

    rol_mean = timeseries.rolling(window = 12).mean()

    rol_std = timeseries.rolling(window = 12).std()

    

    #plot rolling statistics

    plt.figure(figsize=(14,4))

    plt.plot(ts, color = 'b', label = 'Original')

    plt.plot(rol_mean, color = 'r', label = 'Rolling Mean')

    plt.plot(rol_std, color = 'g', label = 'Rolling Std')

    plt.legend(loc='best')

    

    # Dickey fuller test

    print('Perfom Dickey fuller test')

    dftest = adfuller(timeseries, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    print(dfoutput)

    

test_stationarity(ts)
fig,ax = plt.subplots(1,2,figsize=(14,5))

ax1, ax2 = ax.flatten()



ts_log = np.log(ts)

ts_log.plot(ax=ax1, label = 'Log',color = 'r')

ax1.legend(loc = 'best')



ts_ma = ts_log.rolling(12).mean()

ts_ma.plot(ax = ax2, label = 'mean')

ax2.legend(loc = 'best')

plt.figure(figsize=(14,4))

ts_dif = ts_ma - ts_log

ts_dif = ts_dif.dropna() # fill na

ts_dif.plot()

test_stationarity(ts_dif)
# Differencing

ts_log_dif = ts_log - ts_log.shift()

plt.figure(figsize=(14,4))

ts_log_dif.plot()
ts_log_dif.dropna(inplace = True)

test_stationarity(ts_log_dif)
# Decomposing

docom = seasonal_decompose(ts_dif)

fig,ax = plt.subplots(3,1,figsize=(14,8))

ax

ax[0].plot(docom.resid,label = 'Residual', color = 'r')

ax[0].legend(loc= 'best')

ax[1].plot(docom.seasonal, label = 'Seasonal', color = 'b')

ax[1].legend(loc = 'best')

ax[2].plot(docom.trend,  label = 'Trend', color = 'b')

ax[2].legend(loc = 'best')
test_stationarity(docom.resid.dropna())
# ACF

lag_acf = acf(ts_dif,nlags=20)

#PACF

lag_pacf = pacf(ts_dif, nlags=20, method='ols')
fig,ax = plt.subplots(1,2, figsize=(14,4))

ax1, ax2 = ax.flatten()

ax1.plot(lag_acf)

ax1.axhline(y=0,linestyle='--',color= 'gray')

ax1.axhline(y= - 1.96/np.sqrt(len(ts_dif)), linestyle='--',color= 'gray')

ax1.axhline(y=  1.96/np.sqrt(len(ts_dif)), linestyle='--',color= 'gray')



ax2.plot(lag_pacf,)

ax2.axhline(y=0,linestyle = '--', color = 'gray')

ax2.axhline(y = -1.96/np.sqrt(len(ts_dif)), linestyle = '--', color = 'gray')

ax2.axhline(y = 1.96/np.sqrt(len(ts_dif)), linestyle = '--', color = 'gray')
model = ARIMA(ts_dif, order = (5,1,1))

model_fit = model.fit(disp=5)

print(model_fit.summary())

plt.figure(figsize=(14,4))

plt.plot(ts_dif)

plt.plot(model_fit.fittedvalues,color = 'r')
#Evaluate arima model for (p,d,q)

def evaluate_arima_model(X, arima_order):

    # prepare training dataset

    train_size = int(len(X) * 0.66)

    train, test = X[0:train_size], X[train_size:]

    history = [x for x in train]

    # make predictions

    predictions = list()

    for t in range(len(test)):

        model = ARIMA(history, order=arima_order)

        model_fit = model.fit(disp=0)

        yhat = model_fit.forecast()[0]

        predictions.append(yhat)

        history.append(test[t])

    # calculate out of sample error

    error = mean_squared_error(test, predictions)

    return error
def evaluate_models(dataset, p_values, d_values, q_values):

    dataset = dataset.astype('float32')

    best_score, best_cfg = float("inf"), None

    for p in p_values:

        for d in d_values:

            for q in q_values:

                order = (p,d,q)

                try:

                    mse = evaluate_arima_model(dataset, order)

                    if mse < best_score:

                        best_score, best_cfg = mse, order

                    print('ARIMA%s MSE=%.3f' % (order,mse))

                except:

                    continue

    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
# Evaluate parameter

p_value = range(0,2)

d_value = range(0,2)

q_value = range(0,2)

evaluate_models(ts,p_value,d_value,q_value,) 