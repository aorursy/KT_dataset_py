from IPython.display import Image
Image("../input/new-img-23/stt.png")
import warnings
warnings.filterwarnings("ignore")
from pylab import rcParams
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight') 
# Above is a special style template for matplotlib, highly useful for visualizing time series data
%matplotlib inline
import numpy as np
import pandas as pd
import datetime as dt
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())
np.set_printoptions(suppress=True)
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from IPython.display import Image
all_stock = pd.read_csv("../input/stock-time-series-20050101-to-20171231/all_stocks_2006-01-01_to_2018-01-01.csv", 
                        index_col='Date', parse_dates=['Date'])
stock_3_df = all_stock.query('Name == ["AAPL", "MSFT", "INTC"]')
stock_3_df.info()
app_stk = stock_3_df.query('Name == "AAPL"')
ms_stk = stock_3_df.query('Name == "MSFT"')
itl_stk = stock_3_df.query('Name == "INTC"')

# Plotting before normalization
app_stk.Close.plot()
ms_stk.Close.plot()
itl_stk.Close.plot()
plt.legend(['Apple','Microsoft', 'Intel'])
plt.show()
# Normalizing and comparison
# Both stocks start from 100
norm_app_stk = app_stk.Close.div(app_stk.Close.iloc[0]).mul(100)
norm_ms_stk_stk = ms_stk.Close.div(ms_stk.Close.iloc[0]).mul(100)
norm_itl_stk_stk = itl_stk.Close.div(itl_stk.Close.iloc[0]).mul(100)
norm_app_stk.plot()
norm_ms_stk_stk.plot()
norm_itl_stk_stk.plot()
plt.legend(['Apple','Microsoft', 'Intel'])
plt.show()
all_stock = pd.read_csv("../input/stock-time-series-20050101-to-20171231/all_stocks_2006-01-01_to_2018-01-01.csv")
st_3 = all_stock[all_stock['Name'].isin(["AAPL", "MSFT", "INTC"])]

AAPL_trend = st_3[st_3.Name == 'AAPL']
MSFT_trend = st_3[st_3.Name == 'MSFT']
INTC_trend = st_3[st_3.Name == 'INTC']
app = go.Scatter(
        x=AAPL_trend['Date'], 
        y=AAPL_trend['Close'],
        name='Apple',
        line = dict(color = '#1E90FF')
    )

ms = go.Scatter(
        x=MSFT_trend['Date'], 
        y=MSFT_trend['Close'],
        name='Microsoft',
        line = dict(color = '#FF0000')
    )

intl = go.Scatter(
        x=INTC_trend['Date'], 
        y=INTC_trend['Close'],
        name='Intel',
        line = dict(color = '#FFD700')
    )

data = [app, ms, intl]

layout = dict(
    title='Time Series with Rangeslider for every 2 years from 2006 to 2017',
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=2,
                     label='1Yrs',
                     step='year',
                     stepmode='backward'),
                dict(count=4,
                     label='4Yrs',
                     step='year',
                     stepmode='backward'),
                dict(count=6,
                     label='6Yrs',
                     step='year',
                     stepmode='backward'),
                dict(count=8,
                     label='8Yrs',
                     step='year',
                     stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(
            visible = True
        ),
        type='date'
    )
)

fig = dict(data=data, layout=layout)
iplot(fig, filename = "Time Series with Rangeslider")
app_st = AAPL_trend.iloc[:,:-2]

cf.set_config_file(offline=True, world_readable=True, theme='pearl')

app_st.iplot(x='Date',kind='scatter', subplots=True, shape=(4,1), shared_xaxes=True,
            fill = True, title = "Apple Stock w.r.t Open, High, Low and Close price")
ms_st = MSFT_trend.iloc[:,:-2]

cf.set_config_file(offline=True, world_readable=True, theme='pearl')

ms_st.iplot(x='Date',kind='scatter', subplots=True, shape=(4,1), shared_xaxes=True,
            fill = True, title = "Microsoft Stock w.r.t Open, High, Low and Close price")
in_st = INTC_trend.iloc[:,:-2]

cf.set_config_file(offline=True, world_readable=True, theme='pearl')

in_st.iplot(x='Date',kind='scatter', subplots=True, shape=(4,1), shared_xaxes=True,
            fill = True, title = "Intel Stock w.r.t Open, High, Low and Close price")
#Reading the complete Data
st_data = pd.read_csv('../input/stock-time-series-20050101-to-20171231/all_stocks_2006-01-01_to_2018-01-01.csv')
st_data['Date'] = pd.to_datetime(st_data['Date'])

#Creating Seperate dataset for each company stock
app_data = st_data.query('Name == "AAPL"')
ms_data = st_data.query('Name == "MSFT"')
intl_data = st_data.query('Name == "INTC"')

#Creating Time series object for each company stock
app_data = app_data.sort_values(by = 'Date')
app_data = app_data.set_index('Date')
app_ts = app_data['Close']

ms_data = ms_data.sort_values(by = 'Date')
ms_data = ms_data.set_index('Date')
ms_ts = ms_data['Close']

intl_data = intl_data.sort_values(by = 'Date')
intl_data = intl_data.set_index('Date')
intl_ts = intl_data['Close']
#Function to check the stationarity
from statsmodels.tsa.stattools import adfuller

def st_check(timeseries):   
    rolmean = timeseries.rolling(12).mean() ## as month is year divide by 12
    rolstd = timeseries.rolling(12).std()

    #Plot rolling statistics:
    plt.figure(figsize=(20,10))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std  = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
print(st_check(app_ts))
print(st_check(ms_ts))
print(st_check(intl_ts))
#Apple
app_log=np.log(app_ts)
app_log_dif = app_log - app_log.shift()
app_log_dif.dropna(inplace=True)
print(st_check(app_log_dif))
#Microsoft
ms_log=np.log(ms_ts)
ms_log_dif = ms_log - ms_log.shift()
ms_log_dif.dropna(inplace=True)
print(st_check(ms_log_dif))
#Intel
intl_log=np.log(intl_ts)
intl_log_dif = intl_log - intl_log.shift()
intl_log_dif.dropna(inplace=True)
print(st_check(intl_log_dif))
#ACF and PACF for Apple stock
from statsmodels.tsa.stattools import acf,pacf

app_lag_acf = acf(app_log_dif,nlags=20)
app_lag_pacf = pacf(app_log_dif,nlags=20,method='ols')
f, axs = plt.subplots(2,2,figsize=(15,10))

######################### ACF ##########################################

plt.subplot(121)
plt.plot(app_lag_acf)
plt.axhline(y=0,linestyle='--',color='blue')
plt.axhline(y=-1.96/np.sqrt(len(app_lag_acf)),linestyle='--',color='green')
plt.axhline(y=1.96/np.sqrt(len(app_lag_acf)),linestyle='--',color='green')
plt.title('Autocorrelation Function')


######################### PACF ##########################################

plt.subplot(122)
plt.plot(app_lag_pacf)
plt.axhline(y=0,linestyle='--',color='blue')
plt.axhline(y=-1.96/np.sqrt(len(app_lag_pacf)),linestyle='--',color='green')
plt.axhline(y=1.96/np.sqrt(len(app_lag_pacf)),linestyle='--',color='green')
plt.title('partial autocorrelation plot')
plt.show()
#ACF and PACF for Microsoft stock
from statsmodels.tsa.stattools import acf,pacf

ms_lag_acf = acf(ms_log_dif,nlags=20)
ms_lag_pacf = pacf(ms_log_dif,nlags=20,method='ols')
f, axs = plt.subplots(2,2,figsize=(15,10))

######################### ACF ##########################################

plt.subplot(121)
plt.plot(ms_lag_acf)
plt.axhline(y=0,linestyle='--',color='blue')
plt.axhline(y=-1.96/np.sqrt(len(ms_lag_acf)),linestyle='--',color='green')
plt.axhline(y=1.96/np.sqrt(len(ms_lag_acf)),linestyle='--',color='green')
plt.title('Autocorrelation Function')


######################### PACF ##########################################

plt.subplot(122)
plt.plot(ms_lag_pacf)
plt.axhline(y=0,linestyle='--',color='blue')
plt.axhline(y=-1.96/np.sqrt(len(ms_lag_pacf)),linestyle='--',color='green')
plt.axhline(y=1.96/np.sqrt(len(ms_lag_pacf)),linestyle='--',color='green')
plt.title('partial autocorrelation plot')
plt.show()
#ACF and PACF for Intel stock
from statsmodels.tsa.stattools import acf,pacf

intl_lag_acf = acf(intl_log_dif,nlags=20)
intl_lag_pacf = pacf(intl_log_dif,nlags=20,method='ols')
f, axs = plt.subplots(2,2,figsize=(15,10))

######################### ACF ##########################################

plt.subplot(121)
plt.plot(intl_lag_acf)
plt.axhline(y=0,linestyle='--',color='blue')
plt.axhline(y=-1.96/np.sqrt(len(intl_lag_acf)),linestyle='--',color='green')
plt.axhline(y=1.96/np.sqrt(len(intl_lag_acf)),linestyle='--',color='green')
plt.title('Autocorrelation Function')


######################### PACF ##########################################

plt.subplot(122)
plt.plot(intl_lag_pacf)
plt.axhline(y=0,linestyle='--',color='blue')
plt.axhline(y=-1.96/np.sqrt(len(intl_lag_pacf)),linestyle='--',color='green')
plt.axhline(y=1.96/np.sqrt(len(intl_lag_pacf)),linestyle='--',color='green')
plt.title('partial autocorrelation plot')
plt.show()
#Apple

from statsmodels.tsa.arima_model import ARIMA
f, axs = plt.subplots(2,2,figsize=(15,10))

## AR
plt.subplot(131)
model = ARIMA(app_log,order=(2,1,0))
result_AR = model.fit(disp=-1)
plt.plot(app_log_dif)
plt.plot(result_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((result_AR.fittedvalues-app_log_dif)**2))

## MA
plt.subplot(132)
model = ARIMA(app_log,order=(0,1,2))
result_MA = model.fit(disp=-1)
plt.plot(app_log_dif)
plt.plot(result_MA.fittedvalues,color='red')
plt.title('RSS: %.4f'% sum((result_MA.fittedvalues-app_log_dif)**2))
#Microsoft

from statsmodels.tsa.arima_model import ARIMA
f, axs = plt.subplots(2,3,figsize=(15,10))

## AR
plt.subplot(131)
model = ARIMA(ms_log,order=(2,1,0))
result_AR = model.fit(disp=-1)
plt.plot(ms_log_dif)
plt.plot(result_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((result_AR.fittedvalues-ms_log_dif)**2))

## MA
plt.subplot(132)
model = ARIMA(ms_log,order=(0,1,2))
result_MA = model.fit(disp=-1)
plt.plot(ms_log_dif)
plt.plot(result_MA.fittedvalues,color='red')
plt.title('RSS: %.4f'% sum((result_MA.fittedvalues-ms_log_dif)**2))

## ARIMA
plt.subplot(133)
model = ARIMA(ms_log,order=(2,1,2))
result_MA = model.fit(disp=-1)
plt.plot(ms_log_dif)
plt.plot(result_MA.fittedvalues,color='red')
plt.title('RSS: %.4f'% sum((result_MA.fittedvalues-ms_log_dif)**2))
plt.show()
#Intel

from statsmodels.tsa.arima_model import ARIMA
f, axs = plt.subplots(2,3,figsize=(15,10))

## AR
plt.subplot(131)
model = ARIMA(intl_log,order=(2,1,0))
result_AR = model.fit(disp=-1)
plt.plot(intl_log_dif)
plt.plot(result_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((result_AR.fittedvalues-intl_log_dif)**2))

## MA
plt.subplot(132)
model = ARIMA(intl_log,order=(0,1,2))
result_MA = model.fit(disp=-1)
plt.plot(intl_log_dif)
plt.plot(result_MA.fittedvalues,color='red')
plt.title('RSS: %.4f'% sum((result_MA.fittedvalues-intl_log_dif)**2))

## ARIMA
plt.subplot(133)
model = ARIMA(intl_log,order=(2,1,2))
result_MA = model.fit(disp=-1)
plt.plot(intl_log_dif)
plt.plot(result_MA.fittedvalues,color='red')
plt.title('RSS: %.4f'% sum((result_MA.fittedvalues-intl_log_dif)**2))
plt.show()