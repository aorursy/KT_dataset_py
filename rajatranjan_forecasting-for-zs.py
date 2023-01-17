# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)
# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/yds_train2018.csv")
test=pd.read_csv("../input/yds_test2018.csv")
train.head()
trainx=train.drop(['S_No','Week','Merchant_ID'],axis=1)
trainx['yearMonth']=trainx['Year'].astype(str)+"-"+trainx['Month'].astype(str)

trainx.yearMonth=pd.to_datetime(trainx.yearMonth,format='%Y-%m')
trainx.drop(['Year','Month'],axis=1,inplace=True)
trainx.head()
countries=list(set(trainx.Country))
products=list(set(trainx.Product_ID))
def smape(y_truth,y_forecasted):
    denominator = (np.abs(y_truth) + np.abs(y_forecasted))
    diff = np.abs(y_truth - y_forecasted) / denominator
    diff[denominator == 0] = 0.0
    print(200 * np.mean(diff))
trainx.info()
X=trainx[(trainx.Country=='Argentina') & (trainx.Product_ID==1)].drop(['Product_ID','Country'],axis=1).set_index('yearMonth').sort_values('yearMonth')
X.head()
X=X.groupby('yearMonth').sum()
X.head()
plt.figure(figsize=(15,9))
plt.plot(X)
x1=X[:30]
y1=X[:10]
#Moving Average  
def MA(df, n):  
    name = 'SMA_' + str(n)
    #MA = pd.Series(pd.rolling_mean(df['Close'], n), name = 'SMA_' + str(n))  
    #df = df.join(MA)  
    df[name]=df['Sales'].rolling(n).mean()
    return df

#Exponential Moving Average  
def EMA(df, n):
    name = 'EMA_' + str(n)
    #MA = pd.Series(pd.rolling_mean(df['Close'], n), name = 'SMA_' + str(n))  
    #df = df.join(MA)  
    df[name]=df['Sales'].ewm(span = n, min_periods = n - 1).mean()
    return df

sdf=X.copy()
for i in [3,5,12]:
    
    MA(sdf,i)
    
for i in [3,5,12]:
    
    EMA(sdf,i)

dates = np.array(sdf.index)
#print(dates)
dates_check = dates[-12:]
dates = dates[:-12]
dates
sdf.fillna( value=0, inplace=True)
sdf.isnull().sum()
sdf.shape
# pick a forecast column
forecast_col = 'Sales'

# Chosing 30 days as number of forecast days
forecast_out = int(12)
print('length =',len(sdf), "and forecast_out =", forecast_out)
# Creating label by shifting 'Sales' according to 'forecast_out'
sdf['label'] = sdf[forecast_col].shift(-forecast_out)
print(sdf.head())
print('\n')
# If we look at the tail, it consists of n(=forecast_out) rows with NAN in Label column 
print(sdf.tail(2))
# Define features Matrix X by excluding the label column which we just created 
from sklearn import preprocessing
Xf = np.array(sdf.drop(['label'], 1))

# Using a feature in sklearn, preposessing to scale features
Xf = preprocessing.scale(Xf)
print(Xf.shape)
X_forecast_out = Xf[-forecast_out:]
Xf = Xf[:-forecast_out]
print ("Length of X_forecast_out:", len(X_forecast_out), "& Length of Xf :", len(Xf))
# A good test is to make sure length of X and y are identical
yf = np.array(sdf['label'].values)
yf = yf[:-forecast_out]
print('Length of yf: ',len(yf))
print(yf)
from sklearn.model_selection import train_test_split
X_trainf, X_testf, y_trainf, y_testf = train_test_split(Xf, yf, test_size = 0.2)

print('length of X_train and x_test: ', len(X_trainf), len(X_testf))
# Train
from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor()
clf.fit(X_trainf,y_trainf)
# Test
accuracy = clf.score(X_testf, y_testf)
print("Accuracy of RF Regression: ", accuracy)
forecast_prediction = clf.predict(X_testf)
smape(y_testf,forecast_prediction)
forecast_predictionmain = clf.predict(X_forecast_out)
#Make the final DataFrame containing Dates, ClosePrices, and Forecast values
actual = pd.DataFrame(dates, columns = ["Date"])
actual["Sales"] =sdf.iloc[:len(dates),0].values
actual["Forecast"] = np.nan
actual.set_index("Date", inplace = True)
forecast = pd.DataFrame(dates_check, columns=["Date"])
print(forecast)
forecast["Forecast"] = forecast_predictionmain
forecast["Sales"] = np.nan
forecast.set_index("Date", inplace = True)
var = [actual, forecast]
result = pd.concat(var)  #This is the final DataFrame
result.info()
#Plot the results
result.plot(figsize=(20,10), linewidth=1.5)
plt.legend(loc=2, prop={'size':20})
plt.xlabel('Date')
plt.ylabel('Price')
series = X
# display first few rows
# line plot of dataset

series.plot()
# seasonal difference
differenced = X.diff(12)
# trim off the first year of empty data
differenced = differenced[12:]
# save differenced dataset to file
# plot differenced dataset
differenced.plot()
X.shape
from statsmodels.graphics.tsaplots import plot_acf
plt.figure(figsize=(15,9))
plot_acf(X)
# reframe as supervised learning
from pandas import DataFrame
dataframe=DataFrame(index=X.index)
for ii in range(12,0,-1):
    dataframe['t-'+str(ii)] = X.shift(ii)
dataframe['t'] = X.values
dataframe.head(13)

dataframe = dataframe[12:]
from sklearn.ensemble import RandomForestRegressor
# load data
dataframe
array = dataframe.values
# split into input and output
XX = array[:,0:-1]
yy = array[:,-1]
# fit random forest model
model = RandomForestRegressor(n_estimators=500, random_state=1)
model.fit(XX, yy)
# show importance scores
print(model.feature_importances_)
# plot importance scores
names = dataframe.columns.values[0:-1]
ticks = [i for i in range(len(names))]
plt.figure(figsize=(15,9))
plt.bar(ticks, model.feature_importances_)
from pandas.tools.plotting import autocorrelation_plot
autocorrelation_plot(X)
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(X, order=(2,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# plot residual errors
residuals = DataFrame(model_fit.resid)
residuals.plot()
residuals.plot(kind='kde')
print(residuals.describe())
size = int(len(X) * 0.66)
trainxx, testxx = X[0:size], X[size:len(X)]
history = [x for x in trainxx.Sales]
predictions = list()
for t in range(len(testxx)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = testxx.Sales[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
error = smape(testxx.Sales.values, predictions)
# plot
plt.figure(figsize=(15,9))
plt.plot(testxx.Sales.values)
plt.plot(predictions, color='red')
import matplotlib.pyplot as plt
%matplotlib inline
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling( window=12).mean()
    rolstd = timeseries.rolling(window=12).std()

    #Plot rolling statistics:
    plt.figure(figsize=(15,9))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries.Sales.values, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
test_stationarity(X)
ts_log = np.log(X)
plt.figure(figsize=(15,9))
plt.plot(ts_log)
moving_avg = ts_log.rolling(12).mean()
plt.figure(figsize=(15,9))
plt.plot(ts_log)
plt.plot(moving_avg, color='red')
ts_log_moving_avg_diff = ts_log - moving_avg

ts_log_moving_avg_diff.dropna(inplace=True)
test_stationarity(ts_log_moving_avg_diff)
expwighted_avg = ts_log.ewm( halflife=12).mean()
plt.figure(figsize=(15,9))
plt.plot(ts_log)
plt.plot(expwighted_avg, color='red')
ts_log_ewma_diff = ts_log - expwighted_avg
test_stationarity(ts_log_ewma_diff)
ts_log_diff = ts_log - ts_log.shift()
plt.figure(figsize=(15,9))
plt.plot(ts_log_diff)
ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff)
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_log)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
plt.figure(figsize=(15,9))
plt.subplot(411)
plt.plot(ts_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
ts_log_decompose = residual
ts_log_decompose.dropna(inplace=True)
test_stationarity(ts_log_decompose)
#ACF and PACF plots:

from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(ts_log_diff, nlags=20)
lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')
#Plot ACF: 
plt.figure(figsize=(15,9))
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')
#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(ts_log, order=(2, 1, 0))  
results_AR = model.fit(disp=-1)  
plt.figure(figsize=(15,9))
plt.plot(ts_log_diff)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues.values-ts_log_diff.Sales.values)**2))
model = ARIMA(ts_log, order=(0, 1, 2))  
results_MA = model.fit(disp=-1)  
plt.figure(figsize=(15,9))
plt.plot(ts_log_diff)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_MA.fittedvalues.values-ts_log_diff.Sales.values)**2))
# model = ARIMA(ts_log, order=(2, 1, 2))  
# results_ARIMA = model.fit(disp=0)  
# plt.figure(figsize=(12,9))
# plt.plot(ts_log_diff)
# plt.plot(results_ARIMA.fittedvalues, color='red')
# plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues.values-ts_log_diff.Sales.values)**2))

predictions_ARIMA_diff = pd.Series(results_MA.fittedvalues, copy=True)
predictions_ARIMA_diff.head()
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_diff_cumsum.head()
predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()
from fbprophet import Prophet
Xd = pd.DataFrame(index=range(0,len(x1)))
Xd['ds'] = x1.index
Xd['y'] = x1['Sales'].values
Xd.head()
m = Prophet()
m.fit(Xd)
future = m.make_future_dataframe(periods=12,freq='MS')
future.tail()
forecast=m.predict(future)
forecast.head()
y_truth = y1['Sales'].values
y_forecasted = forecast.iloc[0:10,-1].values
plt.figure(figsize=(15,9))
m.plot(forecast)
plt.plot(x1,c='r')
# print(y1['2013'])
forecast[forecast['ds']==pd.to_datetime('2013-01-01')]['yhat'].values
print(y_truth.shape)
print(y_forecasted.shape)
smape(y_truth,y_forecasted)
test.head()
testx=test.drop('S_No',axis=1)
testx['yearMonth']=testx['Year'].astype(str)+"-"+testx['Month'].astype(str)
testx.yearMonth=pd.to_datetime(testx.yearMonth,format='%Y-%m')
testx.drop(['Year','Month'],axis=1,inplace=True)
testx.head()
YtestSales = pd.DataFrame(index=range(0,len(testx)))
d={}
dh={}
for c in countries:
    for p in products:
        XX=trainx[(trainx.Country==c) & (trainx.Product_ID==p)].drop(['Product_ID','Country'],axis=1).set_index('yearMonth').sort_values('yearMonth')
        
        XX=XX.groupby('yearMonth').sum()
        XdX = pd.DataFrame(index=range(0,len(XX)))
        XdX['ds'] = XX.index
        XdX['y'] = XX['Sales'].values
        if len(XdX)!=0:
            m = Prophet()
            m.fit(XdX)
            
            future = m.make_future_dataframe(periods=20,freq='MS')
            forecast = m.predict(future)
            print("Done for country "+c+" and Product id "+str(p))
            m.plot(forecast)
            plt.title("For country "+c+" and Product id "+str(p))
            Y=testx[(testx.Country==c) & (testx.Product_ID==p)].drop(['Product_ID','Country'],axis=1).sort_values('yearMonth')
            sale=[]
            saleh=[]
            for j in Y.yearMonth.values:
                sale.append(forecast[forecast['ds']==j]['yhat'].values[0])
                saleh.append(forecast[forecast['ds']==j]['yhat_upper'].values[0])
                
            d[str(c)+"-"+str(p)]=sale
            dh[str(c)+"-"+str(p)]=saleh
sales_tot=np.hstack(d.values())
test.Sales=sales_tot
test.to_csv("yds_submission2018.csv",index=False)
sales_toth=np.hstack(dh.values())
test.Sales=sales_toth
test.to_csv("yds_submission2018h.csv",index=False)
test.to_csv("yds_submission2018.csv",index=False)
