# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings('ignore')



import matplotlib.pyplot as plt

import seaborn as sns

import datetime

import random



import plotly.graph_objs as go

import plotly.offline as py



from matplotlib import rcParams

rcParams['figure.figsize'] = 15,5



from sklearn.cluster import KMeans



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# DONT COPY BELOW CODE

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Data Preparation
#Reading the Global Superstore Data

df1 = pd.read_csv('/kaggle/input/global-superstore/GlobalSuperstoreData.csv')

                  
#View the data from File

df1.head()
# Find the number of rows and columns

df1.shape
# Display the Column header

df1.columns
# Find the Data types of each column

df1.dtypes
# Visulaize the data

sns.pairplot(df1)



plt.show()
#Check for Nulls

df1.isnull().sum(axis=0)

#Describe the data

df1.describe()

# FInd out unique values in market and segment

df1.nunique()



#Converting Order Date column from object to datetime 

#df1['Order Date'] = pd.to_datetime(df1['Order Date'])



df1['Order Date'] = pd.to_datetime(df1['Order Date']).dt.strftime('%m/%Y')



#Shows the unique values in particular column

df1.Market.unique()

#categories, in terms of sales and orders

categories = df1.groupby(['Market','Segment'],as_index=False).sum().sort_values(by='Market',ascending=True)



categories
#Combining Market and Segment into single Column

df2 = df1



df2['Market-Segment'] = df2["Market"] +"-"+ df2["Segment"]

del df2['Segment']

del df2['Market']

df2
# Group by Order date and Market-Segment

categories2 = df2.groupby(['Order Date','Market-Segment'],as_index=False).sum().sort_values(by=['Market-Segment'],ascending=True)



categories2
# Group by Market-Segment to come with 21 unique rows

categories3 = df2.groupby(['Market-Segment'],as_index=False).sum().sort_values(by=['Profit'],ascending=False)



categories3['Market-Segment']
#from sklearn.model_selection import train_test_split
# Peform Train and Test Split

import sklearn.model_selection as model_selection

from sklearn.model_selection import train_test_split 

X=categories3['Sales']

y=categories3['Profit']

X_train, X_test,y_train,  y_test = model_selection.train_test_split(X, y, train_size=0.80,test_size=0.20, random_state=0)

print ("X_train: ", X_train)

print ("y_train: ", y_train)

print("X_test: ", X_test)

print ("y_test: ", y_test)



X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.5, test_size=0.25, random_state=0)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
df5 = pd.concat([X_train,y_train], axis =1 , sort = False)

df5

# Find Cov

CoV = categories3.std(axis =1)/categories3.mean(axis =1)

CoV
# Plot the data

corrmat=categories3.corr()

top_corr=corrmat.index

plt.figure(figsize=(15,15))

#plot the heatmap

g=sns.heatmap(categories3[top_corr].corr(),annot=True,cmap='RdYlGn')
## Extracting Canada-Consumer as indifed as most profitable from Cov
## Extracting Canada-Consumer as indifed as most profitable from Cov

df6 = df2[df2['Market-Segment'] == 'Canada-Consumer']

df6
# TRain Test split for the profitable segment

X=df6['Market-Segment']

y=df6[['Order Date','Sales','Profit']]

X_train, X_test,y_train,  y_test = model_selection.train_test_split(X, y, train_size=0.80,test_size=0.20, random_state=0)

print ("X_train: ", X_train)

print ("y_train: ", y_train)

print("X_test: ", X_test)

print ("y_test: ", y_test)
# Set Index for data

df6 = df6.set_index('Order Date')

df6.head(12)
# Plot Monthly Sale

df6.plot(figsize=(12, 4))

plt.legend(loc='best')

plt.title('Canada Consumer - Monthly Sale')

plt.show(block=False)
# Profit Mean Imputation

df6 = df6.assign(Profit_Mean_Imputation=df6.Profit.fillna(df6.Profit.mean()))



df6[['Profit_Mean_Imputation']].plot(figsize=(12, 4))

plt.legend(loc='best')

plt.title('Canada Consumer - Profit: Mean imputation')

plt.show(block=False)
# Linear Interpolation

df6 = df6.assign(Profit_Linear_Interpolation=df6.Profit.interpolate(method='linear'))

df6[['Profit_Linear_Interpolation']].plot(figsize=(12, 4))

plt.legend(loc='best')

plt.title('Canada Consumer - Profit: Linear interpolation')

plt.show(block=False)
df6.drop(columns=['Profit_Mean_Imputation','Profit_Linear_Interpolation'],inplace=True)
import seaborn as sns

fig = plt.subplots(figsize=(12, 2))

ax = sns.boxplot(x=df6['Profit'],whis=1.5)
fig = df6.Profit.hist(figsize = (12,4))
# Time Series Decompostion

from pylab import rcParams

import statsmodels.api as sm

rcParams['figure.figsize'] = 12, 8

decomposition = sm.tsa.seasonal_decompose(df6.Profit, model='additive', freq = 1) # additive seasonal index

fig = decomposition.plot()

plt.show()
decomposition = sm.tsa.seasonal_decompose(df6.Sales, model='multiplicative', freq = 1) # multiplicative seasonal index

fig = decomposition.plot()

plt.show()
df7 = df6.groupby(['Order Date']).sum().sort_values(by=['Order Date'],ascending=True)



df7.info()

df7
#Model Building
# Since we received only 43 months of data for CANADA-CONSUMER, dividing train and test for 30 and 13 

train_len = 30

train = df7[0:train_len]

test = df7[train_len:] 
y_hat_naive = test.copy()

y_hat_naive['naive_forecast'] = train['Sales'][train_len-1]
# Perform Naive Method

plt.figure(figsize=(12,4))

plt.plot(train['Sales'], label='Train')

plt.plot(test['Sales'], label='Test')

plt.plot(y_hat_naive['naive_forecast'], label='Naive forecast')

plt.legend(loc='best')

plt.title('Naive Method')

plt.show()
#Calculate RMSE and MAPE for Naive method

from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(test['Sales'], y_hat_naive['naive_forecast'])).round(2)

mape = np.round(np.mean(np.abs(test['Sales']-y_hat_naive['naive_forecast'])/test['Sales'])*100,2)



results = pd.DataFrame({'Method':['Naive method'], 'MAPE': [mape], 'RMSE': [rmse]})

results = results[['Method', 'RMSE', 'MAPE']]

results
y_hat_avg = test.copy()

y_hat_avg['avg_forecast'] = train['Sales'].mean()
# Perform Simple Average Method

plt.figure(figsize=(12,4))

plt.plot(train['Sales'], label='Train')

plt.plot(test['Sales'], label='Test')

plt.plot(y_hat_avg['avg_forecast'], label='Simple average forecast')

plt.legend(loc='best')

plt.title('Simple Average Method')

plt.show()
#Calculate RMSE and MAPE for Simple Average Method

rmse = np.sqrt(mean_squared_error(test['Sales'], y_hat_avg['avg_forecast'])).round(2)

mape = np.round(np.mean(np.abs(test['Sales']-y_hat_avg['avg_forecast'])/test['Sales'])*100,2)



tempResults = pd.DataFrame({'Method':['Simple average method'], 'RMSE': [rmse],'MAPE': [mape] })

results = pd.concat([results, tempResults])

results = results[['Method', 'RMSE', 'MAPE']]

results
y_hat_sma = test.copy()

ma_window = 12

y_hat_sma['sma_forecast'] = test['Sales'].rolling(ma_window).mean()

#y_hat_sma['sma_forecast'][train_len:] = y_hat_sma['sma_forecast'][train_len-1]

#test['Sales']
y_hat_sma['sma_forecast'] = y_hat_sma['sma_forecast'].fillna("0").astype(int)

# Perform Simple Moving Average Method

plt.figure(figsize=(12,4))

plt.plot(train['Sales'], label='Train')

plt.plot(test['Sales'], label='Test')

plt.plot(y_hat_sma['sma_forecast'], label='Simple moving average forecast')

plt.legend(loc='best')

plt.title('Simple Moving Average Method')

plt.show()
#Calculate RMSE and MAPE for Simple moving Average Method



rmse = np.sqrt(mean_squared_error(test['Sales'], y_hat_sma['sma_forecast'])).round(2)

mape = np.round(np.mean(np.abs(test['Sales']-y_hat_sma['sma_forecast'])/test['Sales'])*100,2)



tempResults = pd.DataFrame({'Method':['Simple moving average forecast'], 'RMSE': [rmse],'MAPE': [mape] })

results = pd.concat([results, tempResults])

results = results[['Method', 'RMSE', 'MAPE']]

results
#Simple exponential smoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

model = SimpleExpSmoothing(train['Sales'])

model_fit = model.fit(smoothing_level=1,optimized=False)

model_fit.params

y_hat_ses = test.copy()

y_hat_ses['ses_forecast'] = model_fit.forecast(13)
#Plot train, test and forecast
# Perform Simple Exponential Smoothing Method

plt.figure(figsize=(12,4))

plt.plot(train['Sales'], label='Train')

plt.plot(test['Sales'], label='Test')

plt.plot(y_hat_ses['ses_forecast'], label='Simple exponential smoothing forecast')

plt.legend(loc='best')

plt.title('Simple Exponential Smoothing Method')

plt.show()
#Calculate RMSE and MAPE
y_hat_ses['ses_forecast'] = y_hat_ses['ses_forecast'].fillna("0").astype(int)
#Calculate RMSE and MAPE for Simple Exponential Smoothing Forecast



rmse = np.sqrt(mean_squared_error(test['Sales'], y_hat_ses['ses_forecast'])).round(2)

mape = np.round(np.mean(np.abs(test['Sales']-y_hat_ses['ses_forecast'])/test['Sales'])*100,2)



tempResults = pd.DataFrame({'Method':['Simple exponential smoothing forecast'], 'RMSE': [rmse],'MAPE': [mape] })

results = pd.concat([results, tempResults])

results
#Holt's method with trend
from statsmodels.tsa.holtwinters import ExponentialSmoothing

model = ExponentialSmoothing(np.asarray(train['Sales']) ,seasonal_periods=12 ,trend='additive', seasonal=None)

model_fit = model.fit(smoothing_level=0.2, smoothing_slope=0.01, optimized=False)

print(model_fit.params)

y_hat_holt = test.copy()

y_hat_holt['holt_forecast'] = model_fit.forecast(len(test))
plt.figure(figsize=(12,4))

plt.plot( train['Sales'], label='Train')

plt.plot(test['Sales'], label='Test')

plt.plot(y_hat_holt['holt_forecast'], label='Holt\'s exponential smoothing forecast')

plt.legend(loc='best')

plt.title('Holt\'s Exponential Smoothing Method')

plt.show()
rmse = np.sqrt(mean_squared_error(test['Sales'], y_hat_holt['holt_forecast'])).round(2)

mape = np.round(np.mean(np.abs(test['Sales']-y_hat_holt['holt_forecast'])/test['Sales'])*100,2)



tempResults = pd.DataFrame({'Method':['Holt\'s exponential smoothing method'], 'RMSE': [rmse],'MAPE': [mape] })

results = pd.concat([results, tempResults])

results = results[['Method', 'RMSE', 'MAPE']]

results
#Holt Winters' additive method with trend and seasonality
y_hat_hwa = test.copy()

model = ExponentialSmoothing(np.asarray(train['Sales']) ,seasonal_periods=12 ,trend='add', seasonal='add')

model_fit = model.fit(optimized=True)

print(model_fit.params)

y_hat_hwa['hw_forecast'] = model_fit.forecast(13)
#Plot train, test and forecast
plt.figure(figsize=(12,4))

plt.plot( train['Sales'], label='Train')

plt.plot(test['Sales'], label='Test')

plt.plot(y_hat_hwa['hw_forecast'], label='Holt Winters\'s additive forecast')

plt.legend(loc='best')

plt.title('Holt Winters\' Additive Method')

plt.show()
#Calculate RMSE and MAPE
rmse = np.sqrt(mean_squared_error(test['Sales'], y_hat_hwa['hw_forecast'])).round(2)

mape = np.round(np.mean(np.abs(test['Sales']-y_hat_hwa['hw_forecast'])/test['Sales'])*100,2)



tempResults = pd.DataFrame({'Method':['Holt Winters\' additive method'], 'RMSE': [rmse],'MAPE': [mape] })

results = pd.concat([results, tempResults])

results = results[['Method', 'RMSE', 'MAPE']]

results
#Holt Winter's multiplicative method with trend and seasonality
y_hat_hwm = test.copy()

model = ExponentialSmoothing(np.asarray(train['Sales']) ,seasonal_periods=12 ,trend='add', seasonal='mul')

model_fit = model.fit(optimized=True)

print(model_fit.params)

y_hat_hwm['hw_forecast'] = model_fit.forecast(13)
#Plot train, test and forecast
plt.figure(figsize=(12,4))

plt.plot( train['Sales'], label='Train')

plt.plot(test['Sales'], label='Test')

plt.plot(y_hat_hwm['hw_forecast'], label='Holt Winters\'s mulitplicative forecast')

plt.legend(loc='best')

plt.title('Holt Winters\' Mulitplicative Method')

plt.show()
# Calculate RMSE and MAPE
rmse = np.sqrt(mean_squared_error(test['Sales'], y_hat_hwm['hw_forecast'])).round(2)

mape = np.round(np.mean(np.abs(test['Sales']-y_hat_hwm['hw_forecast'])/test['Sales'])*100,2)



tempResults = pd.DataFrame({'Method':['Holt Winters\' multiplicative method'], 'RMSE': [rmse],'MAPE': [mape] })

results = pd.concat([results, tempResults])

results = results[['Method', 'RMSE', 'MAPE']]

results
#Log Scale Transformation ¶

from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.stattools import acf, pacf

from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.arima_model import ARIMA

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 10, 6
indexedDataset_logScale = np.log(df7)

plt.plot( indexedDataset_logScale['Sales'], label='Sales')

plt.plot(indexedDataset_logScale['Profit'], label='Profit')

plt.legend(loc='best')

plt.plot(indexedDataset_logScale)
#The below transformation is required to make series stationary

movingAverage = indexedDataset_logScale.rolling(window=12).mean()

movingSTD = indexedDataset_logScale.rolling(window=12).std()

plt.plot(indexedDataset_logScale)

plt.plot(movingAverage, color='red')
datasetLogScaleMinusMovingAverage = indexedDataset_logScale - movingAverage

datasetLogScaleMinusMovingAverage.head(12)



#Remove NAN values

datasetLogScaleMinusMovingAverage.dropna(inplace=True)

datasetLogScaleMinusMovingAverage.head(10)
def test_stationarity(timeseries):

    

    #Determine rolling statistics

    movingAverage = timeseries.rolling(window=12).mean()

    movingSTD = timeseries.rolling(window=12).std()

    

    #Plot rolling statistics

    orig = plt.plot(timeseries, color='blue', label='Original')

    mean = plt.plot(movingAverage, color='red', label='Rolling Mean')

    std = plt.plot(movingSTD, color='black', label='Rolling Std')

    plt.legend(loc='best')

    plt.title('Rolling Mean & Standard Deviation')

    plt.show(block=False)



    

    #Perform Dickey–Fuller test:

    print('Results of Dickey Fuller Test:')

    dftest = adfuller(timeseries['Sales'], autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    print(dfoutput)

    
test_stationarity(datasetLogScaleMinusMovingAverage)
#Exponential Decay Transformation 
exponentialDecayWeightedAverage = indexedDataset_logScale.ewm(halflife=12, min_periods=0, adjust=True).mean()

plt.plot(indexedDataset_logScale)

plt.plot(exponentialDecayWeightedAverage, color='red')
datasetLogDiffShifting = indexedDataset_logScale - indexedDataset_logScale.shift()
#AR Model

#making order=(2,1,0) gives RSS=1.5023

model = ARIMA(indexedDataset_logScale['Sales'], order=(2,1,0))

results_AR = model.fit(disp=-1)

plt.plot(datasetLogDiffShifting['Sales'], color='black',label='Sales')

plt.plot(datasetLogDiffShifting['Profit'], color='green',label='Profit')

plt.plot(results_AR.fittedvalues, color='red',label='AR Model')

#std = plt.plot(movingSTD, color='black', label='Rolling Std')

plt.legend(loc='best')

plt.title('Plotting AR model')
#MA Model

model = ARIMA(indexedDataset_logScale['Sales'], order=(0,1,2))

results_MA = model.fit(disp=-1)

plt.plot(datasetLogDiffShifting['Sales'], color='black',label='Sales')

plt.plot(datasetLogDiffShifting['Profit'], color='green',label='Profit')

plt.plot(results_MA.fittedvalues, color='red',label='MA Model')

plt.title('Plotting MA model')

plt.legend(loc='best')



# AR+I+MA = ARIMA model

model = ARIMA(indexedDataset_logScale['Sales'], order=(2,1,0))

results_ARIMA = model.fit(disp=-1)

plt.plot(datasetLogDiffShifting['Sales'], color='black',label='Sales')

plt.plot(datasetLogDiffShifting['Profit'], color='green',label='Profit')

plt.plot(results_ARIMA.fittedvalues, color='red',label='ARIMA Model')

plt.title('Plotting ARIMA model')



plt.legend(loc='best')

#Prediction & Reverse transformations
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)

print(predictions_ARIMA_diff.head())
#Convert to cumulative sum

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()

print(predictions_ARIMA_diff_cumsum)
predictions_ARIMA_log = pd.Series(indexedDataset_logScale['Sales'].iloc[0], index=indexedDataset_logScale.index)

predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)

predictions_ARIMA_log.head()
# Inverse of log is exp.

predictions_ARIMA = np.exp(predictions_ARIMA_log)

plt.plot(df7)

plt.plot(predictions_ARIMA)

plt.plot(df7['Sales'], color='black',label='Sales')

plt.plot(df7['Profit'], color='green',label='Profit')

plt.plot(predictions_ARIMA, color='red',label='predictions_ARIMA')



plt.legend(loc='best')
results_ARIMA.plot_predict(1,60) 