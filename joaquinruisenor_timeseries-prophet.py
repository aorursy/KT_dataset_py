import pandas as pd
import numpy as np
import datetime
%matplotlib notebook
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
%matplotlib inline
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import string
from sklearn.metrics import mean_squared_error, mean_absolute_error
df_e = pd.read_csv('../input/hourly-energy-consumption/PJME_hourly.csv')
df_w = pd.read_csv('../input/hourly-energy-consumption/PJMW_hourly.csv')
df_e.shape
df_e['Datetime'] =  pd.to_datetime(df_e['Datetime'])
df_w['Datetime'] =  pd.to_datetime(df_w['Datetime'])

df =  pd.merge(df_e, df_w,  how = 'left', on= 'Datetime') #
df.head()
df['PJMW_MW'] = df['PJMW_MW'].interpolate(method='linear', axis=0).ffill()
df['Energy'] = df['PJME_MW'] + df['PJMW_MW']
df_ = df.copy()
df = df.drop(['PJME_MW','PJMW_MW'], axis = 1)
df.set_index('Datetime', inplace = True) #make the index is out date time
df.head().append(df.tail())
# Plot time series data
df.plot(y=["Energy"], figsize=(15,4))
df[["Energy"]].resample("1w").median().plot(figsize=(15,4))
df[["Energy"]].resample("1m").median().plot(figsize=(15,4))
def test_stationarity(df, ts):
    """
    Test stationarity using moving average statistics and Dickey-Fuller test
    Source: https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/
    """
    
    # Determing rolling statistics
    rolmean = df[ts].rolling(window = 12, center = False).mean()
    rolstd = df[ts].rolling(window = 12, center = False).std()
    
    # Plot rolling statistics:
    orig = plt.plot(df[ts], 
                    color = 'blue', 
                    label = 'Original')
    mean = plt.plot(rolmean, 
                    color = 'red', 
                    label = 'Rolling Mean')
    std = plt.plot(rolstd, 
                   color = 'black', 
                   label = 'Rolling Std')
    plt.legend(loc = 'best')
    plt.title('Rolling Mean & Standard Deviation for %s' %(ts))
    plt.xticks(rotation = 45)
    plt.show(block = False)
    plt.close()
    
    # Perform Dickey-Fuller test:
    # Null Hypothesis (H_0): time series is not stationary
    # Alternate Hypothesis (H_1): time series is stationary
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(df[ts], 
                      autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], 
                         index = ['Test Statistic',
                                  'p-value',
                                  '# Lags Used',
                                  'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
test_stationarity(df = df, ts = 'Energy')
!pip install pystan
!pip install fbprophet
from fbprophet import Prophet
import datetime
from datetime import datetime
split_date = '2017-01-01'
df_train = df.loc[df.index <= split_date].copy()
df_test = df.loc[df.index > split_date].copy()
# Plot train and test so you can see where we have split
df_test \
    .rename(columns={'Energy': 'TEST SET'}) \
    .join(df_train.rename(columns={'Energy': 'TRAINING SET'}),
          how='outer') \
    .plot(figsize=(15,5), title='PJM Total', style='.')
plt.show()
# get relevant data - note: could also try this with ts_log_diff
df_prophet = df[['Energy']] # can try with ts_log_diff

# reset index
df_prophet = df_prophet.reset_index()

# rename columns
df_prophet = df_prophet.rename(columns = {'Datetime': 'ds', 'Energy': 'y'}) # can try with ts_log_diff, this names are set to be named like this

# Change 'ds' type from datetime to date (necessary for FB Prophet)
df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

# Change 'y' type to numeric (necessary for FB Prophet)
df_prophet['y'] = pd.to_numeric(df_prophet['y'], errors='ignore')

model = Prophet(growth = 'linear',daily_seasonality = True)
model.fit(df_prophet)
# Predict on training set with model
df_test_fcst = model.predict(df=df_test.reset_index() \
                                   .rename(columns={'Datetime':'ds'}))
#we could have also used the function:
#future = m.make_future_dataframe(period = 365), this would have forecasted 1 year into the future
# Plot the forecast
f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
fig = model.plot(df_test_fcst,
                 ax=ax)
plt.show()
# Plot the components of the model
fig = model.plot_components(df_test_fcst)
df_test_fcst.head().append(df_test.tail())
import datetime
# Plot Yearly forecasts
f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
ax.scatter(df_test.index, df_test['Energy'], color='r')
ax.set_xlim([datetime.datetime(2017, 1, 1), datetime.datetime(2018, 1, 1)])
fig = model.plot(df_test_fcst, ax=ax)
plot = plt.suptitle('Yeary Prediction')



# Plot monthly forecasts (last month)
f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(15)
ax.scatter(df_test.index, df_test['Energy'], color='r')
ax.set_xlim([datetime.date(2017, 12, 1), datetime.date(2017, 12, 31)])
fig = model.plot(df_test_fcst, ax=ax)
plot = plt.suptitle('December Prediction')


def mean_absolute_percentage_error(y_true, y_pred): 
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

MSE = mean_squared_error(y_true=df_test['Energy'],
                   y_pred=df_test_fcst['yhat'])

MAE = mean_absolute_error(y_true=df_test['Energy'],
                   y_pred=df_test_fcst['yhat'])

MAPE = mean_absolute_percentage_error(y_true=df_test['Energy'],
                   y_pred=df_test_fcst['yhat'])

print('MSE:'+' '+ str(MSE))
print('MAE:' +' '+str(MAE))
print('MAPE:' +' '+ str(MAPE) + '%')
df.to_csv('Energy_PJM.csv',index=True)
df_test_fcst.to_csv('Prophet_Forecast.csv', index = True)
