import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from statsmodels.tsa.stattools import adfuller,acf, pacf

from statsmodels.tsa.arima_model import ARIMA

import statsmodels.api as sm

from pylab import rcParams
# Set plot size 

rcParams['figure.figsize'] = 10, 6
df = pd.read_csv('https://raw.githubusercontent.com/satishgunjal/datasets/master/Time_Series_AirPassengers.csv')

print('Shape of the data= ', df.shape)

print('Column datatypes= \n',df.dtypes)

df
df['Month'] = pd.to_datetime(df.Month)

df = df.set_index(df.Month)

df.drop('Month', axis = 1, inplace = True)

print('Column datatypes= \n',df.dtypes)

df
plt.figure(figsize= (10,6))

plt.plot(df)

plt.xlabel('Years')

plt.ylabel('No of Air Passengers')

plt.title('Trend of the Time Series')
# To plot the seasonality we are going to create a temp dataframe and add columns for Month and Year values

df_temp = df.copy()

df_temp['Year'] = pd.DatetimeIndex(df_temp.index).year

df_temp['Month'] = pd.DatetimeIndex(df_temp.index).month

# Stacked line plot

plt.figure(figsize=(10,10))

plt.title('Seasonality of the Time Series')

sns.pointplot(x='Month',y='Passengers',hue='Year',data=df_temp)
decomposition = sm.tsa.seasonal_decompose(df, model='additive') 

fig = decomposition.plot()
def stationarity_test(timeseries):

    # Get rolling statistics for window = 12 i.e. yearly statistics

    rolling_mean = timeseries.rolling(window = 12).mean()

    rolling_std = timeseries.rolling(window = 12).std()

    

    # Plot rolling statistic

    plt.figure(figsize= (10,6))

    plt.xlabel('Years')

    plt.ylabel('No of Air Passengers')    

    plt.title('Stationary Test: Rolling Mean and Standard Deviation')

    plt.plot(timeseries, color= 'blue', label= 'Original')

    plt.plot(rolling_mean, color= 'green', label= 'Rolling Mean')

    plt.plot(rolling_std, color= 'red', label= 'Rolling Std')   

    plt.legend()

    plt.show()

    

    # Dickey-Fuller test

    print('Results of Dickey-Fuller Test')

    df_test = adfuller(timeseries)

    df_output = pd.Series(df_test[0:4], index = ['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])

    for key, value in df_test[4].items():

        df_output['Critical Value (%s)' %key] = value

    print(df_output)
# Lets test the stationarity score with original series data

stationarity_test(df)
df_diff = df.diff(periods = 1) # First order differencing

plt.xlabel('Years')

plt.ylabel('No of Air Passengers')    

plt.title('Convert Non Stationary Data to Stationary Data using Differencing ')

plt.plot(df_diff)
df_diff.dropna(inplace = True)# Data transformation may add na values

stationarity_test(df_diff)
df_log = np.log(df)



plt.subplot(211)

plt.plot(df, label= 'Time Series with Variance')

plt.legend()

plt.subplot(212)

plt.plot(df_log, label='Time Series without Variance (Log Transformation)')

plt.legend()  

plt.show()
df_log_diff = df_log.diff(periods = 1) # First order differencing



df_log_diff.dropna(inplace = True)# Data transformation may add na values

stationarity_test(df_log_diff)
df_log_moving_avg = df_log.rolling(window = 12).mean()

plt.xlabel('Years')

plt.ylabel('No of Air Passengers')    

plt.title('Convert Non Stationary Data to Stationary Data using Moving Average')

plt.plot(df_log, color= 'blue', label='Orignal')

plt.plot(df_log_moving_avg, color= 'red', label='Moving Average')

plt.legend()
df_log_moving_avg_diff = df_log - df_log_moving_avg

df_log_moving_avg_diff.dropna(inplace = True)

stationarity_test(df_log_moving_avg_diff)
df_log_weighted_avg = df_log.ewm(halflife = 12).mean()

plt.plot(df_log)

plt.plot(df_log_weighted_avg, color = 'red')
df_log_weighted_avg_diff = df_log - df_log_weighted_avg

stationarity_test(df_log_weighted_avg_diff)
decomposition = sm.tsa.seasonal_decompose(df_log,period =12)

fig = decomposition.plot()
df_log_residual = decomposition.resid

df_log_residual.dropna(inplace = True)

stationarity_test(df_log_residual)
lag_acf = acf(df_log_diff, nlags=20)

lag_pacf = pacf(df_log_diff, nlags=20, method='ols')



# Plot ACF: 

plt.subplot(121) 

plt.plot(lag_acf)

plt.axhline(y=0,linestyle='--',color='gray')

# Draw 95% confidence interval line

plt.axhline(y=-1.96/np.sqrt(len(df_log_diff)),linestyle='--',color='red')

plt.axhline(y=1.96/np.sqrt(len(df_log_diff)),linestyle='--',color='red')

plt.xlabel('Lags')

plt.title('Autocorrelation Function')



#Plot PACF:

plt.subplot(122)

plt.plot(lag_pacf)

plt.axhline(y=0,linestyle='--',color='gray')

# Draw 95% confidence interval line

plt.axhline(y=-1.96/np.sqrt(len(df_log_diff)),linestyle='--',color='red')

plt.axhline(y=1.96/np.sqrt(len(df_log_diff)),linestyle='--',color='red')

plt.xlabel('Lags')

plt.title('Partial Autocorrelation Function')

plt.tight_layout()
# freq = 'MS' > The frequency of the time-series MS = calendar month begin

# The (p,d,q) order of the model for the number of AR parameters, differences, and MA parameters to use

model = ARIMA(df_log, order=(2, 1, 0), freq = 'MS')  

results_AR = model.fit(disp= -1)# If disp < 0 convergence information will not be printed

plt.plot(df_log_diff)

plt.plot(results_AR.fittedvalues, color='red')

plt.title('AR Model, RSS: %.4f'% sum((results_AR.fittedvalues - df_log_diff['Passengers'])**2))
model = ARIMA(df_log, order=(0, 1, 2), freq = 'MS')  

results_MA = model.fit(disp=-1)  

plt.plot(df_log_diff)

plt.plot(results_MA.fittedvalues, color='red')

plt.title('MA Model, RSS: %.4f'% sum((results_MA.fittedvalues-df_log_diff['Passengers'])**2))
model = ARIMA(df_log, order=(2, 1, 2), freq = 'MS')  

results_ARIMA = model.fit(disp=-1)  

plt.plot(df_log_diff)

plt.plot(results_ARIMA.fittedvalues, color='red')

plt.title('Combined Model, RSS: %.4f'% sum((results_ARIMA.fittedvalues-df_log_diff['Passengers'])**2))
# Create a separate series of predicted values

predictions_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)



print('Total no of predictions: ', len(predictions_diff))

predictions_diff.head()
predictions_diff_cumsum = predictions_diff.cumsum()

predictions_diff_cumsum.head()
predictions_log = pd.Series(df_log['Passengers'].iloc[0], index=df_log.index) # Series of base number

predictions_log = predictions_log.add(predictions_diff_cumsum,fill_value=0)

predictions_log.head()
predictions = np.exp(predictions_log)

plt.plot(df)

plt.plot(predictions)
df_predictions =pd.DataFrame(predictions, columns=['Predicted Values'])

pd.concat([df,df_predictions],axis =1).T
results_ARIMA.plot_predict(start = 1, end= 204) 
# Forecasted values in original scale will be

forecast_values_log_scale = results_ARIMA.forecast(steps = 60)

forecast_values_original_scale = np.exp(forecast_values_log_scale[0])



forecast_date_range= pd.date_range("1961-01-01", "1965-12-01", freq="MS")



df_forecast =pd.DataFrame(forecast_values_original_scale, columns=['Forecast'])

df_forecast['Month'] = forecast_date_range



df_forecast[['Month', 'Forecast']]