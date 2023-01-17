import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

from IPython.display import Image

%matplotlib inline
Image('../input/alteryx-workflow-for-the-time-series-notebook/alteryx_workflow_time_series_notebook.PNG')
data = pd.read_excel('../input/video-game-sales-data-for-time-series-forecasting/monthly-sales.xlsx')
data.head()
print(data.shape)

print(data.columns)

print(data.isnull().any())
data['Month'] = pd.to_datetime(data['Month'])
data.head()
data.dtypes
data.set_index(data['Month'], drop = True, inplace = True)
data.head()
data.drop(columns = ['Month'], inplace = True)
data.head()
data.index.freq = 'MS'
plt.style.available
plt.figure(figsize = (10,10))

plt.style.use('Solarize_Light2')

plt.plot(data, color = 'brown')

plt.xlabel('Time', fontsize = 14, labelpad = 20)

plt.ylabel('Sales', fontsize = 14, labelpad = 20)

plt.show()
train = data.iloc[0:65]
train.shape
test = data.iloc[65:]
test
series_values = train['Monthly Sales'].values.tolist()

average_method_prediction = []



for x in range(4):

    average = np.mean(series_values)

    average_method_prediction.append(average)

    series_values.append(average)
from sklearn.metrics import mean_squared_error



average_method_mse = mean_squared_error(test['Monthly Sales'], average_method_prediction)

average_method_rmse = average_method_mse ** (1/2)

print(average_method_mse)

print(average_method_rmse)
model_comparison = pd.DataFrame(columns = ['Model Name', 'MSE', 'RMSE'])
model_comparison
model_comparison = model_comparison.append({'Model Name' : 'Average Method', 'MSE' : average_method_mse, 'RMSE' : average_method_rmse}, ignore_index=True)
model_comparison
train.tail()
naive_method_predictions = [231000, 231000, 231000, 231000]



naive_method_mse = mean_squared_error(test['Monthly Sales'], naive_method_predictions)

naive_method_rmse = naive_method_mse ** (1/2)



print(naive_method_mse)

print(naive_method_rmse)
model_comparison = model_comparison.append({'Model Name' : 'Naive Method', 'MSE' : naive_method_mse, 'RMSE' : naive_method_rmse}, ignore_index=True)

model_comparison
plt.figure(figsize = (10,10))

plt.style.use('Solarize_Light2')

plt.plot(data, color = 'brown')

plt.xlabel('Time', fontsize = 14, labelpad = 20)

plt.ylabel('Sales', fontsize = 14, labelpad = 20)

plt.show()
from statsmodels.tsa.seasonal import seasonal_decompose 
decomposition = seasonal_decompose(data)
trend = decomposition.trend

trend.plot(legend = False, figsize = (9,4))

plt.show()
seasonal = decomposition.seasonal

seasonal.plot(legend = False, figsize = (9,4))

plt.show()
residual = decomposition.resid

residual.plot(legend = False, figsize = (9,4))

plt.show()
from statsmodels.tsa.holtwinters import SimpleExpSmoothing



ses_model = SimpleExpSmoothing(train)

#while it is always good to have the model decide the ideal parameters, we are trying out different values to understand better

ses_fit = ses_model.fit(smoothing_level=0.2)

#don't worry about the other parameters

ses_forecast = ses_fit.forecast(4)

ses_forecast
alpha_values = [0.2,0.5,0.7,0.9]



for alpha in alpha_values:

    model = SimpleExpSmoothing(train)

    fit = model.fit(smoothing_level=alpha)

    forecast = fit.forecast(4)

    mse = mean_squared_error(test, forecast)

    rmse = mse ** (1/2)

    

    print('For the value of alpha : {0}, the mse is : {1} and rmse is {2}'.format(alpha,mse,rmse))

    print('--------------------------')

    
model_comparison.columns
model_comparison = model_comparison.append({'Model Name':'Simple Exponential Smoothing', 'MSE': 11174831408.07449, 'RMSE': 105711.07514387737 }, ignore_index = True)

model_comparison
from statsmodels.tsa.holtwinters import Holt



beta_values = [0,0.05,0.2,0.4,0.6,0.8]



for beta in beta_values:

    hlm_model = Holt(train)

    hlm_fit = hlm_model.fit(smoothing_level = 0.25, smoothing_slope = beta)

    hlm_forecast = hlm_fit.forecast(4)

    hlm_mse = mean_squared_error(test, hlm_forecast)

    hlm_rmse = hlm_mse ** (1/2)

    

    print('For the value of beta : {0}, the mse is : {1} and rmse is {2}'.format(beta,hlm_mse,hlm_rmse))

    print('--------------------------')
model_comparison = model_comparison.append({'Model Name':'Holt\'s Linear Method', 'MSE': 10389594905.350863, 'RMSE': 101929.36233171904 }, ignore_index = True)

model_comparison
fig, ax = plt.subplots(3,1, figsize = (17,10))

decomposition.trend.plot(ax = ax[0], legend = False, color = 'red')

decomposition.seasonal.plot(ax = ax[1], legend = False, color = 'green')

decomposition.resid.plot(ax = ax[2], legend = False, color = 'purple')
from statsmodels.tsa.holtwinters import ExponentialSmoothing



hws_model = ExponentialSmoothing(train, trend = 'add' , seasonal = 'add' , seasonal_periods=12)

hws_fit = hws_model.fit()

hws_forecast = hws_fit.forecast(4)

hws_mse = mean_squared_error(test, hws_forecast)

hws_rmse = hws_mse ** (1/2)

hws_rmse
plt.figure(figsize = (16,8))

plt.plot(data, color = 'grey', label = "Actual values")

plt.plot(hws_forecast, color = 'black', linestyle = '--', label = "Forecasted Values", linewidth = 3.5)

plt.legend()

plt.show()
model_comparison = model_comparison.append({'Model Name':'ETS(MAA)', 'MSE': hws_mse, 'RMSE': hws_rmse }, ignore_index = True)

model_comparison
Image('../input/stationary-data-example/stationary_data_example.PNG')
plt.figure(figsize = (16,8))

plt.plot(data, color = 'grey', label = "Actual values")

plt.plot(data.rolling(window = 12).mean(), label = "Rolling Mean", linestyle = '--', color = 'red')

plt.xlabel('Month', fontsize = 14, labelpad = 20)

plt.legend()

plt.show()
#calculating lags

print(data.head())

print(data.shift(periods = 1).head())
from statsmodels.graphics.tsaplots import plot_acf



acf_plot = plot_acf(data, lags = 25)
print(data.head())

print(data.diff(periods = 1).head())
fig, ax = plt.subplots(2,1, figsize = (17,8))

ax[0].plot(data, color = 'grey', label = "Actual values")

ax[0].plot(data.rolling(window = 12).mean(), label = "Rolling Mean", linestyle = '--', color = 'red')



ax[1].plot(data.diff(periods = 1), label = "Differenced data", color = 'green')

ax[1].plot(data.diff(periods = 1).rolling(window = 12).mean(), linestyle = '--', label = "Rolling Mean", color = 'brown')



plt.legend()

plt.show()
from statsmodels.tsa.arima_model import ARIMA

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
train['Sales Diff'] = train.diff(periods = 1)
train.head()
acf_1 = plot_acf(train['Sales Diff'][1:], lags = 20)
pacf_1 = plot_pacf(train['Sales Diff'][1:], lags = 20)
train.tail()
arima_1 = ARIMA(train['Monthly Sales'], order = (1,1,0))

arima_1_fit = arima_1.fit()

arima_1_results = arima_1_fit.forecast(steps = 4)

arima_1_results



#the forecast result from the ARIMA is a little different than the result from an exponential.

#the first array is our forecast results.
arima_1_mse = mean_squared_error(test, arima_1_results[0])

arima_1_rmse = arima_1_mse ** (1/2)

arima_1_rmse
#acf plot for the original dataset 



acf_plot
# we can clearly see the data is not stationary. Let us seasonally difference the data 



train['Seasonal Diff'] = train['Monthly Sales'].diff(periods = 12)

train.head(15)
train_clean = train[['Seasonal Diff']].dropna()
#let us plot the acf and pacf plots using the seasonally differenced data 

fig, ax = plt.subplots(2,1, figsize = (17,8))

acf_seasonal = plot_acf(train_clean, color = 'blue', ax = ax[0], lags = 25)

pacf_seasonal = plot_pacf(train_clean, color = 'red', ax = ax[1], lags = 25)

train_clean.head()
train_clean['First Diff'] = train_clean.diff(periods = 1)

train_clean.head()
fig, ax = plt.subplots(2,1, figsize = (17,8))

acf_seasonal = plot_acf(train_clean['First Diff'][1:], color = 'blue', ax = ax[0], lags = 25)

pacf_seasonal = plot_pacf(train_clean['First Diff'][1:], color = 'red', ax = ax[1], lags = 25)

plt.figure(figsize = (17,8))

plt.plot(train_clean['First Diff'], label = "Differenced data", color = 'black')

plt.plot(train_clean['First Diff'].rolling(window = 12).mean(), linestyle = '--', label = "Rolling Mean", color = 'red')

plt.title('Differenced Data Set (Seasonal and First)', pad = 25, color = 'red')

plt.legend()
from statsmodels.tsa.statespace.sarimax import SARIMAX 



sarima_model = SARIMAX(train['Monthly Sales'], order = (0,1,1), seasonal_order = (0,1,0,12), enforce_invertibility = False, enforce_stationarity = False)

sarima_fit = sarima_model.fit()

sarima_forecast = sarima_fit.forecast(steps = 4 )

sarima_mse = mean_squared_error(test, sarima_forecast)

sarima_rmse = sarima_mse ** (1/2)

print(sarima_rmse)

print(sarima_fit.aic)
model_comparison = model_comparison.append({"Model Name" : "SARIMA(011)(010)12", "MSE" : sarima_mse, "RMSE" : sarima_rmse}, ignore_index = True)

model_comparison
import itertools 



p=d=q=range(0,3)

P=D=Q=range(0,3)



pdq = list(itertools.product(p,d,q))

pdq
PDQ = list(itertools.product(P,D,Q))
PDQ
m = 12

PDQ_12 = [(y[0], y[1], y[2],m) for y in PDQ]

PDQ_12
import warnings

warnings.filterwarnings('ignore')



error_dict = {}

aic_dict = {}



for ns in pdq:

    for s in PDQ_12:

        try:

            sarima_model2 = SARIMAX(train['Monthly Sales'], order = ns, seasonal_order = s, enforce_invertibility = False, enforce_stationarity = False)

            sarima_fit2 = sarima_model2.fit()

            sarima_forecast2 = sarima_fit2.forecast(steps = 4)

            sarima_mse2 = mean_squared_error(test, sarima_forecast2)

            sarima_rmse2 = sarima_mse2 ** (1/2)

            error_dict[ns,s] = sarima_rmse2 

            aic_dict[ns,s] = sarima_fit2.aic

            

        except:

            continue

        
min(aic_dict.items(), key = lambda x : x[1])
sarima_3 = SARIMAX(train['Monthly Sales'], order = (2,2,2), seasonal_order = (2,2,0,12), enforce_invertibility = False, enforce_stationarity = False)

sarima_fit_3 = sarima_3.fit()

sarima_iteration_diag = sarima_fit_3.plot_diagnostics(lags = 12, figsize = (16,7))
sarima_manual_diag = sarima_fit.plot_diagnostics(lags = 12, figsize = (16,7))