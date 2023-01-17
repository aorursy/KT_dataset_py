import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns


import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARMA, ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from fbprophet import Prophet

from sklearn.metrics import mean_squared_error
file = pd.read_csv("../input/bigdataset/Datasets-master/daily-max-temperatures.csv")
file.head()
file["Date"] = pd.to_datetime(file["Date"])
file = file.set_index("Date")
file.index

#plotting the data
file.isnull().sum()
file.describe()
file.plot(figsize = (16, 10))
plt.show()
plt.figure(1)
plt.subplot(211)
file["Temperature"].hist()
plt.subplot(212)
file["Temperature"].plot(kind = 'kde')
plt.show()

fig, ax = plt.subplots(figsize = (15, 6))
sns.boxplot(file.index.month, file["Temperature"])
#decomposing the model
plt.rcParams['figure.figsize'] = 16, 8
decomposition = sm.tsa.seasonal_decompose(file["Temperature"], model='multiplicative', period=365)
fig = decomposition.plot()
plt.show()
plt.plot(file)
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

plt.figure()
plt.subplot(211)
plot_acf(file["Temperature"], ax=plt.gca(), lags = 30)
plt.subplot(212)
plot_pacf(file["Temperature"], ax=plt.gca(), lags = 30)
plt.show()
rolmean = file["Temperature"].rolling(window = 12).mean()
rolstd = file["Temperature"].rolling(window = 12).std()

#Plot rolling statistics:
orig = plt.plot(file, color='blue',label='Original')
mean = plt.plot(rolmean, color='red', label='Rolling Mean')
std = plt.plot(rolstd, color='black', label = 'Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
plt.show()

#Perform Dickey-Fuller test:
print ('Results of Dickey-Fuller Test:')
dftest = adfuller(file["Temperature"], autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print (dfoutput)

def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=12).mean()
    rolstd = timeseries.rolling(window=12).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
from statsmodels.tsa.ar_model import AR
from random import random

# fit model
model = AR(file["Temperature"])
model_fit = model.fit()
plt.plot(file["Temperature"])
plt.plot(model_fit.fittedvalues, color='red')
plt.title('RSS: %.4f'% np.nansum((model_fit.fittedvalues-file["Temperature"])**2))
plt.show()
#Building the ARIMA model

#splitting the dataset

train = file[:int(0.75*len(file))]
test = file[train.shape[0]:]

train.shape, test.shape
train["Temperature"].plot()
test["Temperature"].plot()
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

# fit model
model = ARIMA(train, order=(1, 0, 1))
model_fit = model.fit(disp=1)
model_fit.summary()
test
#Predictions
end_index = len(file)
predictions = model_fit.predict(start=2737, end = end_index - 1)
predictions
mse = mean_squared_error(file[train.shape[0]:], predictions)
rmse = sqrt(mse)
print('RMSE: {}, MSE:{}'.format(rmse,mse))
plt.plot(file["Temperature"])
plt.plot(predictions)
#plt.title('RMSE: %.4f'% np.sqrt(np.nansum((predictions-file["Temperature"])**2)/len(file)))
predictions = pd.Series(predictions)
train.head()
train_prophet = pd.DataFrame()
train_prophet['ds'] = train.index
train_prophet['y'] = train["Temperature"].values
from fbprophet import Prophet

#instantiate Prophet with only yearly seasonality as our data is monthly 
model = Prophet( yearly_seasonality=True, seasonality_mode = 'multiplicative')
model.fit(train_prophet) #fit the model with your dataframe
future = model.make_future_dataframe(periods = 913, freq = 'D') 
future.tail()
# now lets make the forecasts
forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

fig = model.plot(forecast)
#plot the predictions for validation set

plt.plot(test, label='Valid', color = 'red', linewidth = 2)

plt.show()
model.plot_components(forecast);

y_prophet = pd.DataFrame()
y_prophet['ds'] = test.index
y_prophet['y'] = test["Temperature"].values
y_prophet = y_prophet.set_index('ds')
forecast_prophet = forecast.set_index('ds')