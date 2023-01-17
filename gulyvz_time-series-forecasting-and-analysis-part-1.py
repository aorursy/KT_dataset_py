import pandas as pd

from datetime import datetime

import numpy as np
# Let's create a pandas series that logs time every hour from 1st Feb'20 to 7th Feb'20

df = pd.date_range(start='2/01/2020', end='2/07/2020', freq='H')

df
len(df)
#Now let's turn our series into a dataframe

df = pd.DataFrame(df, columns=['date'])



# And add a 'made up' column for sales data

df['sales'] = np.random.randint(0,1000,size=(len(df)))

df.head()
# Set your date as the index 

df = df.set_index('date')

df.head()
# Selecting using date - getting exact value for cell 

df.loc['2020-02-01 03:00:00', 'sales']
# Selecting using date to return the row corresponding to that date

df.loc['2020-02-01 03:00:00']
# Selecting an entire day

df.loc['2020-02-01']

# Selecting an entire month

df.loc['2020-02']
# Selecting a range of dates

df.loc['2020-02-01':'2020-02-02']
df.index
df.resample('D').mean()
df.resample('D').sum()
df.resample('W').mean()
df = pd.DataFrame({'year': [2015, 2016],

                   'month': [2, 3],

                   'day': [4, 5]})

df
df.info()
pd.to_datetime(df)
pd.to_datetime('2019-01-01', format='%Y-%m-%d', errors='ignore')
import statsmodels.api as sm

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.date_range(start='2/01/2020', end='2/07/2020', freq='H')

df = pd.DataFrame(df, columns=['date'])

df['sales'] = np.random.randint(0,1000,size=(len(df)))

df = df.set_index('date')

df.plot()
time_series=df['sales']

type(time_series)
time_series.plot()
#Determine rolling statistics

rolmean = df['sales'].rolling(window=24).mean() #window size 24 denotes 24 hour, giving rolling mean at daily level

rolstd = df['sales'].rolling(window=24).std()

print(rolmean,rolstd)
orig = plt.plot(df['sales'], color='blue', label='Original')

mean = plt.plot(rolmean, color='orange', label='Rolling Mean')

std = plt.plot(rolstd, color='black', label='Rolling Std')

plt.legend(loc='best')

plt.title('Rolling Mean & Standard Deviation')

plt.show(block=False);
from statsmodels.tsa.seasonal import seasonal_decompose

decomp= seasonal_decompose(time_series,freq=24)
fig= decomp.plot()

fig.set_size_inches(15,10)
df.head()
from statsmodels.tsa.stattools import adfuller
def adf_check(time_series):

    result = adfuller(time_series)

    print("Augmented Dicky-Fuller Test")

    labels=['Adf Test Statistics', 'p-value', '# of lags', 'Num of Observations used']

    

    for value,label in zip(result,labels):

        print(label+ ":"+str(value))

        

    if result[1]<= 0.05:

        print("Strong evidence against null hypothesis")

        print("reject null hypotesis")

        print("data has no unit root and is stationary")

    else:

        print('weak evidence against null hypothesis')

        print('Fail to reject null hypo')

        print('Data has a unit root, it is a non-stationary')

        

        
adf_check(df['sales'])
#Estimating trend

df_logScale = np.log(df)

plt.plot(df_logScale)
movingAverage = df_logScale.rolling(window=24).mean()

movingSTD = df_logScale.rolling(window=24).std()

plt.plot(df_logScale)

plt.plot(movingAverage, color='red')
datasetLogScaleMinusMovingAverage = df_logScale - movingAverage

datasetLogScaleMinusMovingAverage.head(12)



#Remove NAN values

datasetLogScaleMinusMovingAverage.dropna(inplace=True)

datasetLogScaleMinusMovingAverage.head(10)

adf_check(df_logScale['sales'])
exponentialDecayWeightedAverage = df_logScale.ewm(halflife=12, min_periods=0, adjust=True).mean()

plt.plot(df_logScale)

plt.plot(exponentialDecayWeightedAverage, color='red')



datasetLogScaleMinusExponentialMovingAverage = df_logScale - exponentialDecayWeightedAverage

adf_check(datasetLogScaleMinusExponentialMovingAverage)
df_shift= df['sales'] - df['sales'].shift(1)

df_shift.plot()
df_shift.dropna(inplace=True)

adf_check(df_shift)
adf_check(df_shift.dropna())
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
fig_first= plot_acf(df_shift.dropna())
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df_shift.dropna())
result = plot_pacf(df_shift.dropna())
plot_acf(df_shift.dropna())

plot_pacf(df_shift.dropna());
from statsmodels.tsa.arima_model import ARIMA
help(ARIMA)
model = ARIMA(df_shift, order=(2,1,0))

model_fit = model.fit(disp=0)

print(model_fit.summary())
df_shift
import matplotlib.pyplot as plt

residuals = pd.DataFrame(model_fit.resid)

residuals.plot();

plt.show();

residuals.plot(kind='kde')

plt.show();

print(residuals.describe())
from sklearn.metrics import mean_squared_error

size = int(len(df_shift) * 0.66)

train, test = df_shift[0:size], df_shift[size:len(df_shift)]

history = [x for x in train]

predictions = list()

for t in range(len(test)):

    model = ARIMA(history, order=(5,1,0))

    model_fit = model.fit(disp=0)

    output = model_fit.forecast()

    yhat = output[0]

    predictions.append(yhat)

    obs = test[t]

    history.append(obs)

    print('predicted=%f, expected=%f' % (yhat, obs))

error = mean_squared_error(test, predictions)

print('Test MSE: %.3f' % error)

data = pd.read_csv("../input/AirPassengers.csv")

data['Month'] = pd.to_datetime(data['Month'],infer_datetime_format=True)
df = data.set_index('Month')

df.info()
train = df.iloc[:130]

test = df.iloc[130:]
sarima = sm.tsa.statespace.SARIMAX(train,order=(7,1,7),seasonal_order=(7,1,7,12),enforce_stationarity=False, enforce_invertibility=False).fit()

sarima.summary()
res = sarima.resid

fig,ax = plt.subplots(2,1,figsize=(15,8))

fig = sm.graphics.tsa.plot_acf(res, lags=50, ax=ax[0])

fig = sm.graphics.tsa.plot_pacf(res, lags=50, ax=ax[1])

plt.show()
len(test)
test
from sklearn.metrics import mean_squared_error

pred = sarima.predict(test.index[0],test.index[13])

print('SARIMA model MSE:{}'.format(mean_squared_error(test,pred)))
pred
pd.DataFrame({'test':test['#Passengers'],'pred':pred}).plot();plt.show()
train['passenger_age_mean']= np.random.randint(30,60,size=(len(train)))

test['passenger_age_mean']= np.random.randint(30,60,size=(len(test)))

exog_train = train.passenger_age_mean

exog_test = test.passenger_age_mean



exog_train
train


sarimax = sm.tsa.statespace.SARIMAX(train['#Passengers'],order=(7,1,7),seasonal_order=(1,0,5,12),exog = exog_train,enforce_stationarity=False, enforce_invertibility=False).fit()

sarimax.summary()
res = sarimax.resid

fig,ax = plt.subplots(2,1,figsize=(15,10))

fig = sm.graphics.tsa.plot_acf(res, lags=50, ax=ax[0])

fig = sm.graphics.tsa.plot_pacf(res, lags=50, ax=ax[1])

plt.show()
test


pred = sarima.predict(test.index[0],test.index[13],exog = exog_test)

print('SARIMAX model MSE:{}'.format(mean_squared_error(test['#Passengers'],pred)))
pred
pd.DataFrame({'test':test['#Passengers'],'pred':pred}).plot();plt.show()