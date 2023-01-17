# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

!pip install pmdarima

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import seaborn as sns

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import itertools

import warnings



# from statsmodels.tsa.holtwinters import ExponentialSmoothing

from statsmodels.tsa.stattools import adfuller 

from matplotlib.pylab import rcParams

rcParams['figure.figsize']=25, 6

%matplotlib inline

import statsmodels.api as sm

from statsmodels.tsa.seasonal import seasonal_decompose

import statsmodels.tsa.api as smt

from datetime import datetime

from statsmodels.tsa.stattools import acovf, acf,pacf, pacf_yw, pacf_ols

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  

from statsmodels.tsa.arima_model import ARIMA

from pandas import DataFrame

from statsmodels.tsa.stattools import adfuller

# Load specific forecasting tools

from statsmodels.tsa.arima_model import ARMA,ARMAResults,ARIMA,ARIMAResults

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf # for determining (p,q) orders

from pmdarima import auto_arima # for determining ARIMA orders

from statsmodels.tsa.statespace.sarimax import SARIMAX

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import matplotlib

plt.style.use('ggplot')

import warnings

import itertools

from sklearn import preprocessing

from keras import backend as K

from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras.layers import Dense, LSTM, GRU, Dropout, BatchNormalization

from keras.models import Sequential

from keras.optimizers import RMSprop,Adam

from keras import regularizers

from sklearn.preprocessing import MinMaxScaler

from keras.preprocessing.sequence import TimeseriesGenerator

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

df = df.sort_values(by='ObservationDate', ascending=True)
df
df.columns
#Confirmed Covid-19 Time Series Analysis

dfc = df[['ObservationDate', 'Confirmed','Deaths', 'Recovered']]

# Add a "date" datetime column

dfc['Date'] = dfc['ObservationDate']

dfc['Date'] = dfc['Date'].astype('datetime64')

# Set "date" to be the index

cases = dfc.set_index('Date')

cases.index
cases.plot(figsize=(23,4),lw=8)
cases.plot.line(y=['Confirmed','Recovered'],figsize=(23,4),lw=4).set_title('Confirmed VS Recovered EVOLUTION')

cases.plot.line(y=['Deaths'],figsize=(23,4),lw=4,color='red').set_title('Deaths EVOLUTION')
cases.plot.scatter(x='Confirmed',y='Deaths',s=cases['Recovered']*0.01,c='Recovered',cmap='coolwarm' , alpha=0.5,figsize=(25,5)).set_title('Deaths VS Confirmed  Across Recovered EVOLUTION')

cases.plot.scatter(x='Deaths',y='Recovered',s=cases['Confirmed']*0.01, c='Confirmed',cmap='coolwarm', alpha=1.5,figsize=(25,5)).set_title('Recovered VS Deaths Across Confirmed EVOLUTION')

cases.plot.box()
cases.plot.kde(figsize=(20,4))
indexeddf = cases['Confirmed'].groupby(['Date']).mean()

print("Period's Length = {}".format(len(indexeddf)))
indexeddf.head()

indexeddf = cases['Confirmed'].groupby(['Date']).mean()

indexeddf_deaths = cases['Deaths'].groupby(['Date']).mean()

indexeddf_recovered = cases['Recovered'].groupby(['Date']).mean()

plt.figure(figsize=(23, 5))

plt.xlabel("Date")

plt.ylabel("Confirmed")

plt.plot(indexeddf,c='m', lw=5)

plt.plot(indexeddf_deaths, lw=5)

plt.plot(indexeddf_recovered, lw=5)

plt.title('Confirmed Cases Evolution by Day')
fig, axes = plt.subplots(2, 2, sharey=False, sharex=False)

fig.set_figwidth(14)

fig.set_figheight(8)

axes[1][1].plot(indexeddf.index, indexeddf, label='Original')

axes[1][1].plot(indexeddf.index, indexeddf.rolling(window=3).mean(), label='3-Days Rolling Mean')

axes[1][1].set_xlabel("Date")

axes[1][1].set_ylabel("Confirmed")

axes[1][1].set_title("3-Days Moving Average")

axes[1][1].legend(loc='best')

axes[0][0].plot(indexeddf.index, indexeddf, label='Original')

axes[0][0].plot(indexeddf.index, indexeddf.rolling(window=7).mean(), label='7-Days Rolling Mean')

axes[0][0].set_xlabel("Date")

axes[0][0].set_ylabel("Confirmed")

axes[0][0].set_title("7-Days Moving Average")

axes[0][0].legend(loc='best')

axes[0][1].plot(indexeddf.index, indexeddf, label='Original')

axes[0][1].plot(indexeddf.index, indexeddf.rolling(window=14).mean(), label='14-Days Rolling Mean')

axes[0][1].set_xlabel("Date")

axes[0][1].set_ylabel("Confirmed")

axes[0][1].set_title("14-Days Moving Average")

axes[0][1].legend(loc='best')

axes[1][0].plot(indexeddf.index, indexeddf, label='Original')

axes[1][0].plot(indexeddf.index, indexeddf.rolling(window=8).mean(), label='28-Days Rolling Mean')

axes[1][0].set_xlabel("Date")

axes[1][0].set_ylabel("Confirmed")

axes[1][0].set_title("28-Days Moving Average")

axes[1][0].legend(loc='best')



plt.tight_layout()

plt.show()
# Determing rolling statistics

rolmean = indexeddf.rolling(window=7).mean()

rolstd = indexeddf.rolling(window=7).std()
# plot Rolling statistics 

orig = plt.plot(indexeddf, color='blue', label='original')

mean = plt.plot(rolmean, color='red', label='Rollingmean')

std = plt.plot(rolstd, color='black', label='Rolling std')

plt.legend(loc='best')

plt.title ('Rollingmean,Rolling std')

plt.show()
def adf_test(series,title=''):

    """

    Pass in a time series and an optional title, returns an ADF report

    """

    print(f'Augmented Dickey-Fuller Test: {title}')

    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data

    

    labels = ['ADF test statistic','p-value','# lags used','# observations']

    out = pd.Series(result[0:4],index=labels)



    for key,val in result[4].items():

        out[f'critical value ({key})']=val

        

    print(out.to_string())          # .to_string() removes the line "dtype: float64"

    

    if result[1] <= 0.05:

        print("Strong evidence against the null hypothesis")

        print("Reject the null hypothesis")

        print("Data has no unit root and is stationary")

    else:

        print("Weak evidence against the null hypothesis")

        print("Fail to reject the null hypothesis")

        print("Data has a unit root and is non-stationary")
adf_test(df['Confirmed'])
!pip install pmdarima
auto_arima(indexeddf,seasonal=False).summary()
# Set one month for testing

train = indexeddf.iloc[:47]

test = indexeddf.iloc[47:]
model = SARIMAX(indexeddf,order=(1,1,1))

results = model.fit(disp=-1)

results.summary()
# Obtain predicted values

start=len(train)

end=len(train)+len(test)-1

predictions = results.predict(start=start, end=end, dynamic=False).rename('SARIMA(1,1,1) Predictions')
# Plot predictions against known values

title = "CHINA's Confirmed Cases"

ylabel='Confirmed'

xlabel=''



ax = test.plot(legend=True,figsize=(12,6),title=title)

predictions.plot(legend=True)

ax.autoscale(axis='x',tight=True)

ax.set(xlabel=xlabel, ylabel=ylabel);
model = SARIMAX(indexeddf,order=(1,1,1))

results = model.fit()

fcast = results.predict(len(indexeddf),len(indexeddf)+14).rename('SARIMA(1,1,1) Forecast')

fcast.plot()
# Plot predictions against known values

title = 'Predictions VS known Values'

ylabel='Confirmed Cases'

xlabel=''



ax = indexeddf.plot(legend=True,figsize=(20,6),title=title)

fcast.plot(legend=True)

ax.autoscale(axis='x')

ax.set(xlabel=xlabel, ylabel=ylabel);
indexeddf = cases['Confirmed'].groupby(['Date']).mean()

indexeddf.plot(figsize=(15, 6), c='m', lw=4)

plt.show()
d = pd.DataFrame(indexeddf)

d = d.reset_index()
len(d)
train_len = len(d)-11
train = d.iloc[:train_len]

test = d.iloc[train_len:]
train.head()
# change the date column to a datetime format 

train['Date']=pd.to_datetime(train['Date'])

# set the dates as the index of the dataframe, so that it can be treated as a time-series dataframe

train=train.set_index(['Date'])

# check out first 5 samples of the data

train.head()
#Data Preparation

#We apply the MinMax scaler from sklearn

# to normalize data in the (0,1) interval.

sc = MinMaxScaler(feature_range = (0,1))

scaled_train = sc.fit_transform(train)
n_input = 7

n_features=1

generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)
# define model

model = Sequential()

model.add(LSTM(250,activation='relu', input_shape=(n_input, n_features)))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse', metrics=["accuracy"])
scaled_train
model.summary()
# fit model

model.fit_generator(generator,epochs=30)
model.history.history.keys()
loss_per_epoch = model.history.history['loss']

plt.plot(range(len(loss_per_epoch)),loss_per_epoch)
first_eval_batch = scaled_train[-7:]
first_eval_batch = first_eval_batch.reshape((1, n_input, n_features))
test_predictions = []



first_eval_batch = scaled_train[-n_input:]

current_batch = first_eval_batch.reshape((1, n_input, n_features))



for i in range(len(test)):

    

    # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])

    current_pred = model.predict(current_batch)[0]

    

    # store prediction

    test_predictions.append(current_pred) 

    

    # update batch to now include prediction and drop first value

    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
true_predictions = sc.inverse_transform(test_predictions)
test['Predictions'] = true_predictions

test = test.set_index('Date')
test.plot(figsize=(20,4))
test
true_predictions = sc.inverse_transform(test_predictions)
true_predictions
test.plot(figsize=(12,8))
model.save('18_March_Covid19_model.h5')