#!pip install pmdarima
import pandas as pd

import numpy as np

from pandas import datetime

from matplotlib import pyplot as plt

import os



from statsmodels.tsa.arima_model import ARIMA

from matplotlib import pyplot

from pandas.tools.plotting import autocorrelation_plot



#from pyramid.arima import auto_arima

#from pmdarima.arima import auto_arima

import pyflux as pf

from sklearn.cluster import KMeans

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import statsmodels.api as sm

from statsmodels.tsa.statespace.sarimax import SARIMAX



import math



from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error



from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

# Combining all blocks

for num in range(0,112):

    df = pd.read_csv("../input/daily_dataset/daily_dataset/block_"+str(num)+".csv")

    df = df[['day','LCLid','energy_sum']]

    df.reset_index()

    df.to_csv("hc_"+str(num)+".csv")



fout= open("energy.csv","a")

# first file:

for line in open("hc_0.csv"):

    fout.write(line)

# now the rest:    

for num in range(0,112):

    f = open("hc_"+str(num)+".csv")

    f.readline() # skip the header

    for line in f:

         fout.write(line)

    f.close()

fout.close()
energy = pd.read_csv('energy.csv')

len(energy)
housecount = energy.groupby('day')[['LCLid']].nunique()

housecount.head(10)
housecount.plot(figsize=(25,5))
energy = energy.groupby('day')[['energy_sum']].sum()

energy = energy.merge(housecount, on = ['day'])

energy = energy.reset_index()
energy.count()
energy.day = pd.to_datetime(energy.day,format='%Y-%m-%d').dt.date
energy['avg_energy'] =  energy['energy_sum']/energy['LCLid']

print("Starting Point of Data at Day Level",min(energy.day))

print("Ending Point of Data at Day Level",max(energy.day))
energy.describe()
weather = pd.read_csv('../input/weather_daily_darksky.csv')

weather.head(10)
weather.describe()
weather['day']=  pd.to_datetime(weather['time']) # day is given as timestamp

weather['day']=  pd.to_datetime(weather['day'],format='%Y%m%d').dt.date

# selecting numeric variables

weather = weather[['temperatureMax', 'windBearing', 'dewPoint', 'cloudCover', 'windSpeed',

       'pressure', 'apparentTemperatureHigh', 'visibility', 'humidity',

       'apparentTemperatureLow', 'apparentTemperatureMax', 'uvIndex',

       'temperatureLow', 'temperatureMin', 'temperatureHigh',

       'apparentTemperatureMin', 'moonPhase','day']]

weather = weather.dropna()
weather_energy =  energy.merge(weather,on='day')

weather_energy.head(2)
fig, ax1 = plt.subplots(figsize = (20,5))

ax1.plot(weather_energy.day, weather_energy.temperatureMax, color = 'tab:orange')

ax1.plot(weather_energy.day, weather_energy.temperatureMin, color = 'tab:pink')

ax1.set_ylabel('Temperature')

ax1.legend()

ax2 = ax1.twinx()

ax2.plot(weather_energy.day,weather_energy.avg_energy,color = 'tab:blue')

ax2.set_ylabel('Average Energy/Household',color = 'tab:blue')

ax2.legend(bbox_to_anchor=(0.0, 1.02, 1.0, 0.102))

plt.title('Energy Consumption and Temperature')

fig.tight_layout()

plt.show()

fig, ax1 = plt.subplots(figsize = (20,5))

ax1.plot(weather_energy.day, weather_energy.humidity, color = 'tab:orange')

ax1.set_ylabel('Humidity',color = 'tab:orange')

ax2 = ax1.twinx()

ax2.plot(weather_energy.day,weather_energy.avg_energy,color = 'tab:blue')

ax2.set_ylabel('Average Energy/Household',color = 'tab:blue')

plt.title('Energy Consumption and Humidity')

fig.tight_layout()

plt.show()
fig, ax1 = plt.subplots(figsize = (20,5))

ax1.plot(weather_energy.day, weather_energy.cloudCover, color = 'tab:orange')

ax1.set_ylabel('Cloud Cover',color = 'tab:orange')

ax2 = ax1.twinx()

ax2.plot(weather_energy.day,weather_energy.avg_energy,color = 'tab:blue')

ax2.set_ylabel('Average Energy/Household',color = 'tab:blue')

plt.title('Energy Consumption and Cloud Cover')

fig.tight_layout()

plt.show()
fig, ax1 = plt.subplots(figsize = (20,5))

ax1.plot(weather_energy.day, weather_energy.visibility, color = 'tab:orange')

ax1.set_ylabel('Visibility',color = 'tab:orange')

ax2 = ax1.twinx()

ax2.plot(weather_energy.day,weather_energy.avg_energy,color = 'tab:blue')

ax2.set_ylabel('Average Energy/Household',color = 'tab:blue')

plt.title('Energy Consumption and Visibility')

fig.tight_layout()

plt.show()
fig, ax1 = plt.subplots(figsize = (20,5))

ax1.plot(weather_energy.day, weather_energy.windSpeed, color = 'tab:orange')

ax1.set_ylabel('Wind Speed',color = 'tab:orange')

ax2 = ax1.twinx()

ax2.plot(weather_energy.day,weather_energy.avg_energy,color = 'tab:blue')

ax2.set_ylabel('Average Energy/Household',color = 'tab:blue')

plt.title('Energy Consumption and Wind Speed')

fig.tight_layout()

plt.show()
fig, ax1 = plt.subplots(figsize = (20,5))

ax1.plot(weather_energy.day, weather_energy.uvIndex, color = 'tab:orange')

ax1.set_ylabel('UV Index',color = 'tab:orange')

ax2 = ax1.twinx()

ax2.plot(weather_energy.day,weather_energy.avg_energy,color = 'tab:blue')

ax2.set_ylabel('Average Energy/Household',color = 'tab:blue')

plt.title('Energy Consumption and UV Index')

fig.tight_layout()

plt.show()
fig, ax1 = plt.subplots(figsize = (20,5))

ax1.plot(weather_energy.day, weather_energy.dewPoint, color = 'tab:orange')

ax1.set_ylabel('Dew Point',color = 'tab:orange')

ax2 = ax1.twinx()

ax2.plot(weather_energy.day,weather_energy.avg_energy,color = 'tab:blue')

ax2.set_ylabel('Average Energy/Household',color = 'tab:blue')

plt.title('Energy Consumption and Dew Point')

fig.tight_layout()

plt.show()
cor_matrix = weather_energy[['avg_energy','temperatureMax','dewPoint', 'cloudCover', 'windSpeed','pressure', 'visibility', 'humidity','uvIndex', 'moonPhase']].corr()

cor_matrix
#scaling

scaler = MinMaxScaler()

weather_scaled = scaler.fit_transform(weather_energy[['temperatureMax','humidity','windSpeed']])
# optimum K

Nc = range(1, 20)

kmeans = [KMeans(n_clusters=i) for i in Nc]

kmeans



score = [kmeans[i].fit(weather_scaled).score(weather_scaled) for i in range(len(kmeans))]

score

plt.plot(Nc,score)

plt.xlabel('Number of Clusters')

plt.ylabel('Score')

plt.title('Elbow Curve')

plt.show()
kmeans = KMeans(n_clusters=3, max_iter=600, algorithm = 'auto')

kmeans.fit(weather_scaled)

weather_energy['weather_cluster'] = kmeans.labels_
# Cluster Relationships with weather variables

plt.figure(figsize=(20,5))

plt.subplot(1, 3, 1)

plt.scatter(weather_energy.weather_cluster,weather_energy.temperatureMax)

plt.title('Weather Cluster vs. Temperature')

plt.subplot(1, 3, 2)

plt.scatter(weather_energy.weather_cluster,weather_energy.humidity)

plt.title('Weather Cluster vs. Humidity')

plt.subplot(1, 3, 3)

plt.scatter(weather_energy.weather_cluster,weather_energy.windSpeed)

plt.title('Weather Cluster vs. WindSpeed')



plt.show()

# put this in a loop
fig, ax1 = plt.subplots(figsize = (10,7))

ax1.scatter(weather_energy.temperatureMax, 

            weather_energy.humidity, 

            s = weather_energy.windSpeed*10,

            c = weather_energy.weather_cluster)

ax1.set_xlabel('Temperature')

ax1.set_ylabel('Humidity')

plt.show()
holiday = pd.read_csv('../input/uk_bank_holidays.csv')

holiday['Bank holidays'] = pd.to_datetime(holiday['Bank holidays'],format='%Y-%m-%d').dt.date

holiday.head(4)
weather_energy = weather_energy.merge(holiday, left_on = 'day',right_on = 'Bank holidays',how = 'left')

weather_energy['holiday_ind'] = np.where(weather_energy['Bank holidays'].isna(),0,1)
weather_energy['Year'] = pd.DatetimeIndex(weather_energy['day']).year  

weather_energy['Month'] = pd.DatetimeIndex(weather_energy['day']).month

weather_energy.set_index(['day'],inplace=True)
model_data = weather_energy[['avg_energy','weather_cluster','holiday_ind']]

# train = model_data.iloc[0:round(len(model_data)*0.90)]

# test = model_data.iloc[len(train)-1:]

train = model_data.iloc[0:(len(model_data)-30)]

test = model_data.iloc[len(train):(len(model_data)-1)]
train['avg_energy'].plot(figsize=(25,4))

test['avg_energy'].plot(figsize=(25,4))

test.head(5)
plot_acf(train.avg_energy,lags=100)

plt.show()
plot_pacf(train.avg_energy,lags=50)

plt.show()
t = sm.tsa.adfuller(train.avg_energy, autolag='AIC')

pd.Series(t[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
# function for differencing

def difference(dataset, interval):

    diff = list()

    for i in range(interval, len(dataset)):

        value = dataset.iloc[i] - dataset.iloc[i - interval]

        diff.append(value)

    return diff
t  = sm.tsa.adfuller(difference(train.avg_energy,1), autolag='AIC')

pd.Series(t[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
s = sm.tsa.seasonal_decompose(train.avg_energy,freq=12)
s.seasonal.plot(figsize=(20,5))
s.trend.plot(figsize=(20,5))
s.resid.plot(figsize=(20,5))
endog = train['avg_energy']

exog = sm.add_constant(train[['weather_cluster','holiday_ind']])



mod = sm.tsa.statespace.SARIMAX(endog=endog, exog=exog, order=(7,1,1),seasonal_order=(1,1, 0, 12),trend='c')

model_fit = mod.fit()

model_fit.summary()
train['avg_energy'].plot(figsize=(25,10))

model_fit.fittedvalues.plot()

plt.show()
predict = model_fit.predict(start = len(train),end = len(train)+len(test)-1,exog = sm.add_constant(test[['weather_cluster','holiday_ind']]))

test['predicted'] = predict.values

test.tail(5)
test['residual'] = abs(test['avg_energy']-test['predicted'])

MAE = test['residual'].sum()/len(test)

MAPE = (abs(test['residual'])/test['avg_energy']).sum()*100/len(test)

print("MAE:", MAE)

print("MAPE:", MAPE)
test['avg_energy'].plot(figsize=(25,10),color = 'red')

test['predicted'].plot()

plt.show()
model_fit.resid.plot(figsize= (30,5))
model_fit.fittedvalues.plot(figsize = (30,5))

test.predicted.plot()
test['predicted'].tail(5)
np.random.seed(11)

dataframe = weather_energy.loc[:,'avg_energy']

dataset = dataframe.values

dataset = dataset.astype('float32')
# convert series to supervised learning

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):

    n_vars = 1

    df = pd.DataFrame(data)

    cols, names = list(), list()

    # input sequence (t-n, ... t-1)

    for i in range(n_in, 0, -1):

        cols.append(df.shift(i))

        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)

    for i in range(0, n_out):

        cols.append(df.shift(-i))

        if i == 0:

            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]

        else:

            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

    # put it all together

    agg = pd.concat(cols, axis=1)

    agg.columns = names

    # drop rows with NaN values

    if dropnan:

        agg.dropna(inplace=True)

    return agg
reframed = series_to_supervised(dataset, 7,1)

reframed.head(3)
reframed['weather_cluster'] = weather_energy.weather_cluster.values[7:]

reframed['holiday_ind']= weather_energy.holiday_ind.values[7:]
reframed = reframed.reindex(['weather_cluster', 'holiday_ind','var1(t-7)', 'var1(t-6)', 'var1(t-5)', 'var1(t-4)', 'var1(t-3)','var1(t-2)', 'var1(t-1)', 'var1(t)'], axis=1)

reframed = reframed.values
scaler = MinMaxScaler(feature_range=(0, 1))

reframed = scaler.fit_transform(reframed)
# split into train and test sets

train = reframed[:(len(reframed)-30), :]

test = reframed[(len(reframed)-30):len(reframed), :]
train_X, train_y = train[:, :-1], train[:, -1]

test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]

train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))

test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
# design network

model = Sequential()

model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))

model.add(Dense(1))

model.compile(loss='mae', optimizer='adam')

# fit network

history = model.fit(train_X, train_y, epochs=50, batch_size=72, verbose=2, shuffle=False)

# plot history

pyplot.plot(history.history['loss'], label='train')

pyplot.legend()

pyplot.show()
# make a prediction

yhat = model.predict(test_X)
test_X = test_X.reshape(test_X.shape[0], test_X.shape[2])
# invert scaling for forecast

inv_yhat = np.concatenate((yhat, test_X), axis=1)

inv_yhat = scaler.inverse_transform(inv_yhat)
# invert scaling for actual

test_y = test_y.reshape((len(test_y), 1))

inv_y = np.concatenate((test_y, test_X), axis=1)

inv_y = scaler.inverse_transform(inv_y)
act = [i[9] for i in inv_y] # last element is the predicted average energy

pred = [i[9] for i in inv_yhat] # last element is the actual average energy



# calculate RMSE

import math

rmse = math.sqrt(mean_squared_error(act, pred))

print('Test RMSE: %.3f' % rmse)
predicted_lstm = pd.DataFrame({'predicted':pred,'avg_energy':act})

predicted_lstm['avg_energy'].plot(figsize=(25,10),color = 'red')

predicted_lstm['predicted'].plot(color = 'blue')

plt.show()