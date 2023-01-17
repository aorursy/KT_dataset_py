import pandas as pd
import re 
from matplotlib import pyplot
import seaborn as sns
import numpy as np
import os # accessing directory structure


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

#set seed
np.random.seed(12345)

notclean = pd.read_csv('../input/bitcoin-tweets-14m/cleanprep.csv', delimiter=',', error_bad_lines=False,engine = 'python',header = None)
notclean.head()
#-----------------Pre-processing -------------------#

notclean.columns =['dt', 'name','text','polarity','sensitivity']
notclean =notclean.drop(['name','text'], axis=1)
notclean.head()
notclean.info()
notclean['dt'] = pd.to_datetime(notclean['dt'])
notclean['DateTime'] = notclean['dt'].dt.floor('h')
notclean.head()
vdf = notclean.groupby(pd.Grouper(key='dt',freq='H')).size().reset_index(name='tweet_vol')
vdf.head()
vdf.info()
vdf.index = pd.to_datetime(vdf.index)
vdf=vdf.set_index('dt')
vdf.info()
vdf.head()
notclean.info()
notclean.index = pd.to_datetime(notclean.index)
notclean.info()
vdf['tweet_vol'] =vdf['tweet_vol'].astype(float)
vdf.info()
notclean.info()
notclean.head()
#ndf = pd.merge(notclean,vdf, how='inner',left_index=True, right_index=True)
notclean.head()
df = notclean.groupby('DateTime').agg(lambda x: x.mean())
df['Tweet_vol'] = vdf['tweet_vol']
df = df.drop(df.index[0])
df.head()
df.tail()
df.info()
btcDF = pd.read_csv('../input/btc-price/btcSave2.csv', error_bad_lines=False,engine = 'python')
btcDF['Timestamp'] = pd.to_datetime(btcDF['Timestamp'])
btcDF = btcDF.set_index(pd.DatetimeIndex(btcDF['Timestamp']))
btcDF.head()
btcDF = btcDF.drop(['Timestamp'], axis=1)
btcDF.head()
Final_df = pd.merge(df,btcDF, how='inner',left_index=True, right_index=True)
Final_df.head()
Final_df.info()
Final_df=Final_df.drop(['Weighted Price'],axis=1 )
Final_df.head()
Final_df.columns = ['Polarity', 'Sensitivity','Tweet_vol','Open','High','Low', 'Close_Price', 'Volume_BTC', 'Volume_Dollar']
Final_df.head()
Final_df = Final_df[['Polarity', 'Sensitivity','Tweet_vol', 'Open','High','Low', 'Volume_BTC', 'Volume_Dollar', 'Close_Price']]
Final_df
#---------------Stage 1 Complete ------------------#
#Final_df.to_csv('Desktop/Sentiment.csv')
#--------------Analysis----------------------------#

values = Final_df.values
groups = [0,1,2,3,4,5,6,7]
i =1  
pyplot.figure()
for group in groups:
    pyplot.subplot(len(groups), 1,i)
    pyplot.plot(values[:,group])
    pyplot.title(Final_df.columns[group], y=.5, loc='right')
    i += 1
pyplot.show()
Final_df['Volume_BTC'].max()
Final_df['Volume_Dollar'].max()
Final_df['Volume_BTC'].sum()
Final_df['Volume_Dollar'].sum()
Final_df['Tweet_vol'].max()
Final_df.describe()
cor = Final_df.corr()
cor
Top_Vol =Final_df['Volume_BTC'].nlargest(10)
Top_Vol
Top_Sen =Final_df['Sensitivity'].nlargest(10)
Top_Sen
Top_Pol =Final_df['Polarity'].nlargest(10)
Top_Pol
Top_Tweet =Final_df['Tweet_vol'].nlargest(10)
Top_Tweet
import matplotlib.pyplot as plt
sns.set(style="white")
f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(220, 10, as_cmap=True)
ax =sns.heatmap(cor, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .7})
plt.show()
plt.plot(Final_df.index, Final_df['Close_Price'], 'black')
plt.plot(Final_df.index, Final_df['Open'], 'yellow')
plt.plot(Final_df.index, Final_df['Low'], 'red')
plt.plot(Final_df.index, Final_df['High'], 'green')
plt.title('BTC Close Price(hr)')
plt.xticks(rotation='vertical')
plt.ylabel('Price ($)');
plt.show();

plt.plot(Final_df.index, Final_df['Volume_BTC'], 'g')
plt.title('Trading Vol BTC(hr)')
plt.xticks(rotation='vertical')
plt.ylabel('Vol BTC');
plt.show();

plt.plot(Final_df.index, Final_df['Polarity'], 'b')
plt.xticks(rotation='vertical')
plt.title('Twitter Sentiment(hr)')
plt.ylabel('Pol (0-1)');
plt.show();
plt.legend()

plt.plot(Final_df.index, Final_df['Tweet_vol'], 'b')
plt.xticks(rotation='vertical')
plt.title('Tweet Vol(hr)')
plt.ylabel('No. of Tweets');
plt.show();
plt.legend()
#sns Heatmap for Hour x volume 
#Final_df['time']=Final_df.index.time()
Final_df['time']=Final_df.index.to_series().apply(lambda x: x.strftime("%X"))

Final_df.head()
hour_df=Final_df
hour_df=hour_df.groupby('time').agg(lambda x: x.mean())
hour_df
hour_df.head()
#sns Hourly Heatmap
hour_df['hour'] = hour_df.index
result = hour_df.pivot(index='hour', columns='Polarity', values='Volume_BTC')
sns.heatmap(result, annot=True, fmt="g", cmap='viridis')
plt.title('Polarity x BTC Volume avg(Hr)')
plt.show()

#sns daily heatmap?
hour_df['hour'] = hour_df.index
result = hour_df.pivot(index='Volume_BTC', columns='hour', values='Tweet_vol')
sns.heatmap(result, annot=True, fmt="g", cmap='viridis')
plt.title('BTC Vol x Tweet Vol avg(Hr)')
plt.show()
cor = Final_df.corr()
cor
#----------------End Analysis------------------------#
#---------------- LSTM Prep ------------------------#
df = Final_df
df.info()
df = df.drop(['Open','High', 'Low', 'Volume_Dollar'], axis=1)
df.head()
df = df[['Close_Price', 'Polarity', 'Sensitivity','Tweet_vol','Volume_BTC']]
df.head()
cor = df.corr()
import matplotlib.pyplot as plt
sns.set(style="white")
f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(220, 10, as_cmap=True)
ax =sns.heatmap(cor, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .7})
plt.show()
from math import sqrt
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot

from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
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
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
values = df.values
cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]
df = df[cols]
df = df[['Close_Price', 'Polarity', 'Sensitivity','Tweet_vol','Volume_BTC']]
df.head()
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(df.values)
n_hours = 3 #adding 3 hours lags creating number of observations 
n_features = 5 #Features in the dataset.
n_obs = n_hours*n_features
reframed = series_to_supervised(scaled, n_hours, 1)
reframed.head()
reframed.drop(reframed.columns[-4], axis=1)
reframed.head()
print(reframed.head())
values = reframed.values
n_train_hours = 200
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
train.shape
# split into input and outputs
train_X, train_y = train[:, :n_obs], train[:, -n_features]
test_X, test_y = test[:, :n_obs], test[:, -n_features]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
# design network
model = Sequential()
model.add(LSTM(5, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=6, validation_data=(test_X, test_y), verbose=2, shuffle=False,validation_split=0.2)
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], n_hours* n_features,))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, -4:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, -4:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
mse = (mean_squared_error(inv_y, inv_yhat))
print('Test MSE: %.3f' % mse)
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

plt.plot(inv_y, label='Real')
plt.plot(inv_yhat, label='Predicted')
plt.title('Real v Predicted Close_Price')
plt.ylabel('Price ($)')
plt.xlabel('epochs (Hr)')
plt.show()

