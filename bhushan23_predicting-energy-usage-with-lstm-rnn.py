from math import sqrt

from sklearn.cross_validation import train_test_split

from numpy import concatenate

from pandas import read_csv

from pandas import DataFrame

from pandas import concat

from pandas import to_datetime

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_error

from keras.models import Sequential

from keras.layers import LSTM

from keras.layers import Dense

from matplotlib import pyplot

# convert series to supervised learning

def series_to_supevised(dataset, n_in=1, n_out=1, dropnan=True):

    num_vars = 1 if type(dataset) is list else dataset.shape[1]

    dataframe = DataFrame(dataset)

    cols, names = list(), list()

    

    # input sequence (t-n, ....t-1)

    for i in range(n_in, 0, -1):

        cols.append(dataframe.shift(i))

        names += [('var%d(t-%d)' % (j+1, i)) for j in range(num_vars)]

    # forecast sequence (t, t+1 .... t+n)

    for i in range(0, n_out):

        cols.append(dataframe.shift(-i))

        if i == 0:

            names += [('var%d(t)' % (j+1)) for j in range(num_vars)]

        else:

            names += [('var%d(t+%d)' % (j+1, i)) for j in range(num_vars)]

    

    # put it all together 

    agg = concat(cols, axis=1)

    agg.columns = names

    

    # drop rows with NaN values

    if dropnan:

        agg.dropna(inplace=True)

    return agg

#import os

#!ls ../input

dataset1 = read_csv('../input/appliances-energy-prediction/KAG_energydata_complete.csv')

dataset = read_csv('../input/ukdale-house-5-tv-usage/channel_5TV.dat', sep=' ', names=['Date', 'Usage'])

values = dataset.values

print(type(dataset['Date'][0]), type(dataset['Usage'][0]))

print(dataset[:10])

print(values[:10])

print(type(values[0][0]), type(values[0][0]))

#values = values[:100,:]
values[:,0] = to_datetime(values[:,0])

print(type(values[0][0]), type(values[0][0]))
# normalize features

scaler = MinMaxScaler(feature_range=(0,1))

scaled = scaler.fit_transform(values)
# frame as supervised learning

reframed = series_to_supevised(scaled, 1, 1)



# drop columns we don't want to predict

#reframed.drop(reframed.columns[[29,30,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57]], axis=1, inplace=True)

print("reframed: ", reframed.shape, "reframed head: ", reframed.head())



# split into train and test sets

values = reframed.values



X = values[:,:1]

Y = values[:,1]

#Y2 = dataset[:,16]  





# Split Data to Train and Test

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.3)



# reshape input to be 3D [samples, timesteps, features]

X_Train = X_Train.reshape((X_Train.shape[0], 1, X_Train.shape[1]))

X_Test = X_Test.reshape((X_Test.shape[0], 1, X_Test.shape[1]))
# network architecture

model = Sequential()

model.add(LSTM(50, input_shape=(X_Train.shape[1], X_Train.shape[2])))

model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')



# fit

history = model.fit(X_Train, Y_Train, epochs=1, batch_size=10, validation_data=(X_Test, Y_Test), verbose=2, shuffle=False)
# plot history



pyplot.plot(history.history['loss'], label='Train')

pyplot.plot(history.history['val_loss'], label='Test')

pyplot.legend()

pyplot.show()