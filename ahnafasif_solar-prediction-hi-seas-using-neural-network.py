import pandas as pd
df = pd.read_csv('../input/SolarEnergy/SolarPrediction.csv')
print(df.shape)
df.head()
df.isnull().sum()
print(df.index)

df = df.sort_values(['UNIXTime'], ascending = [True])
df.index =  pd.to_datetime(df['UNIXTime'], unit='s')

print(df.index)

df.head()
import pytz

HST = pytz.timezone('Pacific/Honolulu')
                                                                                ## unixtime is in utc
df.index = df.index.tz_localize(pytz.utc)                                       ## we can see that by comparing with "Time",
df.index = df.index.tz_convert(HST)                                             ## converting to hawaiian local time
df.head()
df['DayOfYear'] = df.index.strftime('%j').astype(int)
df.head()
df['TimeOfDay(s)'] = df.index.hour*60*60 + df.index.minute*60 + df.index.second
df.head()
df.drop(['TimeSunRise','TimeSunSet', 'Data', 'Time', 'WindDirection(Degrees)'], inplace=True, axis=1)

df.head()
print(df.shape)
dataset = df.values

X = dataset[:,2:8]
Y = dataset[:,1]
Y = Y.reshape(-1,1)


print(X.shape)
print(Y.shape)
print(type(X))
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


scaler_X = preprocessing.StandardScaler().fit(X)
scaler_Y = preprocessing.StandardScaler().fit(Y)
X_scale = scaler_X.transform(X)
X_train, X_val_and_test, Y_train_unscaled, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.2)
X_val, X_test, Y_val_unscaled, Y_test_unscaled = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)
Y_train = scaler_Y.transform(Y_train_unscaled)
Y_val =  scaler_Y.transform(Y_val_unscaled)
Y_test =  scaler_Y.transform(Y_test_unscaled)
print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers


## kernel_initializer='normal'


model = Sequential()
model.add(Dense(64, activation='relu',  kernel_initializer='normal', input_shape=(6,), kernel_regularizer=regularizers.l2(l=0.01)))
model.add(Dropout(0.1))
model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l=0.01)))
model.add(Dropout(0.1))
model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(l=0.01)))
model.add(Dropout(0.1))
model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l=0.01)))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l=0.01)))
model.add(Dropout(0.1))
model.add(Dense(1, activation='linear'))
from keras.optimizers import Adam

#optimizer = Adam(lr=1e-3, decay=1e-3 / 200)

model.compile(loss='mse',
              optimizer='adam',
              metrics=['mae'])
hist = model.fit(X_train, Y_train,
                batch_size=32, epochs=300,
                validation_data=(X_val, Y_val))
from matplotlib import pyplot

pyplot.title('Loss / Mean Squared Error')
pyplot.plot(hist.history['loss'], label='train')
pyplot.plot(hist.history['val_loss'], label='validate')
pyplot.legend()
pyplot.show()
model.evaluate(X_test, Y_test)
Y_result_scaled= model.predict(X_test)
Y_result = scaler_Y.inverse_transform(Y_result_scaled)
print(Y_result)
print(Y_test_unscaled.reshape(Y_result.shape))
import numpy as np

axis_x = [i for i in range(50)]


pyplot.plot(axis_x, Y_result[:50], label='predict')
pyplot.plot(axis_x, Y_test_unscaled[:50], label='actual')
pyplot.legend()
pyplot.show()
from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score


explained_variance_score = explained_variance_score(Y_test_unscaled.reshape(Y_result.shape), Y_result)
mean_squared_error = mean_squared_error(Y_test_unscaled.reshape(Y_result.shape), Y_result)
r_squared = r2_score(Y_test_unscaled.reshape(Y_result.shape), Y_result)
print('explained variance = {}'.format(explained_variance_score))
print('mse = {}'.format(mean_squared_error))
print('r2 = {}'.format(r_squared))
