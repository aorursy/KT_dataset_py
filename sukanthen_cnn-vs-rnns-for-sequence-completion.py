from numpy import array

from keras.models import Sequential

from keras.layers import Flatten, Dense

from keras.layers.convolutional import Conv1D, MaxPooling1D
x = array([[10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50, 60]])

y = array([40,50,60,70])
x = x.reshape((x.shape[0], x.shape[1], 1))
model = Sequential()

model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(3, 1)))

model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())

model.add(Dense(50, activation='relu'))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

model.fit(x, y, epochs=1000, verbose=1)
ip = array([50, 60, 70])

ip =ip.reshape((1, 3, 1))

y_pred = model.predict(ip, verbose=1)

print(y_pred)
x = array([[5,10,15], [10,15,20], [20,25,30], [35,40,45]])

y = array([20,25,35,50])
x = x.reshape((x.shape[0], x.shape[1], 1))
model = Sequential()

model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(3,1)))

model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())

model.add(Dense(50, activation='relu'))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

model.fit(x, y, epochs=1000, verbose=2)
ip = array([75,80,85])

ip =ip.reshape((1, 3, 1))

y_pred = model.predict(ip, verbose=1)

print(y_pred)
from keras.layers import LSTM

from sklearn.metrics import mean_squared_error
model = Sequential()

model.add(LSTM(4, input_shape=(3,1)))

model.add(Dense(1))

model.compile(loss='mean_squared_error',optimizer='adam')

model.fit(x,y,epochs=1000,batch_size=1,verbose=2)
ip = array([50,55,60])

ip =ip.reshape((1, 3, 1))

y_pred = model.predict(ip, verbose=1)

print(y_pred)
X = array([[10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50, 60]])

y = array([40, 50, 60, 70])

# reshape from [samples, timesteps] into [samples, timesteps, features]

X = X.reshape((X.shape[0], X.shape[1], 1))

# define model

model = Sequential()

model.add(LSTM(4, input_shape=(3,1)))

model.add(Dense(1))

model.compile(loss='mean_squared_error',optimizer='adam')

model.fit(X,y,epochs=1000,batch_size=1,verbose=2)
# demonstrate prediction

x_input = array([40,50,60])

x_input = x_input.reshape((1, 3, 1))

yhat = model.predict(x_input, verbose=0)

print(yhat)