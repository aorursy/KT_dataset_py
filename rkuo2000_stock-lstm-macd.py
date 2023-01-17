import os 
print(os.listdir('../input'))
import numpy as np
import pandas as pd
from sklearn import preprocessing

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate

import matplotlib.pyplot as plt
# take 50 history points (~2 months)
history_points = 50
csv_path = '../input/stocks/GOOGL_daily.csv'
# Read Dataset & Normalise it
def csv_to_dataset(csv_path):
    data = pd.read_csv(csv_path)
    data = data.drop('date', axis=1)
    ## reverse index because .csv top column is most recent price 
    data = data[::-1]
    data = data.reset_index()
    data = data.drop('index', axis=1)
    print(data)

    # normaliser
    data_normaliser = preprocessing.MinMaxScaler()
    data_normalised = data_normaliser.fit_transform(data) 
    # using the last {history_points} open high low close volume data points, predict the next open value
    ohlcv_histories_normalised =      np.array([data_normalised[i  : i + history_points].copy() for i in range(len(data_normalised) - history_points)])
    print(ohlcv_histories_normalised.shape)    
    next_day_open_values_normalised = np.array([data_normalised[:,0][i + history_points].copy() for i in range(len(data_normalised) - history_points)])   

    next_day_open_values_normalised = np.expand_dims(next_day_open_values_normalised, -1)

    next_day_open_values = np.array([data.loc[:,"1. open"][i + history_points].copy() for i in range(len(data) - history_points)])
    next_day_open_values = np.expand_dims(next_day_open_values, -1)
    
    y_normaliser = preprocessing.MinMaxScaler()
    y_normaliser.fit(next_day_open_values)

    # add MACD
    def calc_ema(values, time_period):
        # https://www.investopedia.com/ask/answers/122314/what-exponential-moving-average-ema-formula-and-how-ema-calculated.asp
        sma = np.mean(values[:,3])
        ema_values = [sma]
        k = 2 / (1 + time_period)
        for i in range(len(his) - time_period, len(his)):
            close = his[i][3]
            ema_values.append(close * k + ema_values[-1] * (1 - k))
        return ema_values[-1]
    
    # add technical indicators 
    technical_indicators = []
    for his in ohlcv_histories_normalised:
        # since we are using his[3] we are taking the SMA of the closing price
        sma = np.mean(his[:,3])                            # add SMA
        macd = calc_ema(his, 12) - calc_ema(his, 26)       # add MACD    
        technical_indicators.append(np.array([sma,macd,])) # add MACD

    technical_indicators = np.array(technical_indicators)

    tech_ind_scaler = preprocessing.MinMaxScaler()
    technical_indicators_normalised = tech_ind_scaler.fit_transform(technical_indicators)

    assert ohlcv_histories_normalised.shape[0] == next_day_open_values_normalised.shape[0]
    return ohlcv_histories_normalised, technical_indicators_normalised, next_day_open_values_normalised, next_day_open_values, y_normaliser
ohlcv_histories, technical_indicators, next_day_open_values, unscaled_y, y_scaler = csv_to_dataset(csv_path)
# splitting the dataset up into train and test sets
test_split = 0.9 # 90% stock-history for training, most-recent 10% stock-history for testing
n = int(ohlcv_histories.shape[0] * test_split)

ohlcv_train = ohlcv_histories[:n]
tech_ind_train = technical_indicators[:n] # add technical indicator
y_train = next_day_open_values[:n]

ohlcv_test = ohlcv_histories[n:]
tech_ind_test = technical_indicators[n:] # add technical indicator
y_test = next_day_open_values[n:]

unscaled_y_test = unscaled_y[n:]

print(ohlcv_train.shape)
# Build Model
lstm_input = Input(shape=(history_points, 5), name='lstm_input')
dense_input = Input(shape=(technical_indicators.shape[1],), name='tech_input') # 2nd input for technical indicator

x = LSTM(50, name='lstm_0')(lstm_input)
x = Dropout(0.2, name='lstm_dropout_0')(x)

# the second branch opreates on the second input
lstm_branch = Model(inputs=lstm_input, outputs=x)
y = Dense(25, name='tech_dense_0')(dense_input)
y = Activation("relu", name='tech_relu_0')(y)
y = Dropout(0.2, name='tech_dropout_0')(y)
technical_indicators_branch = Model(inputs=dense_input, outputs=y)
# combine the output of the two branches
combined = concatenate([lstm_branch.output, technical_indicators_branch.output], name='concatenate')
 
z = Dense(50, activation="sigmoid", name='dense_pooling')(combined)
z = Dense(1, activation="linear", name='dense_out')(z)
 
# this model will accept the inputs of the two branches and then output a single value
model = Model(inputs=[lstm_branch.input, technical_indicators_branch.input], outputs=z)

model.summary()
# Compile Model
adam = Adam(lr=0.0005)
model.compile(optimizer=adam, loss='mse')
# Train Model
num_epochs = 100
batch_size = 32
model.fit(x=[ohlcv_train, tech_ind_train], y=y_train, batch_size=batch_size, epochs=num_epochs, shuffle=True, validation_split=0.1)
# Evaluate Model
evaluation = model.evaluate([ohlcv_test, tech_ind_test], y_test)
print(evaluation)
y_test_predicted = model.predict([ohlcv_test, tech_ind_test])

# model.predict returns normalised values, now we scale them back up using the y_scaler from before
y_test_predicted = y_scaler.inverse_transform(y_test_predicted)

# also getting predictions for the entire dataset, just to see how it performs
y_predicted = model.predict([ohlcv_histories,technical_indicators])
y_predicted = y_scaler.inverse_transform(y_predicted)

assert unscaled_y_test.shape == y_test_predicted.shape
real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
print(scaled_mse)
# Plot stock prediction
plt.gcf().set_size_inches(22, 15, forward=True)
start = 0
end = -1
real = plt.plot(unscaled_y_test[start:end], label='real')
pred = plt.plot(y_test_predicted[start:end], label='predicted')
plt.title('symbol = GOOGL')
plt.legend(['Real', 'Predicted'])
plt.show()
buys = []
sells = []
thresh = 0.2

x = 0
for ohlcv, ind in zip(ohlcv_test, tech_ind_test):
    normalised_price_today = ohlcv[-1][0]
    normalised_price_today = np.array([[normalised_price_today]])
    
    price_today = y_scaler.inverse_transform(normalised_price_today)
    
    predicted = np.squeeze(y_scaler.inverse_transform( model.predict( [np.array([ohlcv]), np.array([ind]) ]) ))
                           
    
    delta = predicted - price_today
    # print(delta)
    if delta > thresh:
        buys.append((x, price_today[0][0]))
    elif delta < -thresh:
        sells.append((x, price_today[0][0]))
    x += 1
print(buys)
print(sells)
plt.gcf().set_size_inches(22, 15, forward=True)

start = 0
end = -1

real = plt.plot(unscaled_y_test[start:end], label='real')
pred = plt.plot(y_test_predicted[start:end], label='predicted')

plt.scatter(list(list(zip(*buys))[0]), list(list(zip(*buys))[1]), c= '#00ff00')   # buy points in green
plt.scatter(list(list(zip(*sells))[0]), list(list(zip(*sells))[1]), c= '#ff0000') # sell points in red

# real = plt.plot(unscaled_y[start:end], label='real')
# pred = plt.plot(y_predicted[start:end], label='predicted')

plt.legend(['Real', 'Predicted'])

plt.show()
