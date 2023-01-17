import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import Series
from pandas import concat
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from math import sqrt
# Read the CSV file
df = pd.read_csv('../input/eurusd-historical/EURUSD_D1.csv', header=0, delimiter="\t")
# Rename the columns
df.rename(columns={'Time' : 'timestamp', 'Open' : 'open', 'Close' : 'close', 'High' : 'high', 'Low' : 'low', 'Close' : 'close', 'Volume' : 'volume'}, inplace=True)
# Set the 'timestamp' column as an index for the dataframe
df.set_index('timestamp', inplace=True)
# Convert all values as floating point numbers
df = df.astype(float)
# Check on the data types of the columns
print('Data type of each column in Dataframe: ')
print(df.dtypes)

df = df.iloc[0:]

# Contents of the dataframe
print('Contents of the Dataframe: ')
print(df)
# Drop unused columns from dataframe
columns_to_drop = ['open', 'high', 'low', 'volume']
df.drop(columns_to_drop, axis=1, inplace=True)
print(df.head(15))
# Plotting
plt.title('EURUSD Historical Close Prices')
plt.plot(df['close'].values, label='Close')
plt.legend(loc="upper right")
plt.show()
look_back = 20

# Frame a time series as a supervised learning dataset.
# Arguments:
#    data: Sequence of observations as a list or NumPy array.
#    n_in: Number of lag observations as input (X).
#    n_out: Number of observations as output (y).
#    drop_nan: Boolean whether or not to drop rows with NaN values.
# Returns:
#    Pandas DataFrame of series framed for supervised learning.
def series_to_supervised(data, n_in=1, n_out=1, drop_nan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if drop_nan:
        agg.dropna(inplace=True)
    return agg

# Convert the dataframe to a series
series = df.values

# Convert the series to a dataframe suitable for supervised learning
df_supervised = series_to_supervised(series, look_back, 1, True)
print(df_supervised.head(15))
# Now we should add an additional column to this supervised dataframe which will contain either the value 1 or 0
# If('var1_t' > 'var1_t5') then 'var1(t) >= var1(t-5)' = 1 else 'var1(t) >= var1(t-5)' = 0
df_supervised.loc[df_supervised['var1(t)'] >= df_supervised['var1(t-5)'], 'var1(t) >= var1(t-5)'] = 1
df_supervised.loc[df_supervised['var1(t)'] < df_supervised['var1(t-5)'], 'var1(t) >= var1(t-5)'] = 0
print(df_supervised.head(15))
# After that we can safely remove the column var1_t
df_supervised.drop('var1(t)', axis=1, inplace=True)
print(df_supervised.head(15))
pct_train = 80

series_supervised = df_supervised.values
train_size = int(len(series_supervised) * pct_train / 100)
train = series_supervised[0:train_size]
test = series_supervised[train_size:]
print('Training series:')
print(train)
print(train.shape)
print('Test series:')
print(test)
print(test.shape)
# train_X, train_y
train_X = train[:, 0:-1]
train_y = train[:, -1]
print('Training series X:')
print(train_X)
print('Training series y:')
print(train_y)
# test_X, test_y
test_X = test[:, 0:-1]
test_y = test[:, -1]
print('Test series X:')
print(test_X)
print('Test series y:')
print(test_y)
# Scale each row in train_X
train_X_scaled = np.array([])
for i in range(0, len(train_X)):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    reshaped = train_X[i].reshape(len(train_X[i]), 1)
    scaler = scaler.fit(reshaped)
    # transform train
    scaled = scaler.transform(reshaped)
    train_X_scaled = np.append(train_X_scaled, scaled)

train_X_scaled = train_X_scaled.reshape(train_X.shape[0], train_X.shape[1])
print('Training series X scaled:')
print(train_X_scaled)
print(train_X_scaled.shape)
# Scale each row in test_X
test_X_scaled = np.array([])
for i in range(0, len(test_X)):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    reshaped = test_X[i].reshape(len(test_X[i]), 1)
    scaler = scaler.fit(reshaped)
    # transform train
    scaled = scaler.transform(reshaped)
    test_X_scaled = np.append(test_X_scaled, scaled)

test_X_scaled = test_X_scaled.reshape(test_X.shape[0], test_X.shape[1])
print('Test series X scaled:')
print(test_X_scaled)
print(test_X_scaled.shape)
time_steps = 20
features = 1
neurons = 64
batch_size = 1
nb_epoch = 10

# reshape input to be 3D [samples, timesteps, features]
train_X_scaled = train_X_scaled.reshape((train_X_scaled.shape[0], time_steps, features))
test_X_scaled = test_X_scaled.reshape((test_X.shape[0], time_steps, features))
print(train_X_scaled.shape, train_y.shape, test_X_scaled.shape, test_y.shape)

# design network
model = Sequential()
model.add(LSTM(neurons, batch_input_shape=(batch_size, train_X_scaled.shape[1], train_X_scaled.shape[2]), stateful=True, return_sequences=True))
model.add(Dropout(0.2))
#model.add(LSTM(neurons, return_sequences=True))  # returns a sequence of vectors of dimension 32
#model.add(Dropout(0.2))
model.add(LSTM(neurons, return_sequences=False))  # return a single vector of dimension 32
model.add(Dropout(0.2))
#model.add(Dense(1))
model.add(Dense(1,activation='sigmoid'))
#model.compile(loss='mae', optimizer='adam')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
# fit network
history = model.fit(train_X_scaled, train_y, epochs=nb_epoch, batch_size=1, validation_data=(test_X_scaled, test_y), verbose=1, shuffle=False)
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# Forecast the entire training dataset to build up state for forecasting
predictions_on_training = model.predict(train_X_scaled, batch_size=1)
print('Expected values on training')
print(train_y)
print(train_y.shape)
print('Predicted values on training')
print(predictions_on_training)
print(predictions_on_training.shape)

    
plt.title('Expected and predicted values on Training Set')
plt.plot(train_y[-20:], label="expected")
plt.plot(predictions_on_training[-20:], label="predicted")
plt.legend(loc="upper left")
plt.show()
# make a one-step forecast
print(test_X_scaled)
print(test_X_scaled.shape)
print(test_y)
print(test_y.shape)
# Forecast on test data
predictions_on_test = np.array([])
print(len(test_X_scaled))
for i in range(len(test_X_scaled)):
    X = test_X_scaled[i, :]
    X = X.reshape(1, X.shape[0], X.shape[1])
    y = model.predict(X, batch_size=1)
    predictions_on_test = np.append(predictions_on_test, y)

print('Expected values on test')
print(test_y)
print(test_y.shape)
print('Predicted values on test')
print(predictions_on_test)
print(predictions_on_test.shape)

    
plt.title('Expected and predicted values on Test Set')
plt.plot(test_y[-20:], label="expected")
plt.plot(predictions_on_test[-20:], label="predicted")
plt.legend(loc="upper left")
plt.show()    
    


# How many ups and downs to we have in the test set
unique, counts = np.unique(test_y, return_counts=True)
print(unique)
print(counts)

# How many ups do we have in the predictions
print((predictions_on_test > 0.5).sum())
# How many downs do we have in the predictions
print((predictions_on_test < 0.5).sum())

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
dirs = ['Up', 'Down']
count_test_y = [counts[1],counts[0]]
ax.bar(dirs ,count_test_y)
plt.show()

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
dirs = ['Up', 'Down']
count_predictions_on_test = [(predictions_on_test > 0.5).sum(),(predictions_on_test < 0.5).sum()]
ax.bar(dirs ,count_predictions_on_test)
plt.show()