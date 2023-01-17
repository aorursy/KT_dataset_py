import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
# Loading the price data and moving the closing price to the last column for clarity
df = pd.read_csv("../input/prices-split-adjusted.csv", index_col = 0)
df.rename(columns={'close': 'temp_close'}, inplace=True)
df["close"] = df.temp_close
df.drop(['temp_close'], 1, inplace=True)
df.head()
df.describe()
df.isna().sum()
# Count the number of distinct companies in the dataset
symbols = list(set(df.symbol))
len(symbols)
# Display sample companies symbols
symbols[:10]
# Selection of Google as the stock to analyse
df = df[df.symbol == 'GOOG']
df.drop(['symbol'],1,inplace=True)
df.head()
df.describe()
data = df.values
# specify columns to plot
features = [0, 1, 2, 3, 4]
i = 1
# plot each column
plt.figure()
for feature in features:
    plt.subplot(len(features), 1, i)
    plt.plot(data[:, feature])
    plt.title(df.columns[feature], y=0.5, loc='right')
    i += 1
plt.show()
def difference_for_series(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        # Record of index 0 is skipped as there would not be a prior record to take a difference with
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

def difference_for_entire_df(dataset, interval=1):
    df = DataFrame(dataset)
    column_names = list(df.columns.values)
    columns = [difference_for_series(df.iloc[:,i], interval=1) for i in range(0, df.shape[1])]
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    df.columns = column_names
    return df

def create_lag_column(data, lags=[-1]):
    df = DataFrame(data)
    column_names = list(df.columns.values)
    columns = list()
    columns.append(df)
    
    for lag in lags:
        column_names.append("next_%d_day_close" % (-lag))
        column = df['close'].shift(lag)
        columns.append(column)
    
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    
    # Find the most number of days of lag and remove the corresponding number of records
    # from the back of the dataframe as there would not be expected target for those records
    df.drop(df.tail(-min(lags)).index,inplace=True)
    df.columns = column_names
    return df

def scale(train, test):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    train_scaled = scaler.transform(train)
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

# inverse scaling for a forecasted value
def invert_scale(scaler, X, values, target_size=1):
    row_for_transform_undo = [x for x in X] + [value for value in values]
    array_for_transform_undo = np.array(row_for_transform_undo)
    array_for_transform_undo = array_for_transform_undo.reshape(1, len(array_for_transform_undo))
    inverted = scaler.inverse_transform(array_for_transform_undo)
    return inverted[0, -target_size:]
 
# fit an LSTM network to training data
def fit_lstm(train, batch_size, num_epoch, neurons, target_size=1):
    X, y = train[:, 0:-target_size], train[:, -target_size:]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), return_sequences=True, stateful=True))
    model.add(LSTM(neurons, stateful=True))
    model.add(Dense(target_size))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(num_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
    return model
 
# make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0,0:3]
# Get the difference between each row of data as a new dataframe
diff_df = difference_for_entire_df(df, interval=1)

# Create a new column that contains the lag of the closing price by one day
lag_diff_df = create_lag_column(diff_df)

# Defining the training record size
train_record_size = round(0.9 * lag_diff_df.shape[0])
print("Number of records for Google price: " + str(lag_diff_df.shape[0]))
print("Number of training records to use: " + str(train_record_size))

# Train-Test Split
dataset = lag_diff_df.values
train, test = dataset[0:train_record_size], dataset[train_record_size:]
print("Train set size: " + str(len(train)))
print("Test set size: " + str(len(test)))

# Scaling
scaler, train_scaled, test_scaled = scale(train, test)
# Before transforming prediction back to original scale
rmse = sqrt(mean_squared_error(test_scaled[:,5], test_scaled[:,4]))
print('RMSE: %.3f' % rmse)
# line plot of observed vs predicted
plt.plot(test_scaled[:,5])
plt.plot(test_scaled[:,4])
plt.show()
# After transforming prediction back to original scale
rmse = sqrt(mean_squared_error(df.iloc[-len(test)-1:-1,-1], df.iloc[-len(test)-2:-2,-1]))
print('RMSE: %.3f' % rmse)
# line plot of observed vs predicted
plt.plot(df.iloc[-len(test)-1:-1,-1].values)
plt.plot(df.iloc[-len(test)-2:-2,-1].values)
plt.show()
# fit the model
lstm_model = fit_lstm(train_scaled, 1, 20, 64)
# forecast the entire training dataset to build up state for forecasting
train_reshaped = train_scaled[:, 0:-1].reshape(len(train_scaled), 1, 5)
lstm_model.predict(train_reshaped, batch_size=1)

# walk-forward validation on the test data
predictions = list()
reverted_predictions = list()
for i in range(len(test_scaled)):
    # make one-step forecast
    X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    yhat = forecast_lstm(lstm_model, 1, X)
    # store forecast
    predictions.append(yhat)
    
    # invert scaling
    reverted_yhat = invert_scale(scaler, X, yhat)
    # invert differencing
    reverted_yhat = inverse_difference(df["close"], reverted_yhat, len(test_scaled)+1-i)
    # store forecast
    reverted_predictions.append(reverted_yhat)
    
predictions = np.asarray(predictions)
reverted_predictions = np.asarray(reverted_predictions)
# report performance
rmse = sqrt(mean_squared_error(test_scaled[:, -1], predictions))
print('Test RMSE: %.3f' % rmse)
# line plot of observed vs predicted
plt.plot(test_scaled[:, -1])
plt.plot(predictions)
plt.show()
# report performance
rmse = sqrt(mean_squared_error(df.iloc[-len(test)-1:-1,-1], reverted_predictions))
print('Test RMSE: %.3f' % rmse)
# line plot of observed vs predicted
plt.plot(df.iloc[-len(test)-1:-1,-1])
plt.plot(reverted_predictions)
plt.show()
df_triple_lag = create_lag_column(df,lags=[-1,-5,-20])

# Defining the training record size
triple_lag_train_record_size = round(0.9 * df_triple_lag.shape[0])
print("Number of records for Google price: " + str(df_triple_lag.shape[0]))
print("Number of training records to use: " + str(triple_lag_train_record_size))

# Train-Test Split
dataset = df_triple_lag.values
triple_lag_train, triple_lag_test = dataset[0:triple_lag_train_record_size], dataset[triple_lag_train_record_size:]
print("Train set size: " + str(len(triple_lag_train)))
print("Test set size: " + str(len(triple_lag_test)))

# Scaling
triple_lag_scaler, triple_lag_train_scaled, triple_lag_test_scaled = scale(triple_lag_train, triple_lag_test)
# fit the model
triple_lag_lstm_model = fit_lstm(triple_lag_train_scaled, 1, 20, 64, target_size=3)
# forecast the entire training dataset to build up state for forecasting
triple_lag_train_reshaped = triple_lag_train_scaled[:, 0:-3].reshape(len(triple_lag_train_scaled), 1, 5)
triple_lag_lstm_model.predict(triple_lag_train_reshaped, batch_size=1)

# walk-forward validation on the test data
triple_lag_predictions = list()
triple_lag_reverted_predictions = list()
for i in range(len(triple_lag_test_scaled)):
    # make one-step forecast
    X, y = triple_lag_test_scaled[i, 0:-3], triple_lag_test_scaled[i, -3]
    yhat = forecast_lstm(triple_lag_lstm_model, 1, X)
    # store forecast
    triple_lag_predictions.append(yhat)
    
    # invert scaling
    reverted_yhat = invert_scale(triple_lag_scaler, X, yhat, target_size=3)
    # store forecast
    triple_lag_reverted_predictions.append(reverted_yhat)
    
triple_lag_predictions = np.asarray(triple_lag_predictions)
triple_lag_reverted_predictions = np.asarray(triple_lag_reverted_predictions)
# report performance
combined_rmse = sqrt(mean_squared_error(triple_lag_test_scaled[:, -3:], triple_lag_predictions[:, -3:]))
lag_1_rmse = sqrt(mean_squared_error(triple_lag_test_scaled[:, -3], triple_lag_predictions[:, -3]))
lag_5_rmse = sqrt(mean_squared_error(triple_lag_test_scaled[:, -2], triple_lag_predictions[:, -2]))
lag_20_rmse = sqrt(mean_squared_error(triple_lag_test_scaled[:, -1], triple_lag_predictions[:, -1]))
print('Combined Test RMSE: %.3f' % combined_rmse)
print('Lag 1 Test RMSE: %.3f' % lag_1_rmse)
print('Lag 5 Test RMSE: %.3f' % lag_5_rmse)
print('Lag 20 Test RMSE: %.3f' % lag_20_rmse)
# line plot of observed vs predicted
plt.plot(triple_lag_test_scaled[:, -3])
plt.plot(triple_lag_predictions[:, -3])
plt.show()
# line plot of observed vs predicted
plt.plot(triple_lag_test_scaled[:, -2])
plt.plot(triple_lag_predictions[:, -2])
plt.show()
# line plot of observed vs predicted
plt.plot(triple_lag_test_scaled[:, -2])
plt.plot(triple_lag_predictions[:, -2])
plt.show()
# report performance
combined_rmse = sqrt(mean_squared_error(triple_lag_test[:, -3:], triple_lag_reverted_predictions[:, -3:]))
lag_1_rmse = sqrt(mean_squared_error(triple_lag_test[:, -3], triple_lag_reverted_predictions[:, -3]))
lag_5_rmse = sqrt(mean_squared_error(triple_lag_test[:, -2], triple_lag_reverted_predictions[:, -2]))
lag_20_rmse = sqrt(mean_squared_error(triple_lag_test[:, -1], triple_lag_reverted_predictions[:, -1]))
print('Combined Test RMSE: %.3f' % combined_rmse)
print('Lag 1 Test RMSE: %.3f' % lag_1_rmse)
print('Lag 5 Test RMSE: %.3f' % lag_5_rmse)
print('Lag 20 Test RMSE: %.3f' % lag_20_rmse)
# line plot of observed vs predicted
plt.plot(triple_lag_test[:, -3])
plt.plot(triple_lag_reverted_predictions[:, -3])
plt.show()
# line plot of observed vs predicted
plt.plot(triple_lag_test[:, -2])
plt.plot(triple_lag_reverted_predictions[:, -2])
plt.show()
# line plot of observed vs predicted
plt.plot(triple_lag_test[:, -1])
plt.plot(triple_lag_reverted_predictions[:, -1])
plt.show()
df_lag_20 = create_lag_column(df,lags=[-20])

# Defining the training record size
df_lag_20_train_record_size = round(0.9 * df_lag_20.shape[0])
print("Number of records for Google price: " + str(df_lag_20.shape[0]))
print("Number of training records to use: " + str(df_lag_20_train_record_size))

# Train-Test Split
dataset = df_lag_20.values
lag_20_train, lag_20_test = dataset[0:df_lag_20_train_record_size], dataset[df_lag_20_train_record_size:]
print("Train set size: " + str(len(lag_20_train)))
print("Test set size: " + str(len(lag_20_test)))

# Scaling
lag_20_scaler, lag_20_train_scaled, lag_20_test_scaled = scale(lag_20_train, lag_20_test)
# fit the model
lag_20_lstm_model = fit_lstm(lag_20_train_scaled, 1, 20, 64)
# forecast the entire training dataset to build up state for forecasting
lag_20_train_reshaped = lag_20_train_scaled[:, 0:-1].reshape(len(lag_20_train_scaled), 1, 5)
lag_20_lstm_model.predict(lag_20_train_reshaped, batch_size=1)

# walk-forward validation on the test data
lag_20_predictions = list()
lag_20_reverted_predictions = list()
for i in range(len(lag_20_test_scaled)):
    # make one-step forecast
    X, y = lag_20_test_scaled[i, 0:-1], lag_20_test_scaled[i, -1]
    yhat = forecast_lstm(lag_20_lstm_model, 1, X)
    # store forecast
    lag_20_predictions.append(yhat)
    
    # invert scaling
    reverted_yhat = invert_scale(lag_20_scaler, X, yhat)
    # store forecast
    lag_20_reverted_predictions.append(reverted_yhat)
    
lag_20_predictions = np.asarray(lag_20_predictions)
lag_20_reverted_predictions = np.asarray(lag_20_reverted_predictions)
# report performance
lag_20_rmse = sqrt(mean_squared_error(lag_20_test_scaled[:, -1], lag_20_predictions[:, -1]))
print('Lag 20 Test RMSE: %.3f' % lag_20_rmse)

# line plot of observed vs predicted
plt.plot(lag_20_test_scaled[:, -1])
plt.plot(lag_20_predictions[:, -1])
plt.show()
# report performance
lag_20_rmse = sqrt(mean_squared_error(lag_20_test[:, -1], lag_20_reverted_predictions[:, -1]))
print('Lag 20 Test RMSE: %.3f' % lag_20_rmse)

# line plot of observed vs predicted
plt.plot(lag_20_test[:, -1])
plt.plot(lag_20_reverted_predictions[:, -1])
plt.show()