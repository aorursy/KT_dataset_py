import pandas as pd

import plotly.express as px

from copy import copy

from scipy import stats

import matplotlib.pyplot as plt

import numpy as np

import plotly.figure_factory as ff

from sklearn.linear_model import LinearRegression

from sklearn.svm import SVR

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

from tensorflow import keras

# Read stock prices data

stock_price_df = pd.read_csv(r'/kaggle/input/capital-asset-pricing-model-capm/stock.csv')

stock_price_df.head()
# Read the stocks volume data

stock_vol_df = pd.read_csv(r'/kaggle/input/stock-volume/stock_volume.csv')

stock_vol_df.head()
# Sort the data based on Date

stock_price_df = stock_price_df.sort_values(by = ['Date'])

stock_price_df.head()
# Sort the data based on Date

stock_vol_df = stock_vol_df.sort_values(by = ['Date'])

stock_vol_df.head()
# Check if Null values exist in stock prices data

stock_price_df.isnull().sum()
# Check if Null values exist in stocks volume data

stock_vol_df.isnull().sum()
# Get stock prices dataframe info

stock_price_df.info()
# Get stock volume dataframe info

stock_vol_df.info()
stock_vol_df.describe()
#What is the average trading volume for Apple stock?

print("Average trading volume for Apple stock is",stock_vol_df.AAPL.mean())

#What is the maximum trading volume for sp500?

print("Maximum trading volume for S&P500 is",stock_vol_df.sp500.max())
#Which security is traded the most? Explain it .

print('''The S&P 500 index is a broad-based measure of large corporations traded on U.S. stock markets. 

Over long periods of time, passively holding the index often produces better results than actively trading or picking single stocks. 

Over long-time horizons, the index typically produces better returns than actively managed portfolios.''')
#What is the average stock price of the S&P500 over the specified time period?

print("Average stock price of the S&P500 over the specified time period",stock_vol_df.sp500.mean())
# Function to normalize stock prices based on their initial price

def normalize(df):

  x = df.copy()

  for i in x.columns[1:]:

    x[i] = x[i]/x[i][0]

  return x
# Function to plot interactive plots using Plotly Express

def interactive_plot(df, title):

  fig = px.line(title = title)

  for i in df.columns[1:]:

    fig.add_scatter(x = df['Date'], y = df[i], name = i)

  fig.show()
# plot interactive chart for stocks data

interactive_plot(stock_price_df, 'Stock Prices')
#Plot the volume dataset for all stocks, list any observations we might see

interactive_plot(stock_vol_df, 'Stocks Volume')



# S&P500 trading is orders of magnitude compared to individual stocks
#Plot the normalized stock prices and volume dataset.

# plot interactive chart for normalized stocks prices data

interactive_plot(normalize(stock_price_df), 'Stock Prices')
# Let's normalize the data and re-plot interactive chart for volume data

interactive_plot(normalize(stock_vol_df), 'Normalized Volume')
# Function to concatenate the date, stock price, and volume in one dataframe

def individual_stock(price_df, vol_df, name):

    return pd.DataFrame({'Date': price_df['Date'], 'Close': price_df[name], 'Volume': vol_df[name]})
# Function to return the input/output (target) data for AI/ML Model

# Note that our goal is to predict the future stock price 

# Target stock price today will be tomorrow's price 

def trading_window(data):

  

  # 1 day window 

  n = 1



  # Create a column containing the prices for the next 1 days

  data['Target'] = data[['Close']].shift(-n)

  

  # return the new dataset 

  return data
# Let's test the functions and get individual stock prices and volumes for AAPL

price_volume_df = individual_stock(stock_price_df, stock_vol_df, 'AAPL')

price_volume_df.head()
price_volume_target_df = trading_window(price_volume_df)

price_volume_target_df.head()
# Remove the last row as it will be a null value

price_volume_target_df = price_volume_target_df[:-1]

price_volume_target_df.head()
# Scale the data

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

price_volume_target_scaled_df = sc.fit_transform(price_volume_target_df.drop(columns = ['Date']))
price_volume_target_scaled_df
price_volume_target_scaled_df.shape
# Creating Feature and Target

X = price_volume_target_scaled_df[:,:2]

y = price_volume_target_scaled_df[:,2:]
# Converting dataframe to arrays

# X = np.asarray(X)

# y = np.asarray(y)

X.shape, y.shape
# Spliting the data this way, since order is important in time-series

# Note that we did not use train test split with it's default settings since it shuffles the data

split = int(0.65 * len(X))

X_train = X[:split]

y_train = y[:split]

X_test = X[split:]

y_test = y[split:]
X_train.shape, y_train.shape
X_test.shape, y_test.shape
# Define a data plotting function

def show_plot(data, title):

  plt.figure(figsize = (13, 5))

  plt.plot(data, linewidth = 1)

  plt.title(title)

  plt.grid()



show_plot(X_train, 'Training Data')

show_plot(X_test, 'Testing Data')

#Test the created pipeline with S&P500 and Amazon datasets

# Let's test the functions and get individual stock prices and volumes for S&P500

price_volume_df = individual_stock(stock_price_df, stock_vol_df, 'sp500')

price_volume_df.head()
# Let's test the functions and get individual stock prices and volumes for Amazon 

price_volume_df = individual_stock(stock_price_df, stock_vol_df, 'AMZN')

price_volume_df.head()

from sklearn.linear_model import Ridge

# Note that Ridge regression performs linear least squares with L2 regularization.

# Create and train the Ridge Linear Regression  Model

regression_model = Ridge()

regression_model.fit(X_train, y_train)
# Test the model and calculate its accuracy 

lr_accuracy = regression_model.score(X_test, y_test)

print("Linear Regression Score: ", lr_accuracy)
# Make Prediction

predicted_prices = regression_model.predict(X)

predicted_prices
# Append the predicted values into a list

Predicted = []

for i in predicted_prices:

  Predicted.append(i[0])
len(Predicted)
# Append the close values to the list

close = []

for i in price_volume_target_scaled_df:

  close.append(i[0])

# Create a dataframe based on the dates in the individual stock data

df_predicted = price_volume_target_df[['Date']]

df_predicted.head()
# Add the close values to the dataframe

df_predicted['Close'] = close

df_predicted.head()
# Add the predicted values to the dataframe

df_predicted['Prediction'] = Predicted

df_predicted.head()
from sklearn.metrics import mean_squared_error

mean_squared_error( df_predicted['Prediction'], df_predicted['Close'])**0.5
# Plot the results

interactive_plot(df_predicted, "Original Vs. Prediction")
from sklearn.linear_model import Ridge

# Note that Ridge regression performs linear least squares with L2 regularization.

# Create and train the Ridge Linear Regression  Model

regression_model = Ridge(alpha = 2)

regression_model.fit(X_train, y_train)
# Test the model and calculate its accuracy 

lr_accuracy = regression_model.score(X_test, y_test)

print("Quadratic Regression Score: ", lr_accuracy)

# Make Prediction

predicted_prices = regression_model.predict(X)

predicted_prices
# Append the predicted values into a list

Predicted = []

for i in predicted_prices:

  Predicted.append(i[0])
len(Predicted)
# Append the close values to the list

close = []

for i in price_volume_target_scaled_df:

  close.append(i[0])

# Create a dataframe based on the dates in the individual stock data

df_predicted = price_volume_target_df[['Date']]

df_predicted.head()
# Add the close values to the dataframe

df_predicted['Close'] = close

df_predicted.head()
# Add the predicted values to the dataframe

df_predicted['Prediction'] = Predicted

df_predicted.head()
from sklearn.metrics import mean_squared_error

mean_squared_error( df_predicted['Prediction'], df_predicted['Close'])**0.5
# Plot the results

interactive_plot(df_predicted, "Original Vs. Prediction")
# Let's test the functions and get individual stock prices and volumes for AAPL

price_volume_df = individual_stock(stock_price_df, stock_vol_df, 'sp500')

price_volume_df.head()
# Get the close and volume data as training data (Input)

training_data = price_volume_df.iloc[:, 1:3].values

training_data
# Normalize the data

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))

training_set_scaled = sc.fit_transform(training_data)
# Create the training and testing data, training data contains present day and previous day values

X = []

y = []

for i in range(1, len(price_volume_df)):

    X.append(training_set_scaled [i-1:i, 0])

    y.append(training_set_scaled [i, 0])
X
# Convert the data into array format

X = np.asarray(X)

y = np.asarray(y)
# Split the data

split = int(0.7 * len(X))

X_train = X[:split]

y_train = y[:split]

X_test = X[split:]

y_test = y[split:]
# Reshape the 1D arrays to 3D arrays to feed in the model

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

X_train.shape, X_test.shape
# Create the model

inputs = keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2]))

x = keras.layers.LSTM(50, return_sequences= True)(inputs)

x = keras.layers.Dropout(0.3)(x)

x = keras.layers.LSTM(50, return_sequences=True)(x)

x = keras.layers.Dropout(0.3)(x)

x = keras.layers.LSTM(50)(x)

outputs = keras.layers.Dense(1, activation='linear')(x)



model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss="mse")

model.summary()
# Trail the model

history = model.fit(

    X_train, y_train,

    epochs = 100,

    batch_size = 32,

    validation_split = 0.2

)
# Make prediction

predicted = model.predict(X)
# Append the predicted values to the list

test_predicted = []



for i in predicted:

  test_predicted.append(i[0])
test_predicted
df_predicted = price_volume_df[1:][['Date']]

df_predicted.head()
df_predicted['predictions'] = test_predicted

df_predicted.head()
# Plot the data

close = []

for i in training_set_scaled:

  close.append(i[0])

df_predicted['Close'] = close[1:]

df_predicted.head()
from sklearn.metrics import mean_squared_error

mean_squared_error( df_predicted['predictions'], df_predicted['Close'])**0.5
# Plot the results

interactive_plot(df_predicted, "Original Vs. Prediction")
#Experiment with various LSTM model parameters (Ex: Use 150 units instead of 50), print out the model summary and retrain the model

# Create the model

inputs = keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2]))

x = keras.layers.LSTM(150, return_sequences= True)(inputs)

x = keras.layers.Dropout(0.3)(x)

x = keras.layers.LSTM(150, return_sequences=True)(x)

x = keras.layers.Dropout(0.3)(x)

x = keras.layers.LSTM(150)(x)

outputs = keras.layers.Dense(1, activation='linear')(x)



model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss="mse")

model.summary()
# Trail the model

history = model.fit(

    X_train, y_train,

    epochs = 100,

    batch_size = 32,

    validation_split = 0.2

)
# Make prediction

predicted = model.predict(X)
# Append the predicted values to the list

test_predicted = []



for i in predicted:

  test_predicted.append(i[0])



test_predicted
df_predicted = price_volume_df[1:][['Date']]

df_predicted.head()

df_predicted['predictions'] = test_predicted

df_predicted.head()
# Plot the data

close = []

for i in training_set_scaled:

  close.append(i[0])

df_predicted['Close'] = close[1:]

df_predicted.head()
from sklearn.metrics import mean_squared_error

mean_squared_error( df_predicted['predictions'], df_predicted['Close'])**0.5
# Plot the results

interactive_plot(df_predicted, "Original Vs. Prediction")