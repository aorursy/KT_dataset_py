# importing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
# loading the stock prices of all companies in a dataframe

dataset = pd.read_csv('../input/historical_stock_prices.csv')

stocks = dataset.loc[(dataset['date']>='2017-01-01') & (dataset['date']<='2017-12-31')]
# getting the list of all companies 

companies = stocks.ticker.unique()

companies.sort()
from sklearn.preprocessing import MinMaxScaler



# we are creating 2 arrays, x_train and y_train.¶

# x_train stores the values of adjusted closing prices of past 60 days

# y_train stores the values of adjusted closing prices of the present day



period = 60

x_train = []

y_train = []

companies_sc = []



for company in companies:

    

    sc = MinMaxScaler()

    stock = stocks.loc[stocks['ticker'] == company]

    

    # creating an array with adjusted closing prices

    training_set = stock[['adj_close']].values

    

    # normalizing the values

    training_set_scaled = sc.fit_transform(training_set)

    training_set_scaled.shape

    

    # In the below cell, we are appendding data to x_train and y_train.¶

    

    length = len(training_set)

    for i in range(period, length):

        x_train.append(training_set_scaled[i-period:i, 0])

        y_train.append(training_set_scaled[i, 0])

        

    companies_sc.append(sc)

        

x_train = np.array(x_train)

y_train = np.array(y_train)

x_train.shape
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

x_train.shape
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import LSTM, Dense, Dropout



model = Sequential()



model.add(LSTM(units = 92, return_sequences = True, input_shape = (x_train.shape[1], 1)))

model.add(Dropout(0.2))



# model.add(LSTM(units = 92, return_sequences = True))

# model.add(Dropout(0.2))



# model.add(LSTM(units = 92, return_sequences = True))

# model.add(Dropout(0.2))



model.add(LSTM(units = 92, return_sequences = False))

model.add(Dropout(0.2))



model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')
train = model.fit(x_train, y_train, epochs = 10, batch_size = 3000, validation_split=0.33)
plt.plot(train.history['loss'])

plt.plot(train.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper right')

plt.show()
test_set = dataset.loc[(dataset['ticker'] == 'AAPL') & (dataset['date']>='2018-01-01')]  

test_set = test_set.loc[:, test_set.columns == 'adj_close']
y_test = test_set.iloc[period:, 0:].values
sc = companies_sc[np.where(companies=="AAPL")[0][0]]

# storing all values in a variable for generating an input array for our model 

adj_closing_price = test_set.iloc[:, 0:].values

adj_closing_price_scaled = sc.transform(adj_closing_price)
# the model will predict the values on x_test

x_test = [] 

length = len(test_set)



for i in range(period, length):

    x_test.append(adj_closing_price_scaled[i-period:i, 0])

    

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

x_test.shape
# predicting the stock price values

y_pred = model.predict(x_test)

predicted_price = sc.inverse_transform(y_pred)
# plotting the results

plt.plot(y_test, color = 'blue', label = 'Actual Stock Price')

plt.plot(predicted_price, color = 'red', label = 'Predicted Stock Price')

plt.title('Stock Price Prediction')

plt.xlabel('Time')

plt.ylabel('Stock Price')

plt.legend()

plt.show()
stock_train_set = dataset.loc[(dataset['ticker'] == 'AAPL') & 

                              (dataset['date']>='2017-06-01') & 

                              (dataset['date']<='2017-12-31')]

stock_train_set = stock_train_set.loc[:, stock_train_set.columns == 'adj_close']
y_train = stock_train_set.iloc[period:, 0:].values
stock_acp = stock_train_set.iloc[:, 0:].values

stock_acp_scaled = sc.transform(stock_acp)
x_train = [] 

length = len(stock_train_set)



for i in range(period, length):

    x_train.append(stock_acp_scaled[i-period:i, 0])

    

x_train = np.array(x_train)

x_train.shape
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

x_train.shape
# predicting the adjusted closing price values

y_train_pred = model.predict(x_train)

train_predicted_price = sc.inverse_transform(y_train_pred)
# plotting the results

plt.plot(y_train, color = 'blue', label = 'Actual Adjusted Closing Price')

plt.plot(train_predicted_price, color = 'red', label = 'Predicted Adjusted Closing Price')

plt.title('Adjusted Closing Price Prediction')

plt.xlabel('Time')

plt.ylabel('Adjusted Closing Price')

plt.legend()

plt.show()
from sklearn.metrics import mean_squared_error

# Calculate RMSE

trainScore = mean_squared_error(y_train, train_predicted_price)

print('Train Score: %.2f MSE' % (trainScore))

testScore = mean_squared_error(y_test, predicted_price)

print('Test Score: %.2f MSE' % (testScore))