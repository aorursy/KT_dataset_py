# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# loading the stock prices of all companies in a dataframe
dataset = pd.read_csv('../input/all_stocks_5yr.csv')
dataset.head()
dataset.info()
# getting the list of all companies 
companies = dataset.Name.unique()
companies
# since AAL company is used for training, we are creating a new dataframe with AAL parameters
stock = dataset.loc[dataset['Name'] == 'AAL']
stock.info()
stock.head()
# creating an array with closing prices
training_set = stock.iloc[:, 4:5].values
# normalizing the values
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set_scaled = sc.fit_transform(training_set)
training_set_scaled.shape
x_train = []
y_train = []
timestamp = 60
length = len(training_set)
for i in range(timestamp, length):
    x_train.append(training_set_scaled[i-timestamp:i, 0])
    y_train.append(training_set_scaled[i, 0])
    
x_train = np.array(x_train)
y_train = np.array(y_train)
print (x_train[0])
print ('\n')
print (y_train[0])
x_train.shape
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

model = Sequential()

model.add(LSTM(units = 92, return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units = 92, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 92, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 92, return_sequences = False))
model.add(Dropout(0.2))

model.add(Dense(units = 1))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(x_train, y_train, epochs = 25, batch_size = 32)
test_set = dataset.loc[dataset['Name'] == 'SIG']   # change CBS to whatever company from the list
test_set = test_set.loc[:, test_set.columns == 'close']
# storing the actual stock prices in y_test starting from 60th day as the previous 60 days are used to predict the present day value.
y_test = test_set.iloc[timestamp:, 0:].values
# storing all values in a variable for generating an input array for our model 
closing_price = test_set.iloc[:, 0:].values
closing_price_scaled = sc.transform(closing_price)
# the model will predict the values on x_test
x_test = [] 
length = len(test_set)

for i in range(timestamp, length):
    x_test.append(closing_price_scaled[i-timestamp:i, 0])
    
x_test = np.array(x_test)
x_test.shape
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
