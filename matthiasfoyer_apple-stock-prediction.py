# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Import libraries



import matplotlib.pyplot as plt

import datetime



# Import dataset



df = pd.read_csv(r'/kaggle/input/apple-aapl-historical-stock-data/HistoricalQuotes.csv', index_col='Date', parse_dates=True)
# Test if missing values exist



df.isna().any()
df.info()
# Cleaning data



df = df.rename(columns={' Close/Last':'Close', ' Volume':'Volume', ' Open': 'Open', ' High':'High', ' Low':'Low'})

df['Close'] = df['Close'].str.replace('$', '').astype('float')

df['Open'] = df['Open'].str.replace('$', '').astype('float')

df['High'] = df['High'].str.replace('$', '').astype('float')

df['Low'] = df['Low'].str.replace('$', '').astype('float')

df.head()
df.dtypes
# Split training and testing datasets



df_test = df.head(40)

df = df[40:]
# Moving average



df['Open'].plot(figsize=(16, 6))

df.rolling(100).mean()['Open'].plot()
training_df = df['Open']

training_df = pd.DataFrame(training_df)
# Feature scaling



from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0,1))

training_df_scaled = sc.fit_transform(training_df)
# Create structure with 60 timesteps and 1 output



X_train = []

y_train = []

for i in range(60, 2477):

    X_train.append(training_df_scaled[i-60:i, 0])

    y_train.append(training_df_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)





# Reshape



X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

X_train
# Import Keras



from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout
# Initialize RNN



regressor = Sequential()
# First LSTM layer



regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))

regressor.add(Dropout(0.2))





# Second LSTM layer



regressor.add(LSTM(units = 50, return_sequences = True))

regressor.add(Dropout(0.2))





# Thirs LSTM layer



regressor.add(LSTM(units = 50, return_sequences = True))

regressor.add(Dropout(0.2))





# Fourth LSTM layer



regressor.add(LSTM(units = 50))

regressor.add(Dropout(0.2))



              

# Output layer



regressor.add(Dense(units = 1))
# Compile RNN



regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')





# Fit RNN



regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
real_stock_price = df_test['Open'].values

real_stock_price
df_test.info()
test_set = df_test['Open']

test_set = pd.DataFrame(test_set)

test_set.info()
df_total = pd.concat((df['Open'], df_test['Open']), axis = 0)

inputs = df_total[len(df_total) - len(df_test) - 60:].values

inputs = inputs.reshape(-1,1)

inputs = sc.transform(inputs)

X_test = []

for i in range(60, 100):

    X_test.append(inputs[i-60:i, 0])

X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))



predicted_stock_price = regressor.predict(X_test)

predicted_stock_price = sc.inverse_transform(predicted_stock_price)
predicted_stock_price = pd.DataFrame(predicted_stock_price)

predicted_stock_price = predicted_stock_price.values

predicted_stock_price
real_stock_price
# Plot the results



plt.plot(real_stock_price, color = 'red', label = 'Real Apple Stock Price')

plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Apple Stock Price')

plt.title('Apple Stock Price Prediction')

plt.xlabel('Time')

plt.ylabel('Stock Price')

plt.legend()

plt.show()