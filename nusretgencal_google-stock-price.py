# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import warnings

warnings.filterwarnings('ignore') # filter warnings



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data_train = pd.read_csv("/kaggle/input/gooogle-stock-price/Google_Stock_Price_Train.csv")

data_train.head()

train = data_train.loc[:,["Open"]].values # open valuelerine bakacağız values ile array'e çeviriyoruz

train
from sklearn.preprocessing import MinMaxScaler # 0-1 arasında scale yapacağız normalization

scaler = MinMaxScaler(feature_range = (0,1))

train_scaled = scaler.fit_transform(train) # train datamı alıp 0-1 arasına scale ediyoruz

train_scaled

plt.plot(train_scaled)

plt.show()
train_scaled.shape
# 50 timesteps ve 1 outputtan oluşan bir data structure kuruyoruz, 50 tane data al 51.sini tahmin et gibi..

x_train = []

y_train = []

timesteps = 50



for i in range(timesteps, train_scaled.shape[0]): #1258 'e kadar tüm colon sayısı # burada sorun çıkarsa 1258 yazabilirsin

    x_train.append(train_scaled[i - timesteps:i, 0]) # 0'dan 50'e kadar al x'e at

    y_train.append(train_scaled[i, 0]) # 50'den bir sonrakini y'e at

    

x_train, y_train = np.array(x), np.array(y)



y_train.shape
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) # 1208, 50 , 1

x_train.shape # x'mizin shape'i 3 boyutlu oldu
y_train.shape
x_train.shape[1]
from keras.models import Sequential

from keras.layers import Dense, SimpleRNN, Dropout # dense layer , rnn, dropout overfitting önlemek için



#initialize RNN



model = Sequential() # model oluşturduk



#ilk RNN LAYER ve dropout



model.add(SimpleRNN(units = 50, activation = "relu", return_sequences = True, input_shape = (x_train.shape[1], 1))) # 50x1 'lik bir input gireceğimizi söylüyoruz

model.add(Dropout(0.2)) # regularisation 



# 2. RNN layer ve dropout'



model.add(SimpleRNN(units = 50, activation = "relu", return_sequences = True )) # 50x1 'lik bir input gireceğimizi söylüyoruz

model.add(Dropout(0.2))



#3. RNN layer ve dropout



model.add(SimpleRNN(units = 50, activation = "relu", return_sequences = True )) # 50x1 'lik bir input gireceğimizi söylüyoruz # relu kullanarak deneyelim mi?

model.add(Dropout(0.2))



#4. RNN layer ve dropout



model.add(SimpleRNN(units = 50)) # 50x1 'lik bir input gireceğimizi söylüyoruz # relu kullanarak deneyelim mi?

model.add(Dropout(0.2))



# son olarak output layer dense ile



model.add(Dense(units = 1)) # 1 OUTPUT



# rnn compile ediyoruz



model.compile(optimizer = "Adam", loss = "mean_squared_error", metrics = ["accuracy"])



#fit



model.fit(x_train, y_train, epochs = 50, batch_size = 16)

data_test = pd.read_csv("/kaggle/input/gooogle-stock-price/Google_Stock_Price_Test.csv")

data_test.head()
real_stock_price = data_test.loc[:, ["Open"]].values

real_stock_price
data_total = pd.concat((data_train['Open'], data_test['Open']), axis = 0)

inputs = data_total[len(data_total) - len(data_test) - timesteps:].values.reshape(-1,1)

inputs = scaler.transform(inputs)  # min max scaler

inputs
inputs.shape[0]
x_test = []

for i in range(timesteps, inputs.shape[0]):

    x_test.append(inputs[i-timesteps:i, 0])

x_test = np.array(x_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1)) # 3 boyut

predicted_stock_price = model.predict(x_test)

predicted_stock_price = scaler.inverse_transform(predicted_stock_price)



# Visualising the results

plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')

plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')

plt.title('Google Stock Price Prediction')

plt.xlabel('Time')

plt.ylabel('Google Stock Price')

plt.legend()

plt.show()
from keras.models import Sequential

from keras.layers import LSTM, Dense, SimpleRNN, Dropout # dense layer , rnn, dropout overfitting önlemek için

model = Sequential() # model oluşturduk



#ilk LSTM LAYER ve dropout



model.add(LSTM(15, activation = "relu", input_shape = (x_train.shape[1], 1))) #10 LSTM block. One layer has 10 LSTM unit (node).

"""

model.add(LSTM(units = 50, activation = "relu", return_sequences = True, dropout = 0.2, input_shape = (x_train.shape[1], 1))) # 50x1 'lik bir input gireceğimizi söylüyoruz

#model.add(Dropout(0.2)) # regularisation 



# 2. LSTM layer ve dropout'



model.add(LSTM(units = 50, activation = "relu", return_sequences = True, dropout = 0.2)) # 50x1 'lik bir input gireceğimizi söylüyoruz

#model.add(Dropout(0.2))



#3. LSTM layer ve dropout



model.add(LSTM(units = 50, activation = "relu", return_sequences = True, dropout = 0.2)) # 50x1 'lik bir input gireceğimizi söylüyoruz # relu kullanarak deneyelim mi?

#model.add(Dropout(0.2))



#4. LSTM layer ve dropout



model.add(LSTM(units = 50, activation = "relu", dropout = 0.2)) # 50x1 'lik bir input gireceğimizi söylüyoruz # relu kullanarak deneyelim mi?

#model.add(Dropout(0.2))"""



# son olarak output layer dense ile



model.add(Dense(units = 1)) # 1 OUTPUT



# rnn compile ediyoruz



model.compile(optimizer = "Adam", loss = "mean_squared_error", metrics = ["accuracy"])



#fit



model.fit(x_train, y_train, epochs = 50, batch_size = 5)
predicted_data2 = model.predict(x_test)

predicted_data2 = scaler.inverse_transform(predicted_data2)
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')

plt.plot(predicted_stock_price, color = 'blue', label = 'RNN Predicted Google Stock Price')

plt.plot(predicted_data2, color = 'green', label = 'LSTM Predicted Google Stock Price')

plt.title('Google Stock Price Prediction')

plt.xlabel('Time')

plt.ylabel('Google Stock Price')

plt.legend()

plt.show()