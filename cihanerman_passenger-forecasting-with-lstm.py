import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.preprocessing import MinMaxScaler

from keras.preprocessing.sequence import TimeseriesGenerator

from pandas.tseries.offsets import DateOffset

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout

import warnings

warnings.filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("../input/air-passengers/AirPassengers.csv")

data.head()
data["Month"] = pd.to_datetime(data["Month"])

data.head()
data.set_index("Month", inplace=True)

data.columns = ["passengers"]

data.index.name = "date"

data.head()
data.info()

print(f"Dataset shape: {data.shape}")
train_data = data[:len(data)-12]

test_data = data[len(data)-12:]
scaler = MinMaxScaler()

scaler.fit(train_data)

train = scaler.transform(train_data)

test = scaler.transform(test_data)
n_input = 12

n_features = 1

generator = TimeseriesGenerator(train, train, length=n_input, batch_size=6)
model = Sequential()

model.add(LSTM(200, activation='relu', input_shape=(n_input, n_features)))

model.add(Dropout(0.15))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

model.fit_generator(generator,epochs=90, verbose= 0)

model.summary()
pred_list = []

batch = train[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):   

    pred_list.append(model.predict(batch)[0]) 

    batch = np.append(batch[:,1:,:],[[pred_list[i]]],axis=1)
df_predict = pd.DataFrame(scaler.inverse_transform(pred_list),index=data[-n_input:].index, columns=['Prediction'])

df_test = pd.concat([data,df_predict], axis=1)
df_test.tail(13)
plt.figure(figsize=(20, 5))

plt.plot(df_test.index, df_test['passengers'])

plt.plot(df_test.index, df_test['Prediction'], color='r')

plt.legend(loc='best', fontsize='xx-large')

plt.xticks(fontsize=18, color= "white")

plt.yticks(fontsize=16, color= "white")

plt.show()
train = data

scaler.fit(train)

train = scaler.transform(train)

n_input = 12

n_features = 1

generator = TimeseriesGenerator(train, train, length=n_input, batch_size=6)

model.fit_generator(generator,epochs=90, verbose= 0);
pred_list = []  

batch = train[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):

    pred_list.append(model.predict(batch)[0])      

    batch = np.append(batch[:,1:,:],[[pred_list[i]]],axis=1)
add_dates = [data.index[-1] + DateOffset(months=x) for x in range(0,13) ]

future_dates = pd.DataFrame(index=add_dates[1:],columns=data.columns)

future_dates.head(12)
df_predict = pd.DataFrame(scaler.inverse_transform(pred_list),

                          index=future_dates[-n_input:].index, columns=['Prediction'])



df_proj = pd.concat([data,df_predict], axis=1)
plt.figure(figsize=(20, 5))

plt.plot(df_proj.index, df_proj['passengers'])

plt.plot(df_proj.index, df_proj['Prediction'], color='r')

plt.legend(loc='best', fontsize='xx-large')

plt.xticks(fontsize=18, color = "white")

plt.yticks(fontsize=16, color = "white")

plt.show()
losses_lstm = model.history.history['loss']

plt.figure(figsize=(12,4))

plt.xlabel("Epochs", color = "white")

plt.ylabel("Loss", color = "white")

plt.xticks(  color = "white")

plt.yticks(  color = "white")

plt.plot(range(len(losses_lstm)),losses_lstm);