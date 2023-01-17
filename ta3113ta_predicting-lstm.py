import math

import pandas_datareader as web

import numpy as np

import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential

from keras.layers import Dense, LSTM



import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
#ดึงข้อมูลของหุ้นที่เราต้องการมาเก็บไว้ใน DataFrame

df = web.DataReader('TFG', data_source='yahoo', start='1900-01-01', end='2222-01-01')

df.head(10)
#เช็คขนาดของข้อมูล

df.shape
# Plot เบี้องต้นออกมาดูราคากันหน่อย

plt.figure(figsize=(16,8))

plt.title('Closing Price')

plt.plot(df['Close'])

plt.xlabel('Date', fontsize=18)

plt.ylabel('Close Price', fontsize=18)

# ดึงเฉพาะราคาปิดมา

data = df.filter(['Close'])



#แปลงข้อมูลใน data ให้เป็น Numpy array

dataset = data.values



#แบ่งข้อมูลแบบ train 80, test 20

#ดูว่า train 80% มีขนาดเท่าไร

training_data_len = math.ceil(len(dataset) * 0.8)

training_data_len
# ปรับ Scaling

scaler = MinMaxScaler(feature_range=(0,1))

scaled_data = scaler.fit_transform(dataset)

# เลือก 80 % จาก scaled_data

train_data = scaled_data[:training_data_len, :]



x_train = [] 

y_train = []



# เลือกข้อมูลมา 60วัน(x_train) เพือทำนายวันถัดไป(y_train)

for i in range(60, len(train_data)):

    x_train.append(train_data[i-60:i, 0])

    y_train.append(train_data[i, 0])

    

#จากนั้นแปลงกับให้เป็น Numpy array

x_train, y_train = np.array(x_train), np.array(y_train)
# reshape

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))
# init model ขึ้นมาโดยใช้ LSTM

model = Sequential()

model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))

model.add(LSTM(50, return_sequences=False))

model.add(Dense(25))

model.add(Dense(1))



# Compile

model.compile(optimizer='adam', loss='mean_squared_error')

# ทำการ Train model

model.fit(x_train, y_train, batch_size=1, epochs=1)
# ดึง test data 20% ออกมา

test_data = scaled_data[training_data_len - 60: , :] 



x_test = [] 

y_test = dataset[training_data_len: :]



for i in range(60, len(test_data)):

    x_test.append(test_data[i-60:i, 0])



x_test = np.array(x_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# มาลองทำนาย model กันบ้าง

predictions = model.predict(x_test)



# transform scaler กลับเป็นค่าเดิม

predictions = scaler.inverse_transform(predictions)

# ผมจะใช้ metrics แบบ root mean squared error

rmse = np.sqrt(np.mean(predictions - y_test)**2)

rmse
# plot ค่าจริงกับค่าที่ทำนายไว้มาดูซักหน่อย

train = data[:training_data_len]

valid = data[training_data_len:]

valid['Predictions'] = predictions



plt.figure(figsize=(16,8))

plt.title('Model')

plt.xlabel('Date', fontsize=18)

plt.ylabel('Close Price', fontsize=18)

plt.plot(train['Close'])

plt.plot(valid[['Close', 'Predictions']])

plt.legend(['Train', 'Val', 'Predictions'] , loc='lower right')

plt.show()
# แสดงค่าที่ทำนาย กับ ค่าจริง

valid
# ดึง data มาอีกรอบเพื่ออ้างอิง

quote = web.DataReader('TFG', data_source='yahoo', start='1900-01-01', end='2222-01-01')



#สร้าง DataFrame มาใหม่

new_df = quote.filter(['Close'])



# ดึงราคาปิดจาก 60 วันล่าสุด

last_60_days = new_df[-60:].values



# Scale

last_60_days_scaled = scaler.transform(last_60_days)



X_test = []

X_test.append(last_60_days_scaled)

X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#predicted ข้อมูลกันหน่อย

pred_price = model.predict(X_test)



#Undo the scaling

pred_price = scaler.inverse_transform(pred_price)



quote2 = web.DataReader('TFG', data_source='yahoo', start='1900-01-01', end='2222-01-01')

quote2['Close']