import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras import optimizers

pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv("../input/stock-time-series-20050101-to-20171231/all_stocks_2006-01-01_to_2018-01-01.csv",index_col='Date', parse_dates=['Date'])
df.head()
df.shape
df.info()
df['Name'].unique()
name_count = df.groupby(['Name']).count()
print(name_count['Volume'].unique())
stock = list(df['Name'].unique())
df_ = {}
for i in stock:
    df_[i] = df.loc[df['Name']==i]
def splitDataset(data, date, col):
    return data.loc[:date,col], data.loc[date:,col]
df_new_ = {}
for i in stock:
    df_new_[i] = {}
    df_new_[i]['Train'], df_new_[i]['Test'] = splitDataset(df_[i], '2016', 'High')
for i in stock:
    plt.figure(figsize=(10,4))
    plt.plot(df_new_[i]['Train'])
    plt.plot(df_new_[i]['Test'])
    plt.ylabel("Price")
    plt.xlabel("Date")
    plt.legend(["Training Set", "Test Set"])
    plt.title(i + " Highest Stock Price")

for i in stock:
    df_new_[i]['Train'] = pd.DataFrame(df_new_[i]['Train']).dropna()
    df_new_[i]['Test'] = pd.DataFrame(df_new_[i]['Test']).dropna()
sc= {}
for i in stock:
    df_new_[i]['Train'] = df_new_[i]['Train'].values.reshape(-1,1)
    df_new_[i]['Test'] = df_new_[i]['Test'].values.reshape(-1,1)
    scaler =  MinMaxScaler()
    df_new_[i]['Train'] = scaler.fit_transform(df_new_[i]['Train'])
    df_new_[i]['Test'] = scaler.transform(df_new_[i]['Test'])
    sc[i] = scaler
train_count = 10000000000
test_count = 100000000
for i in stock:
    print(i, df_new_[i]['Train'].shape, df_new_[i]['Test'].shape)
    train_count = min(train_count, df_new_[i]['Train'].shape[0])
    test_count = min(test_count, df_new_[i]['Test'].shape[0])
time_steps = 75
train = {}
for i in stock:
    train[i] = {}
    X_train = []
    y_train = []
    for j in range(time_steps, train_count):
        X_train.append(df_new_[i]['Train'][j-time_steps:j,0])
        y_train.append(df_new_[i]['Train'][j,0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    train[i]['X_train'] = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    train[i]['y_train'] = y_train
    print(i, train[i]['X_train'].shape, train[i]['y_train'].shape)
test = {}
for i in stock:
    test[i] = {}
    X_test = []
    y_test = []
    for j in range(time_steps, test_count):
        X_test.append(df_new_[i]['Test'][j-time_steps:j,0])
        y_test.append(df_new_[i]['Test'][j,0])
    X_test, y_test = np.array(X_test), np.array(y_test)
    test[i]['X_test'] = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    test[i]['y_test'] = y_test
    print(i, test[i]['X_test'].shape, test[i]['y_test'].shape)
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.4))

model.add(Dense(units=1))
optimizer = optimizers.Adam(lr=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

for i in stock:
    print("Fitting to", i)
    model.fit(train[i]['X_train'], train[i]['y_train'], epochs=10, batch_size=150)
for i in stock:
    y_true = sc[i].inverse_transform(test[i]['y_test'].reshape(-1,1))
    predicted_stock_price = model.predict(test[i]['X_test'])
    y_pred = sc[i].inverse_transform(predicted_stock_price)
    
    plt.figure(figsize=(10,4))
    plt.plot(y_true)
    plt.plot(y_pred)
    plt.legend(['True', 'Predicted'])
    plt.title(i)
rmse = math.sqrt(mean_squared_error(y_true, y_pred))
print('Root mean square error = {}'.format(rmse))