import numpy as np 
import pandas as pd 
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
df = pd.read_csv("../input/nyse/prices-split-adjusted.csv", index_col = 0)
df.info()
df.head()

print('\nNumber of different Stocks: ', len(list(set(df.symbol))))
print(list(set(df.symbol))[:10])
df.tail()

df.describe()

plt.figure(figsize=(15, 5))
plt.plot(df[df.symbol == 'YHOO'].open.values, color='red', label='open')
plt.plot(df[df.symbol == 'YHOO'].close.values, color='green', label='close')
plt.plot(df[df.symbol == 'YHOO'].low.values, color='blue', label='low')
plt.plot(df[df.symbol == 'YHOO'].high.values, color='black', label='high')
plt.title('Stock Price')
plt.xlabel('Time[days]')
plt.ylabel('Price')
#plt.legend(loc='best')
plt.show()
def normalize_data(df):
    min_max_scaler = MinMaxScaler()
    df['open'] = min_max_scaler.fit_transform(df.open.values.reshape(-1,1))
    df['high'] = min_max_scaler.fit_transform(df.high.values.reshape(-1,1))
    df['low'] = min_max_scaler.fit_transform(df.low.values.reshape(-1,1))
    df['close'] = min_max_scaler.fit_transform(df['close'].values.reshape(-1,1))
    return df
def create_data(stock, seq_len):
    data_raw = stock.to_numpy()    #convert to numpy array
    data = []
    
    #Create all possible sequences of length seq_len
    
    for idx in range(len(data_raw) - seq_len): 
        data.append(data_raw[idx: idx + seq_len])
    percentage_of_val_set = 10 
    percentage_of_test_set = 10   
    data = np.array(data);
    validation_set_size = int(np.round(percentage_of_val_set/100*data.shape[0]));  
    test_set_size = int(np.round(percentage_of_test_set/100*data.shape[0]));
    train_set_size = data.shape[0] - (validation_set_size + test_set_size);
    
    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]
    
    x_valid = data[train_set_size:train_set_size+validation_set_size,:-1,:]
    y_valid = data[train_set_size:train_set_size+validation_set_size,-1,:]
    
    x_test = data[train_set_size+validation_set_size:,:-1,:]
    y_test = data[train_set_size+validation_set_size:,-1,:]
    
    return [x_train, y_train, x_valid, y_valid, x_test, y_test]

df_stock= pd.read_csv("../input/nyse/prices.csv", index_col = 0)

df_stock = df[df.symbol == 'YHOO'].copy()
df_stock.drop(['symbol'],1,inplace=True)
df_stock.drop(['volume'],1,inplace=True)

df_stock_norm = df_stock.copy()
df_stock_norm = normalize_data(df_stock_norm)
df_stock_norm.shape
seq_len = 20        
train_x, train_y, val_x, val_y, test_x, test_y = create_data(df_stock_norm, seq_len)

plt.figure(figsize=(15, 5));
plt.plot(df_stock_norm.open.values, color='red', label='open')
plt.plot(df_stock_norm.close.values, color='green', label='low')
plt.plot(df_stock_norm.low.values, color='blue', label='low')
plt.plot(df_stock_norm.high.values, color='black', label='high')


plt.title('Stock')
plt.xlabel('Time [days]')
plt.ylabel('Normalized price/Volume')
plt.legend(loc='best')
plt.show()


lstm_model = Sequential()

lstm_model.add(LSTM(input_shape = (19, 4), units = 50, return_sequences=True)) 
lstm_model.add(Dropout(0.2))                                                   

lstm_model.add(LSTM(100, return_sequences = False))                           
lstm_model.add(Dropout(0.2))                                                   

lstm_model.add(Dense(units=4))                                                 
lstm_model.add(Activation('linear'))


'''Compiling the model'''

lstm_model.compile(loss='mse', optimizer='rmsprop', metrics = ['accuracy'])


lstm_model.fit(train_x, train_y, batch_size=128, epochs=5, validation_data=(val_x, val_y))
closing_price = lstm_model.predict(test_x)
output=(np.mean(closing_price))
output