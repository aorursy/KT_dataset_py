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
df.info()
'''Visualizing the Stock prices of a particular Stock "YHOO" from the dataset over the time (in days)'''

plt.figure(figsize=(15, 5))
plt.plot(df[df.symbol == 'YHOO'].open.values, color='red', label='open')
plt.plot(df[df.symbol == 'YHOO'].close.values, color='green', label='close')
plt.plot(df[df.symbol == 'YHOO'].low.values, color='blue', label='low')
plt.plot(df[df.symbol == 'YHOO'].high.values, color='black', label='high')
plt.title('Stock Price')
plt.xlabel('Time[days]')
plt.ylabel('Price')
plt.legend(loc='best')
plt.show()
'''This function will do the Min/Max normalization of the dataset'''

def normalize_data(df):
    min_max_scaler = MinMaxScaler()
    df['open'] = min_max_scaler.fit_transform(df.open.values.reshape(-1,1))
    df['high'] = min_max_scaler.fit_transform(df.high.values.reshape(-1,1))
    df['low'] = min_max_scaler.fit_transform(df.low.values.reshape(-1,1))
    df['close'] = min_max_scaler.fit_transform(df['close'].values.reshape(-1,1))
    return df
'''This function create the Train/Validation/Test data of a particular Stock data "YHOO" and sequence length'''

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
'''Choosing the Stock "YHOO" and removing the column "symbol" from the dataset'''

df_stock = df[df.symbol == 'YHOO'].copy()
df_stock.drop(['symbol'],1,inplace=True)
df_stock.drop(['volume'],1,inplace=True)

col = list(df_stock.columns.values)
print('Columns in the resulted dataset = ', col)
'''Normalizing the Stock'''

df_stock_norm = df_stock.copy()
df_stock_norm = normalize_data(df_stock_norm)
'''Create Train/Validation/Test data'''

seq_len = 20           #Choose sequence length
train_x, train_y, val_x, val_y, test_x, test_y = create_data(df_stock_norm, seq_len)

'''Summary of dataset'''

print("Summary of dataset:\n")
print('train_x = ',train_x.shape)
print('train_y = ', train_y.shape)
print('val_x = ',val_x.shape)
print('val_y = ', val_y.shape)
print('test_x = ', test_x.shape)
print('test_y = ',test_y.shape)
'''Visualizing the Normalized Stock prices per volume of a particular Stock "YHOO" from the dataset over the time (in days)'''

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
'''Making the LSTM model for the datset'''

lstm_model = Sequential()

lstm_model.add(LSTM(input_shape = (19, 4), units = 50, return_sequences=True)) #Adding LSTM layer
lstm_model.add(Dropout(0.2))                                                   #Adding Dropout

lstm_model.add(LSTM(100, return_sequences = False))                            #Adding LSTM layer
lstm_model.add(Dropout(0.2))                                                   #Adding Dropout

lstm_model.add(Dense(units=4))                                                 #Adding Dense layer with activation = "linear"
lstm_model.add(Activation('linear'))


'''Compiling the model'''

lstm_model.compile(loss='mse', optimizer='rmsprop')
'''Fitting the dataset into the model'''

lstm_model.fit(train_x, train_y, batch_size=128, epochs=100, validation_data=(val_x, val_y))
'''Predicted values of train/val/test dataset'''

train_y_pred = lstm_model.predict(train_x)
val_y_pred = lstm_model.predict(val_x)
test_y_pred = lstm_model.predict(test_x)
'''Visualizing the trained/predicted/test dataset'''

c = 0 # 0 = open, 1 = close, 2 = high, 3 = low

plt.figure(figsize=(15, 5));
plt.subplot(1,2,1);

plt.plot(np.arange(train_y.shape[0]), train_y[:, c], color='blue', label='Train target')

plt.plot(np.arange(train_y.shape[0], train_y.shape[0] + val_y.shape[0]), val_y[:, c], color='gray', label='Validation target')

plt.plot(np.arange(train_y.shape[0]+val_y.shape[0], train_y.shape[0]+test_y.shape[0]+test_y.shape[0]), test_y[:, c], color='black', label='Test target')

plt.plot(np.arange(train_y_pred.shape[0]),train_y_pred[:, c], color='red', label='Train Prediction')

plt.plot(np.arange(train_y_pred.shape[0], train_y_pred.shape[0]+val_y_pred.shape[0]), val_y_pred[:, c], color='orange', label='Validation Prediction')

plt.plot(np.arange(train_y_pred.shape[0]+val_y_pred.shape[0], train_y_pred.shape[0]+val_y_pred.shape[0]+test_y_pred.shape[0]), test_y_pred[:, c], color='green', label='Test Prediction')

plt.title('Past and Future Stock Prices')
plt.xlabel('Time [days]')
plt.ylabel('Normalized Price')
plt.legend(loc='best');

plt.subplot(1,2,2);
plt.plot(np.arange(train_y.shape[0], train_y.shape[0]+test_y.shape[0]), test_y[:, c], color='black', label='Test target')

plt.plot(np.arange(train_y_pred.shape[0], train_y_pred.shape[0]+test_y_pred.shape[0]), test_y_pred[:, c], color='green', label='Test Prediction')

plt.title('Future Stock Prices')
plt.xlabel('Time [days]')
plt.ylabel('Normalized Price')
plt.legend(loc='best');

train_acc = np.sum(np.equal(np.sign(train_y[:,1]-train_y[:,0]), np.sign(train_y_pred[:,1]-train_y_pred[:,0])).astype(int)) / train_y.shape[0]
val_acc = np.sum(np.equal(np.sign(val_y[:,1]-val_y[:,0]), np.sign(val_y_pred[:,1]-val_y_pred[:,0])).astype(int)) / val_y.shape[0]
test_acc = np.sum(np.equal(np.sign(test_y[:,1]-test_y[:,0]), np.sign(test_y_pred[:,1]-test_y_pred[:,0])).astype(int)) / test_y.shape[0]

print('Accuracy for Close - Open price for Train/Validation/Test Set: %.2f/%.2f/%.2f'%(train_acc, val_acc, test_acc))