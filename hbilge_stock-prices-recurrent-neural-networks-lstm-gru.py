import warnings
warnings.filterwarnings('ignore')
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight') 
%matplotlib inline
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
def plot(y_true, y_prediction):
    plt.plot(y_true, color='red', label='Actual Stock Prices')
    plt.plot(y_prediction, color='blue', label='Predicted Stock Prices')
    plt.title(str.capitalize(key))
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
dataset = pd.read_csv("../input/stock-time-series-20050101-to-20171231/all_stocks_2006-01-01_to_2018-01-01.csv", index_col="Date")
print(dataset.shape)
print(dataset.Name.unique())
microsoft = dataset[dataset['Name']=='MSFT']
google = dataset[dataset['Name']=='GOOGL']
apple = dataset[dataset['Name']=='AAPL']
amazon = dataset[dataset['Name']=='AMZN']
ibm = dataset[dataset['Name']=='IBM']
print(dataset.duplicated().sum())
stocks = [microsoft, google, apple, amazon, ibm]
for i in stocks:
    print('\033[1mNULL VALUES\033[0m\n'+ str(i.isnull().sum()))
ibm.fillna(method='ffill', inplace=True)
print(ibm.isnull().sum().any())
microsoft.plot(subplots=True, figsize=(10,12))
plt.suptitle('Microsoft stock values from 2006 to 2018', fontsize=15)
plt.show()
google.plot(subplots=True, figsize=(10,12))
plt.suptitle('Google stock values from 2006 to 2018', fontsize=15)
plt.show()
apple.plot(subplots=True, figsize=(10,12))
plt.suptitle('Apple stock values from 2006 to 2018', fontsize=15)
plt.show()
amazon.plot(subplots=True, figsize=(10,12))
plt.suptitle('Amazon stock values from 2006 to 2018', fontsize=15)
plt.show()
ibm.plot(subplots=True, figsize=(10,12))
plt.suptitle('Ibm stock values from 2006 to 2018', fontsize=15)
plt.show()
plt.figure(figsize=(25,25))

plt.subplot(5,1,1)
microsoft.High.pct_change().mul(100).plot()
plt.title("Microsoft Daily Highest Price Variation in Stocks")

plt.subplot(5,1,2)
google.High.pct_change().mul(100).plot()
plt.title("Google Daily Highest Price Variation in Stocks")

plt.subplot(5,1,3)
apple.High.pct_change().mul(100).plot()
plt.title("Apple Daily Highest Price Variation in Stocks")

plt.subplot(5,1,4)
amazon.High.pct_change().mul(100).plot()
plt.title("Amazon Daily Highest Price Variation in Stocks")

plt.subplot(5,1,5)
ibm.High.pct_change().mul(100).plot()
plt.title("IBM Daily Highest Price Variation in Stocks")

plt.show()
microsoft["High"].plot(figsize=(16,4),legend=True)
google["High"].plot(figsize=(16,4),legend=True)
apple["High"].plot(figsize=(16,4),legend=True)
amazon["High"].plot(figsize=(16,4),legend=True)
ibm["High"].plot(figsize=(16,4),legend=True)
plt.legend(['Microsoft','Google', 'Apple', 'Amazon', 'IBM'])
plt.show()
microsoft["Low"].plot(figsize=(16,4),legend=True)
google["Low"].plot(figsize=(16,4),legend=True)
apple["Low"].plot(figsize=(16,4),legend=True)
amazon["Low"].plot(figsize=(16,4),legend=True)
ibm["Low"].plot(figsize=(16,4),legend=True)
plt.legend(['Microsoft','Google', 'Apple', 'Amazon', 'IBM'])
plt.show()
def split(data):
    return data[:'2016'].iloc[:, 1:2].values, data['2016':].iloc[:,1:2].values

microsoft_train, microsoft_test = split(microsoft)
google_train, google_test = split(google)
apple_train, apple_test = split(apple)
amazon_train, amazon_test = split(amazon)
ibm_train, ibm_test = split(ibm)
print("\033[1mShapes of train and test sets\033[0m\n")
print("Microsoft train set:", microsoft_train.shape, "Microsoft test set:", microsoft_test.shape)
print("Google train set:", google_train.shape, "Google test set:", google_test.shape)
print("Apple train set:", apple_train.shape, "Apple test set:", apple_test.shape)
print("Amazon train set:", amazon_train.shape, "Amazon test set:", amazon_test.shape)
print("IBM train set:", ibm_train.shape, "IBM test set:", ibm_test.shape)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
microsoft_train_scaled, microsoft_test_scaled = sc.fit_transform(microsoft_train), sc.fit_transform(microsoft_test)
google_train_scaled, google_test_scaled = sc.fit_transform(google_train), sc.fit_transform(google_test)
apple_train_scaled, apple_test_scaled = sc.fit_transform(apple_train), sc.fit_transform(apple_test)
amazon_train_scaled, amazon_test_scaled = sc.fit_transform(amazon_train), sc.fit_transform(amazon_test)
ibm_train_scaled, ibm_test_scaled = sc.fit_transform(ibm_train), sc.fit_transform(ibm_test)
train_map = {"microsoft": microsoft_train_scaled, "google": google_train_scaled, "apple": apple_train_scaled, "amazon": amazon_train_scaled, "ibm": ibm_train_scaled}
train_map_X = {"microsoft": [], "google": [], "apple": [], "amazon": [], "ibm": []}
train_map_y = {"microsoft": [], "google": [], "apple": [], "amazon": [], "ibm": []}

for key in train_map.keys():
    for i in range(60,2516):   
        train_map_X[key].append(train_map[key][i-60:i,0])
        train_map_y[key].append(train_map[key][i, 0])
    train_map_X[key], train_map_y[key] = np.array(train_map_X[key]), np.array(train_map_y[key])
    train_map_X[key] = np.reshape(train_map_X[key], (train_map_X[key].shape[0],train_map_X[key].shape[1],1))
for key in train_map_X.keys():
    
    # The LSTM architecture
    lstm = Sequential()

    # First LSTM layer with Dropout regularisation
    lstm.add(LSTM(units=50, return_sequences=True, input_shape=(train_map_X[key].shape[1],1)))
    lstm.add(Dropout(0.2))

    # Second LSTM layer
    lstm.add(LSTM(units=50, return_sequences=True))
    lstm.add(Dropout(0.2))

    # Third LSTM layer
    lstm.add(LSTM(units=50, return_sequences=True))
    lstm.add(Dropout(0.2))

    # Fourth LSTM layer
    lstm.add(LSTM(units=50))
    lstm.add(Dropout(0.2))

    # The output layer
    lstm.add(Dense(units=1))

    # Compiling the RNN
    lstm.compile(optimizer='rmsprop',loss='mean_squared_error')
    
    # Fitting to the training set
    print("Training:", key)
    lstm.fit(train_map_X[key],train_map_y[key],epochs=20,batch_size=32)

test_map = {"microsoft": microsoft_test_scaled, "google": google_test_scaled, "apple": apple_test_scaled, "amazon": amazon_test_scaled, "ibm": ibm_test_scaled}
test_map_X = {"microsoft": [], "google": [], "apple": [], "amazon": [], "ibm": []}
test_map_y = {"microsoft": [], "google": [], "apple": [], "amazon": [], "ibm": []}

for key in test_map.keys():
    for i in range(60,503):
        test_map_X[key].append(test_map[key][i-60:i,0])
        test_map_y[key].append(test_map[key][i, 0])
    test_map_X[key], test_map_y[key]= np.array(test_map_X[key]), np.array(test_map_y[key])
    test_map_X[key] = np.reshape(test_map_X[key], (test_map_X[key].shape[0], test_map_X[key].shape[1], 1))
for key in test_map_X.keys():
    y_true = sc.inverse_transform(test_map_y[key].reshape(-1,1)) 
    y_prediction = sc.inverse_transform(lstm.predict(test_map_X[key]))   
    plot(y_true, y_prediction)
    rmse = sqrt(mean_squared_error(y_true, y_prediction))
    print(str.capitalize(key), "Root Mean Squared Error:", rmse)
for key in train_map_X.keys():
    
    # The GRU architecture
    gru = Sequential()
    
    # First GRU layer with Dropout regularisation

    gru.add(GRU(units=50, return_sequences=True, input_shape=(train_map_X[key].shape[1],1), activation='tanh'))
    gru.add(Dropout(0.2))
    
    # Second GRU layer
    gru.add(GRU(units=50, return_sequences=True, input_shape=(train_map_X[key].shape[1],1), activation='tanh'))
    gru.add(Dropout(0.2))
    
    # Third GRU layer
    gru.add(GRU(units=50, return_sequences=True, input_shape=(train_map_X[key].shape[1],1), activation='tanh'))
    gru.add(Dropout(0.2))
    
    # Fourth GRU layer
    gru.add(GRU(units=50, activation='tanh'))
    gru.add(Dropout(0.2))
    
    # The output layer
    gru.add(Dense(units=1))
    
    # Compiling the RNN
    gru.compile(optimizer=SGD(),loss='mean_squared_error')
    
    # Fitting to the training set
    print("Training:", key)
    gru.fit(train_map_X[key],train_map_y[key], epochs=20,batch_size=150)
test_map = {"microsoft": microsoft_test_scaled, "google": google_test_scaled, "apple": apple_test_scaled, "amazon": amazon_test_scaled, "ibm": ibm_test_scaled}
test_map_X = {"microsoft": [], "google": [], "apple": [], "amazon": [], "ibm": []}
test_map_y = {"microsoft": [], "google": [], "apple": [], "amazon": [], "ibm": []}

for key in test_map.keys():
    for i in range(60,503):
        test_map_X[key].append(test_map[key][i-60:i,0])
        test_map_y[key].append(test_map[key][i, 0])
    test_map_X[key], test_map_y[key]= np.array(test_map_X[key]), np.array(test_map_y[key])
    test_map_X[key] = np.reshape(test_map_X[key], (test_map_X[key].shape[0], test_map_X[key].shape[1], 1))
for key in test_map_X.keys():
    y_true = sc.inverse_transform(test_map_y[key].reshape(-1,1)) 
    y_prediction = sc.inverse_transform(gru.predict(test_map_X[key]))   
    plot(y_true, y_prediction)
    rmse = sqrt(mean_squared_error(y_true, y_prediction))
    print(str.capitalize(key), "Root Mean Squared Error:", rmse)