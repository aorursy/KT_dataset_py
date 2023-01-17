#import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
import pandas_datareader as web
df = web.DataReader('AAPL', data_source = 'yahoo', start = '2010-01-01', end = '2020-09-09')
#df.head(5)
df.tail(12)
#df.shape
df = web.DataReader('AAPL', data_source = 'yahoo', start = '2010-01-01', end = '2020-08-31')
#df.head(5)
df.tail(5)
#df.shape
df.columns.tolist()
#check null values
df.isnull().sum().sort_values(ascending = False)
#data visulization
with plt.style.context('dark_background'):
    plt.figure(figsize = (20,10))
    plt.title('Price History of Yahoo stock', fontsize = 22)

    plt.xlabel('Date', fontsize = 22)
    plt.ylabel('Close Price $', fontsize = 22)
    plt.plot(df['Close'])
plt.show()
#target column
df_target = df.filter(['Close']).values
df_target.shape
df_train_len = int((df_target.shape[0]-60)* 0.8 + 60) #since we will use each 60 days of data
df_train_len
#scale the data
scaler = MinMaxScaler(feature_range = (0,1))
df_target_scaled = scaler.fit_transform(df_target)
#split train and test data
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_data = df_target_scaled[0:df_train_len, :]
test_data = df_target_scaled[df_train_len - 60:, :]

#create new X and y training datasets
X_train = []
y_train = []
for i in range (60, len(train_data)):
    X_train.append(train_data[i-60:i, ]) #each X represent 60 days previous data
    y_train.append(train_data[i, 0])

#create new X and y testing datasets
X_test = []
y_test = df_target_scaled[df_train_len:, :] #actual data
for i in range (60, len(test_data)):
    X_test.append(test_data[i-60:i, ]) #each X represent 60 days previous data
    #y_train.append(test_data[i, 0])

X_train, y_train, X_test = np.array(X_train), np.array(y_train), np.array(X_test)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_train.shape
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
X_test.shape
#import unsupervised learning models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
#build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(LSTM(50, return_sequences = False))
model.add(Dense(25))
model.add(Dense(1))
#compile model
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
#train model
model.fit(X_train, y_train, batch_size = 1, epochs = 1)
#predict model
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test)
rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
rmse
#Plot the data
train = df.filter(['Close'])[:df_train_len]
valid = df.filter(['Close'])[df_train_len:]
valid['Predictions'] = predictions

with plt.style.context('bmh'):
    plt.figure(figsize = (20,10))
    plt.title('Model of Yahoo stock Price', fontsize = 22)

    plt.xlabel('Date', fontsize = 22)
    plt.ylabel('Close Price $', fontsize = 22)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc = 'upper right')
plt.show()
X_test_lastday = []
X_test_lastday.append(scaler.transform(df.filter(['Close'])[-60:].values))
X_test_lastday = np.array(X_test_lastday)
X_test_lastday = np.reshape(X_test_lastday, (X_test_lastday.shape[0], X_test_lastday.shape[1], 1))
pred_price = model.predict(X_test_lastday)# using previous model
print('Our predict 09/01/2020 stock price is: {}$'.format(scaler.inverse_transform(pred_price)[0][0]))
df_today = web.DataReader('AAPL', data_source = 'yahoo', start = '2020-09-01', end = '2020-09-01')
print('09/01/2020 stock price is: {}$'.format(df_today['Close'][0]))
from sklearn.neighbors import KNeighborsRegressor
#by original dataset
df_scaled = pd.DataFrame(df_target_scaled, columns = ['Close Price'])
df_scaled['Prediction'] = df_scaled[['Close Price']].shift(-60)
df_scaled
X = np.array(df_scaled.drop(['Prediction'],1))[: -60]
y = np.array(df_scaled.drop(['Close Price'],1))[: -60]
X_last_60days = np.array(df_scaled.drop(['Prediction'],1))[-60:]
X_train, y_train, X_test, y_test = X[:df_train_len], y[:df_train_len], X[df_train_len:], y[df_train_len:]
model = KNeighborsRegressor(n_neighbors = 60)
#if we need to find the best parameter, we can use GridSearchCV
from sklearn.model_selection import GridSearchCV
parameters = {'n_neighbors': list(range(11)) + [20,40,60],
             'weights': ['uniform', 'distance'],
             'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
grid = GridSearchCV(model, parameters)
grid.fit(X_train, y_train)
predictions = grid.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test)
rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
rmse
train = df.filter(['Close'])[:df_train_len]
valid = df.filter(['Close'])[df_train_len:-60]
valid['Predictions'] = predictions
with plt.style.context('bmh'):
    plt.figure(figsize = (20,10))
    plt.title('Model of Yahoo stock Price KNN', fontsize = 22)

    plt.xlabel('Date', fontsize = 22)
    plt.ylabel('Close Price $', fontsize = 22)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Prediction'], loc = 'upper right')
plt.show()