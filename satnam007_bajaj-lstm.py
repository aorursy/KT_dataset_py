import pandas as pd

import numpy as np

import pickle

import matplotlib.pyplot as plt

from scipy import stats

from keras.layers.core import Dense, Activation, Dropout

from keras.layers.recurrent import LSTM

from keras.models import Sequential, load_model, save_model

import seaborn as sns

from pylab import rcParams

from sklearn.preprocessing import MinMaxScaler

from sklearn import metrics



%matplotlib inline



sns.set(style='whitegrid', palette='muted', font_scale=1.5)



rcParams['figure.figsize'] = 14, 8



RANDOM_SEED = 42
df = pd.read_csv('../input/bajajcombined/bajaj-2003-2020.csv')
df.columns
plot_x = df['Date'].copy()

df.set_index("Date", inplace = True)

df.index = pd.to_datetime(df.index)

df.head(2)
print(type(plot_x))
dff = df[df['Symbol'] == 'BAJFINANCE']



google_stock = dff[dff['Series'] == 'EQ']

google_stock.head()
x_scaler = MinMaxScaler()

y_scaler = MinMaxScaler()

goog_df = google_stock.copy()
goog_df.drop(['Symbol', 'Series', 'No. of Trades'], axis=1, inplace=True)
goog_df.head()
x = goog_df[['Prev Close', 'Open Price', 'High Price', 'Low Price', 'Last Price',

        'Average Price', 'Total Traded Quantity', 'Turnover']].copy()



y = goog_df['Close Price'].copy()



x[['Prev Close', 'Open Price', 'High Price', 'Low Price', 'Last Price',

        'Average Price', 'Total Traded Quantity', 'Turnover']] = x_scaler.fit_transform(x)



y = y_scaler.fit_transform(y.values.reshape(-1, 1))
def load_data(X, seq_len, train_size=0.9):

    amount_of_features = X.shape[1]

    X_mat = X.values

    sequence_length = seq_len + 1

    data = []

    

    for index in range(len(X_mat) - sequence_length):

        data.append(X_mat[index: index + sequence_length])

    

    data = np.array(data)

    train_split = int(round(train_size * data.shape[0]))

    train_data = data[:train_split, :]

    

    x_train = train_data[:, :-1]

    y_train = train_data[:, -1][:,-1]

    

    x_test = data[train_split:, :-1] 

    y_test = data[train_split:, -1][:,-1]



    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))

    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))  



    return x_train, y_train, x_test, y_test
window = 22

x['close'] = y

X_train, y_train, X_test, y_test = load_data(x, window)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

print (X_train[0], y_train[0])
def build_model(input_shape):

    d = 0.2

    model = Sequential()

    

    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))

    model.add(Dropout(d))

        

    model.add(LSTM(128, input_shape=input_shape, return_sequences=False))

    model.add(Dropout(d))

        

    model.add(Dense(32,kernel_initializer="uniform",activation='relu'))        

    model.add(Dense(1,kernel_initializer="uniform",activation='linear'))

    

    model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])

    return model
goog_df.shape
model = build_model(input_shape=(window, 9))
model.fit(X_train, y_train, batch_size=32, epochs=500, verbose=0)
save_model(model, "model.h5")
model = load_model("model.h5")
trainPredict = model.predict(X_train)

testPredict = model.predict(X_test)
testPredict.shape
trainPredict.shape
trainPredict = y_scaler.inverse_transform(trainPredict)

trainY = y_scaler.inverse_transform([y_train])

testPredict = y_scaler.inverse_transform(testPredict)

testY = y_scaler.inverse_transform([y_test])
plot_predicted = testPredict.copy()

plot_predicted = plot_predicted.reshape(241, 1)

plot_actual = testY.copy()

plot_actual = plot_actual.reshape(241, 1)

print(plot_actual.shape)

print(plot_predicted.shape)
plot_x = pd.to_datetime(plot_x.iloc[-174:])
plt.plot(pd.DataFrame(plot_predicted), label='Predicted')

plt.plot(pd.DataFrame(plot_actual), label='Actual')

plt.legend(loc='best')

plt.show()
trainScore = metrics.mean_squared_error(trainY[0], trainPredict[:,0]) ** .5

print('Train Score: %.2f RMSE' % (trainScore))

testScore = metrics.mean_squared_error(testY[0], testPredict[:,0]) ** .5

print('Test Score: %.2f RMSE' % (testScore))
google_stock.columns
google_stock_prices = google_stock['Close Price'].values.astype('float32')

google_stock_prices = google_stock_prices.reshape(len(google_stock_prices), 1)
trainPredictPlot = np.empty_like(google_stock_prices)

trainPredictPlot[:, :] = np.nan

trainPredictPlot[window:len(trainPredict)+window, :] = trainPredict



testPredictPlot = np.empty_like(google_stock_prices)

testPredictPlot[:, :] = np.nan

testPredictPlot[(len(google_stock_prices) - testPredict.shape[0]):len(google_stock_prices), :] = testPredict
plt.plot(pd.DataFrame(google_stock_prices, columns=["close"], index=goog_df.index).close, label='Actual')

plt.plot(pd.DataFrame(trainPredictPlot, columns=["close"], index=goog_df.index).close, label='Training')

plt.plot(pd.DataFrame(testPredictPlot, columns=["close"], index=goog_df.index).close, label='Testing')

plt.legend(loc='best')

plt.show()