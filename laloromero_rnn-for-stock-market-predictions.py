import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

file = '../input/CryptocoinsHistoricalPrices.csv'

df = pd.read_csv(file,
                      names = ['No','Date','Open','High','Low','Close','Volume','Market_Cap','Coin','Open_Cero','Close_Cero','Delta'],
                      index_col = ['Date'], 
                      parse_dates = ['Date'],
                      usecols = [0, 1, 2, 3, 4, 5, 8, 11],
                      na_filter = True)
df.head()
# Looking at first row, it has the names of columns but we already have them, so we'll remove it.
df = df[1:]
# Some values are still non numeric, we can see it by looking at their type
print('Open column type', type(df.Open[0]))
print('Close column type', type(df.Close[0]))
# But, why they are still str and not numeric? Let's try to convert it into numeric as it
try:
    df.Open = pd.to_numeric(df.Open)
except Exception as e:
    print(e)
# Check for Nan    
df = df.reset_index()
print(df.isnull().sum())
# First, we remove the index to manage date column as well
df = df.reset_index()
df.dropna(inplace=True)
print(df.isnull().sum())
# Once we clean de data set, let's try again to convert Open and Close columns to numeric
print("Open parsing errors",pd.to_numeric(df.Open, errors='occurrs').isnull().sum())
print("Close parsing errors",pd.to_numeric(df.Close, errors='occurrs').isnull().sum())
# Converting the data, we found there are no more error while parsing. So, move on!
df.Open = pd.to_numeric(df.Open)
df.Close = pd.to_numeric(df.Close)
from pandas import Timestamp
from sklearn.preprocessing import MinMaxScaler

# Get all rows of BTC
df_btc = df[df.Coin == 'BTC']
# Sort by date (just in case)
df_btc.sort_values(by=['Date'])
# Create dataframe only with dates, this will be used to create the transformation.
date_range = pd.date_range(start = '2013-01-01', end = '2018-12-31')
df_date = pd.DataFrame({'date': date_range, 'random': np.random.randint(1, high=100, size=len(date_range))})
# Change date to timestamp to be able to perform transformation
date2scal = df_date.date.apply(pd.to_datetime).apply(Timestamp.timestamp)

sc = MinMaxScaler(feature_range = (0,1))
sc.fit(date2scal.values.reshape(-1,1))
# Create two copies, both are goingo to be used to train, but one will be for the input and, the other one, for the output.
df_x = df_btc.copy()
df_y = df_btc.copy()
df_x.drop(df_x.head(1).index,inplace=True)
df_y.drop(df_y.tail(1).index,inplace=True)
# The data frame used for input needs to be transformed.
df_date = df_x.Date.apply(pd.to_datetime).apply(Timestamp.timestamp)
date_scaled = sc.transform(df_date.values.reshape(-1,1))
# Plot transformed information
import matplotlib.pyplot as plt
open_log = np.log(df_x.Open.values) 
y_log = np.log(df_y.Open.values)
plt.plot(open_log)
# Create an array of zeros to store all data transformed.
struct = np.zeros((open_log.shape[0], 2))
struct[:,0] = date_scaled[:,0]
struct[:,1] = open_log
df_struct = pd.DataFrame({'data': struct[:,1], 'date': df_x.Date})
# Create the array for the training with samples of 60 (this is taken as it from Derrick's writting)
X_train = np.array([struct[i-60:i] for i in range(60, open_log.shape[0])])
Y_train = np.array([y_log[i] for i in range(60, open_log.shape[0])])
# Import keras modules to build model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU
from keras.layers import Dropout
from keras import initializers, optimizers
import tensorflow as tf


sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
normal = initializers.glorot_normal()
bias_init = initializers.Constant(value=1)
    
regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 2),
                   bias_initializer=bias_init, unit_forget_bias=True))
regressor.add(Dropout(0.2))
    
regressor.add(LSTM(units = 50, return_sequences = True, recurrent_initializer = normal,
                   bias_initializer=bias_init, unit_forget_bias=True))
regressor.add(Dropout(0.2))
    
regressor.add(LSTM(units = 50, return_sequences = True, recurrent_initializer = normal,
                   bias_initializer=bias_init, unit_forget_bias=True))
regressor.add(Dropout(0.2))
    
regressor.add(LSTM(units = 50, recurrent_initializer = normal,
                   bias_initializer=bias_init, unit_forget_bias=True))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = sgd, loss = 'mean_squared_error')

regressor.fit(X_train, Y_train, epochs = 100, batch_size = 32)
# Let's test the model using the training data set
val_sp = regressor.predict(X_train)
# The resuts looks much the same. Now it's time predict some particular period
plt.plot(np.exp(Y_train),label='original')
plt.plot(np.exp(val_sp),label='prediction')
plt.legend(loc='best')
# We take the last 60 samples of original data
df_test = df[0:60]
# Again, we just want BTC (this part isn't necessary because looking at original dataframe, the first 60 values are from BTC)
df_test = df_test[df_test.Coin == 'BTC']
# Convert the open values to numeric
df_test.Open = pd.to_numeric(df_test.Open)
# Convert Date to timestamp for convertion using the scalar function from above
conv_date = df_test.Date.apply(pd.to_datetime).apply(Timestamp.timestamp)
date_test_scaled = sc.transform(conv_date.values.reshape(-1,1))
test_log = np.log(df_test.Open.values)
# Combine all convertions into one array
X_test = np.zeros((test_log.shape[0],2), dtype="float64")
X_test[:,0] = date_test_scaled[:,0]
X_test[:,1] = test_log
from keras.preprocessing.sequence import pad_sequences
# Keras has a functionality to create a pad sequence based on a list, so we convert all into list. Then pass it to function.
l = X_test.shape[0]
x_test = [X_test[v:v+l].tolist() for v in range(0, l)]
x_test = pad_sequences(x_test, dtype='float64')
# Predict the corresponding output
stock_pred = regressor.predict(x_test)
plt.plot(np.exp(stock_pred))
plt.plot(df_test.Open.values)
from statsmodels.tsa.stattools import adfuller
def test_stationarity(tieseries):

    rollmean = tieseries.rolling(window=12).mean()
    rollstd = tieseries.rolling(window=12).std()

    plt.plot(tieseries, color="blue", label="Original")
    plt.plot(rollmean, color="red", label="Rolling Mean")
    plt.plot(rollstd, color="black", label="Rolling Std")    
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(tieseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput,'%f' % (1/10**8))
#df_x = df_btc.copy()
#df_y = df_btc.copy()
df_struct = df_x
df_struct.Open = np.log(df_x.Open)
df_struct.Date = df_x.Date.apply(pd.to_datetime)
ts = df_struct.set_index(['Date'])
ts = ts['Open']
test_stationarity(ts)
# Moving average
mov_average = ts.rolling(12).mean()
# Plot it with the serie
plt.plot(mov_average)
plt.plot(ts)
# Substract the trend
ts_log_mov_av_diff = ts - mov_average
# Remove Nan values
ts_log_mov_av_diff.dropna(inplace=True)
# Test again stationarity
test_stationarity(ts_log_mov_av_diff)
new_df = ts.reset_index()
new_date = new_df.Date.apply(pd.to_datetime).apply(Timestamp.timestamp)
new_date_scaled = sc.transform(new_date.values.reshape(-1,1))
new_open_log = np.log(new_df.Open.values) 
X_test2 = np.zeros((new_open_log.shape[0],2), dtype="float64")
X_test2[:,0] = new_date_scaled[:,0]
X_test2[:,1] = new_open_log
y_log2 = np.log(new_df.Open.values)
X_train2 = np.array([X_test2[i-60:i] for i in range(60, open_log.shape[0])])
Y_train2 = np.array([y_log2[i] for i in range(60, open_log.shape[0])])

bias_init2 = initializers.Constant(value=-1)
    
regressor2 = Sequential()

regressor2.add(GRU(units = 50, return_sequences = True, input_shape = (X_train2.shape[1], 2),
                   bias_initializer=bias_init2))
regressor2.add(Dropout(0.2))
    
regressor2.add(GRU(units = 50, return_sequences = True, recurrent_initializer = normal,
                   bias_initializer=bias_init2))
regressor2.add(Dropout(0.2))
    
regressor2.add(GRU(units = 50, return_sequences = True, recurrent_initializer = normal,
                   bias_initializer=bias_init2))
regressor2.add(Dropout(0.2))
    
regressor2.add(GRU(units = 50, recurrent_initializer = normal,
                   bias_initializer=bias_init2))
regressor2.add(Dropout(0.2))

regressor2.add(Dense(units = 1))

regressor2.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor2.fit(X_train2, Y_train2, epochs = 100, batch_size = 32)
# Test the result with the training
new_pred = regressor2.predict(X_train2)
plt.plot(new_pred, label='Prediction')
plt.plot(Y_train2, label='Real value')
plt.legend(loc='best')
l = X_test2[:60].shape[0]
x_test2 = [X_test2[v:v+l].tolist() for v in range(0, l)]
x_test2 = pad_sequences(x_test2, dtype='float64')
stock_pred = regressor2.predict(x_test2)
plt.plot(stock_pred, label='pred')
plt.plot(X_test2[:60,1], label='real')
plt.legend(loc='best')