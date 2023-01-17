# Basic
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import datetime, pytz
import math

# Visualization
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go

# Model
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from keras.optimizers import SGD
from sklearn.metrics import mean_squared_error

# Defaults
init_notebook_mode(connected=True)
%matplotlib inline
plt.style.use('fivethirtyeight')

# Dataset
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Create a function to convert a timestamp in the file to date
def dateparse (time_in_secs):    
    return pytz.utc.localize(datetime.datetime.fromtimestamp(float(time_in_secs)))
df = pd.read_csv('/kaggle/input/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv', parse_dates=[0], date_parser=dateparse, index_col='Timestamp')
df.shape
df.head()
# Visualize
df['Close']['2017':].plot(figsize=(16,4))
# Fix no trade values on a minute scale
df['Volume_(BTC)'].fillna(value=0, inplace=True)
df['Volume_(Currency)'].fillna(value=0, inplace=True)
df['Weighted_Price'].fillna(value=0, inplace=True)

# Forward fill those price values
df['Open'].fillna(method='ffill', inplace=True)
df['High'].fillna(method='ffill', inplace=True)
df['Low'].fillna(method='ffill', inplace=True)
df['Close'].fillna(method='ffill', inplace=True)
# Check
df.isnull().sum().sum()
# Create daily df[Date, Close]
df['Date'] = df.index.date
df_d = df.groupby('Date')['Close'].mean()
df_d = pd.DataFrame(df_d)
df_d.shape
# Split data
split = len(df_d) - int(len(df_d) * 0.8)
df_train = df_d.iloc[split:]
df_test = df_d.iloc[:split]
'''
# Split data
df_d.index = pd.to_datetime(df_d.index)
df_d_train = df_d['2017':'2020-04'].iloc[:, 0:1]
df_d_test = df_d['2020-04':].iloc[:, 0:1]
'''
step = 21
def prepeare_data(df, step):
    data = []
    
    for i in range(len(df) - step):
        data.append((df[i: (i + step)]).values)
        
    return np.array(data)
# Get model data
X_train = prepeare_data(df_train, step)
X_test = prepeare_data(df_test, step)

print("X_train shape= ", X_train.shape)
print("X_test shape= ", X_test.shape)
# Get targets
y_train = df_train.Close[step:].values
y_test = df_test.Close[step:].values

print("y_train shape= ", y_train.shape)
print("y_test shape= ", y_test.shape)
# Model
model = Sequential()
model.add(LSTM(units=128, activation='relu', dropout=0.2, input_shape=(21,1))) # Input layer
model.add(Dense(units=1)) # Output layer
model.compile(optimizer='adam', loss='mae') #rmse?
model.fit(X_train, y_train, epochs=30, batch_size=7)
# Predictions
preds = model.predict(X_test)
preds.shape
from sklearn.metrics import mean_absolute_error

mean_absolute_error(preds, y_test)
def return_rmse(test,predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    print("The root mean squared error is {}.".format(rmse))
return_rmse(y_test, preds)
# Some functions to help out with
def plot_predictions(test,predicted):
    fig, ax = plt.subplots(1, figsize=(16, 9))
    plt.plot(test, color='red',label='Real Price')
    plt.plot(predicted, color='blue',label='Predicted Price')
    plt.title('Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
# Visualizing the results for LSTM
plot_predictions(y_test,preds)
# actual correlation
targets = df_test['Close'][step:]
preds = model.predict(X_test).squeeze()

corr = np.corrcoef(targets, preds)[0][1]
print('R={:.2f}'.format(corr))
# plot
from matplotlib import pyplot

pyplot.scatter(y_test, preds)
pyplot.title('r = {:.2f}'.format(corr), fontsize=18)
pyplot.show()