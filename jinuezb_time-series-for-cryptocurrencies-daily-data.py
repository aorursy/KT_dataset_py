# ► First libs

import matplotlib.pyplot as plt
import plotly.express as px 
import pandas as pd
import numpy as np
import re

# ► More libs for the extraction function

import datetime 
import csv
# ► For PyCaret model (Regression models)

!pip install pycaret==2.0
#from pycaret.regression import *  # we will rin this line in next sections
# ► For prophet model (Time series)

import fbprophet
import matplotlib.pyplot as plt
# ► For arima model (Time series)

!pip install pmdarima
from pmdarima.arima import auto_arima
from scipy import stats
from itertools import product
import warnings
import statsmodels.api as sm
# ► Libraries for xgboost

import xgboost as xgb
from xgboost import plot_importance, plot_tree
# ► libraries for LSTM and GRU RNN's

# ► To flat a list of lists
from pandas.core.common import flatten
# ► Scale Data
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler                
# ► Indicator
from sklearn import linear_model 
from sklearn.metrics import mean_absolute_error
# ► Model Keras modules
from keras.models import Sequential
from keras.layers import Activation, Dense, GRU, LSTM, Dropout, Bidirectional
from keras.optimizers import SGD
# ► Scraping main table wit pandas

url = 'https://coinmarketcap.com/coins/views/all/'
df = pd.read_html(url)[2]
df.head()
# ► Getting rid of '$,' simbols

df.Price = df['Price'].apply(lambda x: float(re.sub('([$,]*)','',x)))
df.sort_values(by=['Price'], inplace=True, ascending=False)
# ► Below Bitcoin is omitted because its price is too high to appreciate the rest of the coins

fig = px.bar(df[1:], y='Price', x='Name', text='Name')
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout(template='seaborn', title='Coins available for scraping')
fig.show()
coins_available = list(df.Name)
# ► Repacing ' ' and '.' for '-' and deleting '[]#'  

coins_available = [re.sub('([\s+.])','-',coin.lower()).replace('[','').replace(']','').replace('#','') for coin in coins_available]
print(coins_available)
# ► Function to extract data

def get_data(coins_available):
 
  # ► User types a number

    print(coins_available)
    coin_name = input('Type a valid coin name: ').lower()

    while True:

        if coin_name not in coins_available:
            print('Coin name should not contain spaces instead use "-" and type all in lowercase')
            coin_name = input('Type again a valid coin name: ').lower()
        else:
            break
  
  # ► Scrapping the coin name

    today = datetime.date.today()  # todays date
    mktcap_page = 'https://coinmarketcap.com/currencies/'+ coin_name +'/historical-data/?start=20130428&amp;end='+"{:%Y%m%d}".format(today)
    full_data = pd.read_html(mktcap_page)[2]
    full_data.rename(columns={'Open*': 'Open', 'Close**': 'Close'}, inplace=True)

  # ► This function replaces ',' for nothing and then changes data type from object to float

    def obj_to_num(df,cols):
      
        def obj_to_num(row):
            if ',' in str(row):
                row = row.replace(',','')
                return float(row)
            else:
                return float(row)

        for col in cols:     
            df[col] = df[col].apply(obj_to_num) 

        return df

  # ► This function changes data column type from object to timestamp[ns] and also can changes other columns to float 

    def prepare_data(file):
  
        columns = ['Open','High','Low','Close','Volume','Market Cap']
        file['Date'] = pd.to_datetime(file['Date'])
        num_data = obj_to_num(file,columns)

        return num_data

    return prepare_data(full_data)
# ► Type the coin name you want

my_coin = get_data(coins_available)
# ► df visualization

my_coin.head()
my_coin.dtypes
# ► The graph

import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(x=my_coin['Date'], y=my_coin['Open'],
                    mode='lines',
                    name='Open'))
fig.add_trace(go.Scatter(x=my_coin['Date'], y=my_coin['High'],
                    mode='lines',
                    name='High'))
fig.add_trace(go.Scatter(x=my_coin['Date'], y=my_coin['Low'],
                    mode='lines',
                    name='Low'))
fig.add_trace(go.Scatter(x=my_coin['Date'], y=my_coin['Close'],
                    mode='lines',
                    name='Close'))

#fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout(title="Coin behavior from its beginning, price in USD")

fig.show()
# ► Saving data to csv

my_coin.to_csv('my_coin_daily_data.csv')
# ► Changing the df order 

my_coin = my_coin.copy()[::-1]
# ► Data for training till last 20 samples

my_coin_high_train = my_coin[['High','Date']][:-20]
# ► Data for testing 20 samples

my_coin_high_test = my_coin[['High','Date']][-20:]
# ► Renaming columns for Prophet syntax

my_coin_high_p = my_coin_high_train.rename(columns = {'Date':'ds','High':'y'})
# ► setting this hyperparameter to 0.8  https://facebook.github.io/prophet/docs/trend_changepoints.html#adjusting-trend-flexibility
# ► 1.1 = mse(2805.25), 0.8 = mse(2857.13), 1.2 = mse(3015.23), 0.75 = mse(3038.22), 0.9 = mse(3039.19), 0.5 = mse(3182.94)   

m = fbprophet.Prophet(changepoint_prior_scale=1.1)
m.fit(my_coin_high_p)
# ► Generating future 20 dates

future = m.make_future_dataframe(periods=20)
# ► Predicting those days

forecast_p = m.predict(future)
m.plot(forecast_p);
m.plot_components(forecast_p);
# ► predictions and real price

my_coin_inverse = my_coin.copy() 
my_coin_inverse.High = my_coin_inverse.High.values[::-1]                        # To change value orders from the original data
my_coin_forecast = pd.concat([forecast_p['ds'],my_coin_inverse['High'],forecast_p['yhat']], axis=1,keys=['date','high','prophet_high'])
# ► The model didn't see this high values

my_coin_forecast.tail()
# ► Plot las 20 predictions and samples

ax = my_coin_forecast.set_index('date')[-20:].plot(figsize=(15, 5))
ax.set_ylabel('High')
ax.set_xlabel('Date')

plt.show()
# ► Create a copy with values in reverse

my_coin_high_a = (my_coin[['Date','High']][:-20].copy()).set_index('Date')
my_coin_high_a.tail()
# Arima tunning hyperparameters
# Initial approximation of parameters

Qs = range(0, 2)
qs = range(0, 3)
Ps = range(0, 3)
ps = range(0, 3)
D=1
d=1
parameters = product(ps, qs, Ps, Qs)
parameters_list = list(parameters)
len(parameters_list)

# Model Selection
results = []
best_aic = float("inf")
warnings.filterwarnings('ignore')
for param in parameters_list:
    try:
        model=sm.tsa.statespace.SARIMAX(my_coin_high_a.High, order=(param[0], d, param[1]), 
                                        seasonal_order=(param[2], D, param[3], 12),enforce_stationarity=False,
                                            enforce_invertibility=False).fit(disp=-1)
    except ValueError:
        #print('wrong parameters:', param)
        continue
    aic = model.aic
    if aic < best_aic:
        best_model = model
        best_aic = aic
        best_param = param
    results.append([param, model.aic])
# ► Forecasting and renaming forecasting column

forecast_arima = pd.DataFrame(best_model.predict(start=0, end=len(my_coin_forecast)-1))
forecast_arima = forecast_arima.rename(columns = {0:'arima_high'})
# ► Joining data frames

my_coin_forecast_cp = pd.concat([my_coin_forecast.set_index('date'),forecast_arima], axis = 1, sort = False)
# ► Comparing 20 unseen data by both models

my_coin_forecast_cp.tail()
plt.figure(figsize=(21,10))
my_coin_forecast_cp.high[-20:].plot()
my_coin_forecast_cp.arima_high[-20:].plot(color='r', ls='--', label='ARIMA Predicted High_Price')
plt.legend()
plt.title('ETH Prices (USD) Predicted vs Actuals, by months')
plt.ylabel('High USD')
plt.show()
# ► Function to bring more dates info to the xgboost model

def create_features(df, label=None):
    """
    Creates time series features from datetime index
    """
    df['date'] = df.Date
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear
    
    X = df[['hour','dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekofyear']]
    if label:
        y = df[label]
        return X, y
    return X
# ► Creating data for training and testing 

X_train, y_train = create_features(my_coin_high_train, label='High')
X_test, y_test = create_features(my_coin_high_test, label='High')
# ► XGBoost model and hyperparameters (I recomend to change it for each coin)

model =  xgb.XGBRegressor(objective ='reg:squarederror',min_child_weight=10, booster='gbtree', colsample_bytree = 0.8, learning_rate = 0.4,
                max_depth = 4, alpha = 10, n_estimators = 80)
model.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds=50,
       verbose=False)
# ► Forecasting values

xgb_preds = model.predict(X_test)
# ► Data frame with dates as index

xgb_preds_df = (pd.DataFrame({'Date':my_coin_high_test.Date.values, 'xgb_forec':xgb_preds})).set_index('Date')
# ► Joining to the rest of predictions 

my_coin_forecast_cp = pd.concat([my_coin_forecast_cp,xgb_preds_df], axis = 1, sort = False)
plt.figure(figsize=(21,10))
my_coin_forecast_cp.high[-20:].plot()
my_coin_forecast_cp.xgb_forec[-20:].plot(color='r', ls='--', label='XGBoost Predicted High_Price')
plt.legend()
plt.title('ETH Prices (USD) Predicted vs Actuals, by months')
plt.ylabel('High USD')
plt.show()
# ► To use regression models

from pycaret.regression import *
# ► Taking same data to train the XGBoost but in the same df

my_coin_high_py = pd.concat([X_train, y_train], axis = 1, sort = False)
# ► Pycaret syntax, just press Enter

exp_reg = setup(data = my_coin_high_py, target = 'High')
best = compare_models()
# ► Creating an ensemble meta-estimator that fits a base regressor on the whole dataset

blender_top3 = blend_models(compare_models(n_select = 3))
# ► It does not plot Hyperparams cause is blended model  

evaluate_model(blender_top3)
# ► Stage of predictions taking only dates

lr_pred_new = predict_model(blender_top3, data = X_test)
my_coin_forecast_cp['pycaret_high'] = lr_pred_new['Label']
my_coin_forecast_cp['pycaret_high'][-20:] = lr_pred_new['Label']
plt.figure(figsize=(21,10))
my_coin_forecast_cp.high[-20:].plot()
my_coin_forecast_cp.pycaret_high[-20:].plot(color='r', ls='--', label='Pycaret Predicted High_Price')
plt.legend()
plt.title('ETH Prices (USD) Predicted vs Actuals, by months')
plt.ylabel('High USD')
plt.show()
# ► For more info: https://github.com/llSourcell/How-to-Predict-Stock-Prices-Easily-Demo/blob/master/lstm.py

# ► function inversed (to predict more data)

def predict_sequence_full(model, data, window_size, prediction_len):
    #Predict sequence of <prediction_len> steps before shifting prediction run forward by <prediction_len> steps
    prediction_seqs = []
    for i in range(int(len(data)/prediction_len)):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[np.newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    #return prediction_seqs
    return [scaler.inverse_transform(np.array(i).reshape(1, -1))[0].tolist() for i in prediction_seqs]
# ► Function to plot data windows

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()

print ('Support functions defined')
# ► Loading Data and changing the order

df = pd.read_csv('my_coin_daily_data.csv', index_col='Date', parse_dates = ['Date'])
df = df[::-1]
df.head()
dflstm = df['High']
# ► Splitting data graph 20 samples to test

ax = dflstm.plot(figsize=(14, 7))
ax.axvline(dflstm.index[-20], linestyle="--", c="black")
# ► Scaling all data

scaler = MinMaxScaler(feature_range=(-1,1))
sc_lstm = scaler.fit_transform(dflstm.reset_index().drop(['Date'], 1))
def split_data(stock, lookback):
    
    data_raw = stock
    data = []
    
    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - lookback): 
        data.append(data_raw[index: index + lookback])
    
    data = np.array(data);
    test_set_size = 20 
    train_set_size = data.shape[0] - (test_set_size);
    
    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]
    
    x_test = data[train_set_size:,:-1]
    y_test = data[train_set_size:,-1,:]
    
    return [x_train, y_train, x_test, y_test]
lookback = 21 # choose sequence length
LSTM_train_inputs, LSTM_train_outputs, LSTM_test_inputs, LSTM_test_outputs = split_data(sc_lstm, lookback)
print('x_train.shape = ',LSTM_train_inputs.shape)
print('y_train.shape = ',LSTM_train_outputs.shape)
print('x_test.shape = ',LSTM_test_inputs.shape)
print('y_test.shape = ',LSTM_test_outputs.shape)
# ► LSTM model structure

def build_model(inputs, output_size, neurons, activ_func="linear",
                dropout=0.10, loss="mae", optimizer="adam"):
    
    model = Sequential()

    model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model
# ► initialise model architecture
nn_model = build_model(LSTM_train_inputs, output_size=1, neurons = 32)

# ► train model on data
nn_history = nn_model.fit(LSTM_train_inputs, LSTM_train_outputs, 
                            epochs=14, batch_size=1, shuffle=True)
# ► Plot "Loss"
plt.plot(nn_history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()
# ► Calling functions to predict and plot windows

predsl = predict_sequence_full(nn_model, LSTM_test_inputs, 20, 20)
LSTM_test_outputs_inversed = scaler.inverse_transform(np.array(LSTM_test_outputs).reshape(1, -1))[0]
plot_results_multiple(predsl, LSTM_test_outputs_inversed, 20)
MAE = mean_absolute_error(LSTM_test_outputs_inversed, list(flatten(predsl)))
print('The Mean Absolute Error is: {}'.format(MAE))
# ► New column to the final data frame

my_coin_forecast_cp['LSTM_Forec'] = 'NaN'
my_coin_forecast_cp['LSTM_Forec'][-20:] = predsl[0]
GRU_train_inputs = LSTM_train_inputs
GRU_train_outputs = LSTM_train_outputs
GRU_test_inputs = LSTM_test_inputs
GRU_test_outputs = LSTM_test_outputs
# The GRU architecture
regressorGRU = Sequential()
# First GRU layer with Dropout regularisation
regressorGRU.add(GRU(units=20, return_sequences=True, input_shape=(GRU_train_inputs.shape[1],1), activation='tanh'))
regressorGRU.add(Dropout(0.2))
# Second GRU layer
regressorGRU.add(GRU(units=60, return_sequences=True, input_shape=(GRU_train_inputs.shape[1],1), activation='tanh'))
regressorGRU.add(Dropout(0.2))
# Third GRU layer
regressorGRU.add(GRU(units=120, return_sequences=True, input_shape=(GRU_train_inputs.shape[1],1), activation='tanh'))
regressorGRU.add(Dropout(0.2))
# Fourth GRU layer
regressorGRU.add(GRU(units=20, activation='tanh'))
regressorGRU.add(Dropout(0.2))
# The output layer
regressorGRU.add(Dense(units=1))
# Compiling the RNN
regressorGRU.compile(optimizer=SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=False),loss='mean_squared_error')
# Fitting to the training set
regressorGRU.fit(GRU_train_inputs, GRU_train_outputs,epochs=8,batch_size=80)
plt.plot(regressorGRU.history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()
# ► Calling functions to predict and plot windows

predsg = predict_sequence_full(regressorGRU, GRU_test_inputs, 20, 20)
GRU_test_inversed = scaler.inverse_transform(np.array(GRU_test_outputs).reshape(1, -1))[0]
plot_results_multiple(predsg, GRU_test_inversed, 20)
MAE = mean_absolute_error(GRU_test_inversed, list(flatten(predsg)))
print('The Mean Absolute Error is: {}'.format(MAE))
# ► New column to the final data frame

my_coin_forecast_cp['GRU_Forec'] = 'NaN'
my_coin_forecast_cp['GRU_Forec'][-20:] = predsg[0]
# ► Renaming columns

my_coin_forecast_cp.rename(columns={'high': 'High', 'prophet_high': 'Prophet_forec', 'arima_high': 'Arima_forec', 'xgb_forec': 'Xgboost_forec', 'pycaret_high': 'Pycaret_forec', 'LSTM_Forec': 'LSTM_forec', 'GRU_Forec': 'GRU_forec'}, inplace=True)
# ► The final data frame real price ('high') with all forecastings

my_coin_forecast_cp[-20:]
import plotly.graph_objects as go
fig = go.Figure()

for i in my_coin_forecast_cp.columns:
    
    fig.add_trace(go.Scatter(x=my_coin_forecast_cp.index[-20:], y=my_coin_forecast_cp[i][-20:],
                    mode='lines',
                    name=i))
    
fig.update_layout(title="Coin behavior (Ethereum) forecastings vs actuals, price in USD")

fig.show()
# ► In the last step to compare models, not before cause there are issues with sklearn versions for pmdarima and pycaret 

from sklearn.metrics import mean_squared_error as mse

print('Last 20 days mse Prophet: ',mse(my_coin_forecast_cp.High[-20:], my_coin_forecast_cp.Prophet_forec[-20:]))
print('Last 20 days mse Arima: ',mse(my_coin_forecast_cp.High[-20:], my_coin_forecast_cp.Arima_forec[-20:]))
print('Last 20 days mse Xgboost: ',mse(my_coin_forecast_cp.High[-20:], my_coin_forecast_cp.Xgboost_forec[-20:]))
print('Last 20 days mse Pycaret: ',mse(my_coin_forecast_cp.High[-20:], my_coin_forecast_cp.Pycaret_forec[-20:]))
print('Last 20 days mse LSTM: ',mse(my_coin_forecast_cp.High[-20:], my_coin_forecast_cp.LSTM_forec[-20:]))
print('Last 20 days mse GRU: ',mse(my_coin_forecast_cp.High[-20:], my_coin_forecast_cp.GRU_forec[-20:]))