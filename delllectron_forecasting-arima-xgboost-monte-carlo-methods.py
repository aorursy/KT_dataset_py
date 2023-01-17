#VIZ AND DATA MANIPULATION LIBRARY
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
sns.set_style('whitegrid')

from plotly import tools
import plotly_express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


#Preprocessing
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import uniform

#MODELS
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import xgboost as xgb
from xgboost import XGBRegressor
from bayes_opt import BayesianOptimization


#CLASSICAL STATS
import scipy
import statsmodels
from scipy.stats import boxcox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from fbprophet import Prophet
from statsmodels.tsa.statespace import sarimax
from statsmodels.tsa.seasonal import seasonal_decompose


#DEEP LEARNING LIB
from keras.models import Model,Sequential
from keras.utils import np_utils, to_categorical
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from tensorflow.keras.utils import plot_model
import itertools
import lightgbm as lgb



#METRICS
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report, r2_score,mean_absolute_error,mean_squared_error

from random import randrange
import warnings 
warnings.filterwarnings('ignore')
df = pd.read_csv('../input/foreign-exchange-rates-per-dollar-20002019/Foreign_Exchange_Rates.csv')
print('DATASET SHAPE: ', df.shape)
df.head()
#show columns
df.columns = [col.lower() for col in df.columns]
df['time serie'] = pd.to_datetime(df['time serie'])
df.columns
#get the date and rates of singapore dollar
data = df[['time serie', 'singapore - singapore dollar/us$']]
data.columns = ['date', 'rate']
#show new dataframe
data.head()
#show feature data types
data.info()
#remove rates with a value of ND
data = data.drop(data[data['rate']=='ND'].index)
#converte the rates to numeric value
data['rate'] = pd.to_numeric(data.rate)
#sort values by date
data = data.sort_values('date', ascending=True)
#show basic stats
data.rate.describe()
plt.figure(figsize=(10,5))
sns.distplot(data.rate, bins=10, color='steelblue');
fig = go.Figure()

fig.add_trace(go.Scatter(x=data.date, y=data.rate, marker_color='lightgreen'))

fig.update_layout(title='TIME-SERIES PLOT OF SINGAPORE DOLLAR RATE', 
                  height=450, width=1000, template='plotly_dark', font_color='lightgreen', 
                  font=dict(family="sans serif",
                            size=16,
                            color="grey"
                            ))

fig.update_xaxes(title='Date')
fig.update_yaxes(title='Rate / $')
fig.show()
fig, ax = plt.subplots(1,2,figsize=(14,4))
plot_acf(data.rate, lags=20, ax=ax[0]);
plot_pacf(data.rate, lags=20, ax=ax[1]);
sdec = seasonal_decompose(data.rate, model='multiplicative', freq=1)
sdec.plot();

X_train, X_val = data[:-30], data[-30:]

print('X_train Shape: ', X_train.shape)
print('X_val Shape: ', X_val.shape)
predictions = []

arima = sm.tsa.statespace.SARIMAX(X_train.rate,order=(1,1,1),seasonal_order=(1,1,1,6),
                                  enforce_stationarity=False, enforce_invertibility=False,).fit()
#get a 30 days prediction
predictions.append(arima.forecast(30))
#converting and reshaping 
predictions = np.array(predictions).reshape((30,))
arima.summary()
res = arima.resid
fig,ax = plt.subplots(1,2,figsize=(14,5))
plt.suptitle('ACF AND PACF PLOT OF RESIDUALS', fontsize=22, x=0.5, y=1.04)
fig = sm.graphics.tsa.plot_acf(res, lags=50, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(res, lags=50, ax=ax[1])
plt.show()
y_val = data.rate[-30:]
plt.figure(figsize=(14,5))
plt.plot(np.arange(len(y_val)), y_val, color='steelblue');
plt.plot(np.arange(len(y_val)), predictions, color='salmon');
plt.legend(['True Value', 'Prediction']);


arima_mae = mean_absolute_error(y_val, predictions)
arima_mse = mean_squared_error(y_val, predictions)
arima_rmse = np.sqrt(mean_squared_error(y_val, predictions))

print('Mean Absolute Error:   ', arima_mae)
print('Mean Squared Error:   ', arima_mse)
print('Root Mean Squared Error:   ', arima_rmse)
arima_error_rate = abs(((y_val - predictions) / y_val).mean()) * 100
print('MAPE:', round(arima_error_rate,2), '%')
print('R2-SCORE: ', r2_score(y_val, predictions))
#extract the date feature
data['day'] = data.date.dt.day
data['dayofweek'] = data.date.dt.dayofweek
data['dayofyear'] = data.date.dt.dayofyear
data['week'] = data.date.dt.week
data['month'] = data.date.dt.month
data['year'] = data.date.dt.year
#add lag feature
for i in range(1,8):
    data['lag'+str(i)] = data.rate.shift(i).fillna(0)
#drop the date feature
data.drop('date', axis=1, inplace=True)
#show new data frame
data.head(7)
X = data.drop('rate', axis=1)
y = data.rate

X_train, X_test = X[:-30], X[-30:]
y_train, y_test = y[:-30], y[-30:]

print('X_train: ', X_train.shape)
print('y_train: ', y_train.shape)
print('X_test: ', X_test.shape)
print('y_test: ', y_test.shape)
#convert data to xgb matrix form
dtrain = xgb.DMatrix(X_train,label=y_train)
dtest = xgb.DMatrix(X_test)
#bayesian hyper parameter tuning
#define the params
def xgb_evaluate(max_depth, gamma, colsample_bytree):
    params = {'eval_metric': 'rmse',
              'max_depth': int(max_depth),
              'subsample': 0.8,
              'eta': 0.1,
              'gamma': gamma,
              'colsample_bytree': colsample_bytree}
    
    cv_result = xgb.cv(params, dtrain, num_boost_round=250, nfold=3)    
    return -1.0 * cv_result['test-rmse-mean'].iloc[-1]
#run optimizer
xgb_bo = BayesianOptimization(xgb_evaluate, {'max_depth': (3, 7), 
                                             'gamma': (0, 1),
                                             'colsample_bytree': (0.3, 0.9)})
#define iter points
xgb_bo.maximize(init_points=10, n_iter=15, acq='ei')

#get the best parameters
params = xgb_bo.max['params']
params['max_depth'] = int(round(params['max_depth']))
#train the data
model = xgb.train(params, dtrain, num_boost_round=200)
#predict the test data 
predictions = model.predict(dtest)
y_val = data.rate[-30:]
plt.figure(figsize=(14,5))
sns.set_style('darkgrid')
plt.plot(np.arange(len(y_val)), y_val, color='salmon');
plt.plot(np.arange(len(y_val)), predictions, color='lightgreen');
plt.legend(['True Value', 'Prediction']);

xgb_mae = mean_absolute_error(y_val, predictions)
xgb_mse = mean_squared_error(y_val, predictions)
xgb_rmse = np.sqrt(mean_squared_error(y_val, predictions))

print('Mean Absolute Error:   ', xgb_mae)
print('Mean Squared Error:   ', xgb_mse)
print('Root Mean Squared Error:   ', xgb_rmse)
xgb_error_rate = abs(((y_val - predictions) / y_val).mean()) * 100
print('MAPE:', round(xgb_error_rate,2), '%')
print('R2-SCORE: ', r2_score(y_val, predictions))

#function that can generate a monte carlo simulation    
def monte_carlo_simulation(data,t_intervals ,iteration , figsize = (10,4), lw=1):
    from scipy.stats import norm

    #log returns of data
    log_returns = np.log(1 + data.pct_change())

    #Setting up the drift and random component
    mean_  = log_returns.mean()
    var = log_returns.var()
    stdev = log_returns.std()
    drift = mean_ - (0.5 *var)

    daily_returns = np.exp(drift + stdev * norm.ppf(np.random.rand(t_intervals, iteration)))

    S0 = data.iloc[-1]
    #Empty daily returns
    price_list = np.zeros_like(daily_returns)
    price_list[0] = S0

    #appliying montecarlo simulation
    for i in range(1 , t_intervals):
        price_list[i] = price_list[i-1] * daily_returns[i]
    fig_title = str(t_intervals)+ ' DAYS SIMULATION WITH ' +str(iteration)+' DIFFERENT POSSIBILITIES'
    #Show the result of 30 days simulation
    plt.figure(figsize=figsize)
    plt.plot(price_list, lw=lw)
    plt.title(fig_title)
    plt.xlabel('Interval', fontsize=16)
    plt.ylabel('Value', fontsize=16)
#fit the X_train and show the figure
monte_carlo_simulation(y_train,30,20, figsize=(13,6))