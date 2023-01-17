import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from pylab import rcParams
from keras.models import Sequential
from keras.layers import Dense
train_data = pd.read_csv("../input/demand-forecasting-kernels-only/train.csv", parse_dates=True, index_col =0)
#y_train = train_data.iloc[:,-1].values
x_train=train_data.copy()
y_train=x_train[['sales']]
x_train['y'] = x_train.index.year-train_data.index.year.min()
x_train['m'] = x_train.index.month
x_train['d'] = x_train.index.day
x_train['dow'] = x_train.index.dayofweek
x_train= x_train.drop('sales',axis=1)
x_train.head()
#y_train.head()
train_data_null=train_data.isna().any(axis=0)
print (train_data_null)
n_cols=x_train.shape[1]
print(n_cols)
def regression_model():
    # create model
    model = Sequential()
    model.add(Dense(20, activation='relu', input_shape=(n_cols,)))
    #model.add(Dense(10, activation='relu'))
    #model.add(Dense(3, activation='relu'))
    model.add(Dense(1))
    
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])
    return model
model = regression_model()
model.fit(x_train, y_train, validation_split=0.1, epochs=30, verbose=2)
test_data_y_2017=x_train[(x_train.index.year==2017)]
test_data_y_2017.head()
pred_sales_y_2017=model.predict(test_data_y_2017)


pred_2017_df=pd.DataFrame(pred_sales_y_2017)

pred_2017_df.head()
Results = test_data_y_2017
Results.describe()
sales_2017=y_train[(y_train.index.year==2017)]
#sales_2017_df=pd.DataFrame(sales_2017)
Results_1 = pd.concat([Results,sales_2017], join='inner', axis=1)

Results_1.head(100)
Results_1=Results_1.reset_index()
Results_2=pd.merge(Results_1,pred_2017_df, how = 'left',left_index = True, right_index = True)

Results_2.head(100)
print (train_data[(train_data.store==1) & (train_data.item==1) & (train_data.index.month==4) & (train_data.index.day==9)])

df_test
!pip install pmdarima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf 
from statsmodels.tsa.seasonal import seasonal_decompose 
from pmdarima import auto_arima                        
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse
import warnings

x_train_1_1 = train_data[(train_data.store==1) & (train_data.item==1) & (train_data.index.year!=2017)]
x_train_1_1_w=x_train_1_1.resample('W').sum()
x_train_1_1_w.head()
from statsmodels.tsa.stattools import adfuller
from numpy import log
result = adfuller(x_train_1_1_w.sales.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
ax = x_train_1_1_w['sales'].plot(figsize = (16,5), title = "Daily Sales")
ax.set(xlabel='Dates', ylabel='Sales');
x_test_1_1 = train_data[(train_data.store==1) & (train_data.item==1) & (train_data.index.year==2017)]
x_test_1_1_w=x_test_1_1.resample('W').sum()
x_test_1_1_w.head()
#auto_arima(x_train_1_1_w['sales'], seasonal=True, m=52,max_p=5, max_d=7,max_q=4, max_P=4, max_D=4,max_Q=4).summary()
arima_model = auto_arima(x_train_1_1_w['sales'], start_p=1, start_q=1,
                         test='adf',
                         max_p=3, max_q=3, m=52,
                         start_P=0, seasonal=True,
                         d=None, D=1, trace=True,
                         error_action='ignore',  
                         suppress_warnings=True, 
                         stepwise=True)


arima_model.summary()
arima_pred,confint = arima_model.predict(53, return_conf_int=True)
arima_pred
index_of_fc = pd.date_range(x_train_1_1_w.index[-1], periods = 53, freq='W')
fitted_arima_pred = pd.Series(arima_pred, index=index_of_fc)
arima_rmse_error = rmse(x_test_1_1_w['sales'], fitted_arima_pred)
arima_mse_error = arima_rmse_error**2
mean_value = x_train_1_1_w['sales'].mean()
print(f'MSE Error: {arima_mse_error}\nRMSE Error: {arima_rmse_error}\nMean: {mean_value}')
x_train_1_1_w['sales'].plot(figsize = (16,5), legend=True)
fitted_arima_pred.plot(legend = True);
x_test_1_1_w['sales'].plot(figsize = (16,5), legend=True)
fitted_arima_pred.plot(legend = True);
sarimax_model = SARIMAX(x_train_1_1_w['sales'], order = (1,0,0), seasonal_order = (0, 1, [2],52))
sarimax_result = sarimax_model.fit()
sarimax_result.summary()
sarimax_pred = sarimax_result.predict(start= x_test_1_1_w.index.min(),end = x_test_1_1_w.index.max(), typ="levels").rename("ARIMA Predictions")
sarimax_pred
x_train_1_1_w['sales'].plot(figsize = (16,5), legend=True)
sarimax_pred.plot(legend = True);
x_test_1_1_w['sales'].plot(figsize = (16,5), legend=True)
sarimax_pred.plot(legend = True);
arima_rmse_error = rmse(x_test_1_1_w['sales'], sarimax_pred)
arima_mse_error = arima_rmse_error**2
mean_value = x_train_1_1_w['sales'].mean()

print(f'MSE Error: {arima_mse_error}\nRMSE Error: {arima_rmse_error}\nMean: {mean_value}')
