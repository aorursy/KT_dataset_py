import pandas as pd

cal = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')

stval = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')

price = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv') 
stval.info()
cal.info()
price.info()
# Taken from https://www.kaggle.com/gemartin/load-data-reduce-memory-usage

import numpy as np

def reduce_mem_usage(df):

   

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df
reduce_mem_usage(stval)
reduce_mem_usage(price)
reduce_mem_usage(cal)
sales = pd.melt(stval, id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name='d', value_name='demand').dropna()
sales = pd.merge(sales, cal, on='d', how='left')

sales = pd.merge(sales, price, on=['store_id','item_id','wm_yr_wk'], how='left') 
sales = sales.drop(['d','event_name_1','event_type_1','event_name_2','event_type_2','snap_CA','snap_TX','snap_WI','wm_yr_wk','weekday'], axis=1)
sales = sales.dropna()
sale_food_ca = sales[sales['id'] == 'FOODS_3_555_CA_1_validation']
del sales, price, cal #free some space
sale_food_ca['date'] = pd.to_datetime(sale_food_ca['date'])
import matplotlib.pyplot as plt

plt.rc('xtick', labelsize=16) 

plt.rc('ytick', labelsize=16) 

params = {'legend.fontsize': 16,'legend.handlelength': 2}

plt.rcParams.update(params)



train = sale_food_ca[sale_food_ca['date'] <= '2016-03-27']

test = sale_food_ca[(sale_food_ca['date'] > '2016-03-27') & (sale_food_ca['date'] <= '2016-04-24')]



fig, ax = plt.subplots(figsize=(25,5))

train.plot(x='date',y='demand',label='Train',ax=ax)

test.plot(x='date',y='demand',label='Test',ax=ax)
# Taken from https://gist.github.com/Deffro/e54bdb90dd1c1392fc85e1db1cfbab7d



import pandas as pd

from statsmodels.tsa.stattools import adfuller



def adf_test(series,title=''):

    """

    Pass in a time series and an optional title, returns an ADF report

    """

    print('Augmented Dickey-Fuller Test: {}'.format(title))

    result = adfuller(series.dropna(),autolag='AIC') 

    

    labels = ['ADF test statistic','p-value','# lags used','# observations']

    out = pd.Series(result[0:4],index=labels)



    for key,val in result[4].items():

        out['critical value ({})'.format(key)]=val

        

    print(out.to_string())          

    

    if result[1] <= 0.05:

        print("Strong evidence against the null hypothesis")

        print("Reject the null hypothesis")

        print("Data has no unit root and is stationary")

    else:

        print("Weak evidence against the null hypothesis")

        print("Fail to reject the null hypothesis")

        print("Data has a unit root and is non-stationary")
# Augmented Dickey-Fuller Test

adf_test(train[['date','demand']]['demand'],title='Demand')
# KPSS Test



from statsmodels.tsa.stattools import kpss

result = kpss(train[['date','demand']]['demand'].values, regression='c')

print("KPSS Statistic: {}".format(result[0]))

print("P-Value: {}".format(result[1]))

for key, value in result[3].items():

    print("Critial Values: {}, {}".format(key,value))
# Granger Causality Test



from statsmodels.tsa.stattools import grangercausalitytests

grangercausalitytests(train[['demand','sell_price']], maxlag=3)
# Lag PLots



from pandas.plotting import lag_plot

fig, ax = plt.subplots(2,3,figsize=(15,5))

lag_plot(train['demand'], lag=1, ax=ax[0][0])

lag_plot(train['demand'], lag=30, ax=ax[0][1])

lag_plot(train['demand'], lag=60, ax=ax[0][2])

lag_plot(train['sell_price'], lag=1, ax=ax[1][0])

lag_plot(train['sell_price'], lag=30, ax=ax[1][1])

lag_plot(train['sell_price'], lag=60, ax=ax[1][2])

plt.show()
import pandas as pd

predictions = pd.DataFrame()

predictions['date'] = test['date']

stats = pd.DataFrame(columns=['Model Name','Execution Time','RMSE'])
from statsmodels.graphics.tsaplots import plot_acf

fig, ax = plt.subplots(figsize=(15,3))

plot_acf(sale_food_ca['demand'].tolist(), lags=50, ax=ax)

plt.show()
from statsmodels.graphics.tsaplots import plot_pacf

fig, ax = plt.subplots(figsize=(15,3))

plot_pacf(sale_food_ca['demand'].tolist(), lags=50, ax=ax)

plt.show()
# Moving Average model with window size 7 

y = train[['date','demand']]

y = y.set_index('date')

y['MA7'] = y.rolling(window=7).mean() 

y.plot(figsize=(15,4))
# Moving Average model with expanding window size 2

y1 = train[['date','demand']]

y1 = y1.set_index('date')

y1['demand'].expanding(min_periods=2).mean().plot(figsize=(15,4))
import time

from statsmodels.tsa.holtwinters import SimpleExpSmoothing

from sklearn.metrics import mean_squared_error



t0 = time.time()

model_name='Simple Exponential Smoothing'

span = 7

alpha = 2/(span+1)

#train

simpleExpSmooth_model = SimpleExpSmoothing(train['demand']).fit(smoothing_level=alpha,optimized=False)

t1 = time.time()-t0

#predict

predictions[model_name] = simpleExpSmooth_model.forecast(28).values

fig, ax = plt.subplots(figsize=(25,5))

train[-28:].plot(x='date',y='demand',label='Train',ax=ax, marker='o', color='blue')

test.plot(x='date',y='demand',label='Test',ax=ax, marker='o', color='red');

predictions.plot(x='date',y=model_name,label=model_name,ax=ax, marker='x', color='green');

#evaluate

score = np.sqrt(mean_squared_error(predictions[model_name].values, test['demand']))

print('RMSE for {}: {:.4f}'.format(model_name,score))



stats = stats.append({'Model Name':model_name, 'Execution Time':t1, 'RMSE':score},ignore_index=True)
from statsmodels.tsa.holtwinters import ExponentialSmoothing



t0 = time.time()

model_name='Double Exponential Smoothing'

#train

doubleExpSmooth_model = ExponentialSmoothing(train['demand'],trend='add',seasonal_periods=7).fit()

t1 = time.time()-t0

#predict

predictions[model_name] = doubleExpSmooth_model.forecast(28).values

fig, ax = plt.subplots(figsize=(25,5))

train[-28:].plot(x='date',y='demand',label='Train',ax=ax, marker='o', color='blue')

test.plot(x='date',y='demand',label='Test',ax=ax, marker='o', color='red');

predictions.plot(x='date',y=model_name,label=model_name,ax=ax, marker='x', color='green');

#evaluate

score = np.sqrt(mean_squared_error(predictions[model_name].values, test['demand']))

print('RMSE for {}: {:.4f}'.format(model_name,score))



stats = stats.append({'Model Name':model_name, 'Execution Time':t1, 'RMSE':score},ignore_index=True)
t0 = time.time()

model_name='Triple Exponential Smoothing'

#train

tripleExpSmooth_model = ExponentialSmoothing(train['demand'],trend='add',seasonal='add',seasonal_periods=7).fit()

t1 = time.time()-t0

#predict

predictions[model_name] = tripleExpSmooth_model.forecast(28).values

fig, ax = plt.subplots(figsize=(25,4))

train[-28:].plot(x='date',y='demand',label='Train',ax=ax, marker='o', color='blue')

test.plot(x='date',y='demand',label='Test',ax=ax, marker='o', color='red');

predictions.plot(x='date',y=model_name,label=model_name,ax=ax, marker='x', color='green');

#evaluate

score = np.sqrt(mean_squared_error(predictions[model_name].values, test['demand']))

print('RMSE for {}: {:.4f}'.format(model_name,score))



stats = stats.append({'Model Name':model_name, 'Execution Time':t1, 'RMSE':score},ignore_index=True)
!pip install pmdarima
from pmdarima import auto_arima

t0 = time.time()

model_name='ARIMA'

arima_model = auto_arima(train['demand'], start_p=0, start_q=0,

                          max_p=20, max_q=5,

                          seasonal=False,

                          d=None, trace=True,random_state=12345,

                          error_action='ignore',   

                          suppress_warnings=True,  

                          stepwise=True)

arima_model.summary() 
#train

arima_model.fit(train['demand'])

t1 = time.time()-t0

#predict

predictions[model_name] = arima_model.predict(n_periods=28)

fig, ax = plt.subplots(figsize=(25,5))

train[-28:].plot(x='date',y='demand',label='Train',ax=ax, marker='o', color='blue')

test.plot(x='date',y='demand',label='Test',ax=ax, marker='o', color='red');

predictions.plot(x='date',y=model_name,label=model_name,ax=ax, marker='x', color='green');

#evaluate

score = np.sqrt(mean_squared_error(predictions[model_name].values, test['demand']))

print('RMSE for {}: {:.4f}'.format(model_name,score))



stats = stats.append({'Model Name':model_name, 'Execution Time':t1, 'RMSE':score},ignore_index=True)
t0 = time.time()

model_name='SARIMA'

sarima_model = auto_arima(train['demand'], start_p=0, start_q=0,

                          max_p=20, max_q=5,

                          seasonal=True, m=7,

                          d=None, trace=True,random_state=12345,

                          out_of_sample_size=28,

                          error_action='ignore',   

                          suppress_warnings=True,  

                          stepwise=True)

sarima_model.summary()
#train

sarima_model.fit(train['demand'])

t1 = time.time()-t0

#predict

predictions[model_name] = sarima_model.predict(n_periods=28)

fig, ax = plt.subplots(figsize=(25,4))

train[-28:].plot(x='date',y='demand',label='Train',ax=ax, marker='o', color='blue')

test.plot(x='date',y='demand',label='Test',ax=ax, marker='o', color='red');

predictions.plot(x='date',y=model_name,label=model_name,ax=ax, marker='x', color='green');

#evaluate

score = np.sqrt(mean_squared_error(predictions[model_name].values, test['demand']))

print('RMSE for {}: {:.4f}'.format(model_name,score))



stats = stats.append({'Model Name':model_name, 'Execution Time':t1, 'RMSE':score},ignore_index=True)
t0 = time.time()

model_name='SARIMAX'

sarimax_model = auto_arima(train['demand'], start_p=0, start_q=0,

                          max_p=20, max_q=5,

                          seasonal=True, m=7,

                          exogenous = train[['sell_price']].values,

                          d=None, trace=True,random_state=2020,

                          out_of_sample_size=28,

                          error_action='ignore',   

                          suppress_warnings=True,  

                          stepwise=True)

sarimax_model.summary()
#train

sarimax_model.fit(train['demand'])

t1 = time.time()-t0

#predict

predictions[model_name] = sarimax_model.predict(n_periods=28)

fig, ax = plt.subplots(figsize=(25,4))

train[-28:].plot(x='date',y='demand',label='Train',ax=ax, marker='o', color='blue')

test.plot(x='date',y='demand',label='Test',ax=ax, marker='o', color='red');

predictions.plot(x='date',y=model_name,label=model_name,ax=ax, marker='x', color='green');

#evaluate

score = np.sqrt(mean_squared_error(predictions[model_name].values, test['demand']))

print('RMSE for {}: {:.4f}'.format(model_name,score))



stats = stats.append({'Model Name':model_name, 'Execution Time':t1, 'RMSE':score},ignore_index=True)
# RMSE Comparison

stats.plot(kind='line',x='Model Name', y='RMSE', figsize=(12,4), title="RMSE Comparison for Different TS Models")

plt.xticks(rotation='vertical')
# Execution Time Comparison

stats.plot(kind='bar',x='Model Name', y='Execution Time', figsize=(12,4), title="Execution Time Comparison for Different TS Models")

plt.xticks(rotation='vertical')