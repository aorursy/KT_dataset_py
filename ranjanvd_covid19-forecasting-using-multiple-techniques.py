# used for data analysis

import pandas as pd

import numpy as np





# Data visualization libraries

# 1. matplotlib

import matplotlib.pyplot as plt



# 2. plotly

import cufflinks as cf

import plotly.offline

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from plotly.subplots import make_subplots

import plotly.graph_objects as go



cf.go_offline()

cf.set_config_file(offline=False, world_readable=True)

init_notebook_mode(connected=True)



from statsmodels.tsa.arima_model import ARIMA

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf 

from statsmodels.tsa.seasonal import seasonal_decompose 

#from pmdarima import auto_arima                        

from sklearn.metrics import mean_squared_error

from statsmodels.tools.eval_measures import rmse
import warnings

warnings.filterwarnings('ignore')
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Loading the train dataset

test = pd.read_csv('/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_test.csv')



# Loading the test dataset

train = pd.read_csv('/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_train.csv')
train.tail()
print('The train data has',train.shape[0],'rows.')

print('The tarin data has',train.shape[1],'columns.')
fig = go.Figure()



fig.add_trace(go.Scatter(x=train['Date'], y=train['ConfirmedCases'],

                    mode='lines+markers',

                    name='Confirmed Cases'))

fig.add_trace(go.Scatter(x=train['Date'], y=train['Fatalities'],

                    mode='lines+markers',

                    name='Fatalities'))

fig.update_layout(

    title="Confirmed Cases and Fatalities in CA",

    xaxis_title="Date",

    yaxis_title="Count",

)



fig.show()
df = train[train['Date'] > '2020-03-08']

fig = go.Figure()



fig.add_trace(go.Scatter(x=df['Date'], y=np.log(df['ConfirmedCases']+1),

                    mode='lines+markers',

                    name='Confirmed Cases'))

fig.add_trace(go.Scatter(x=df['Date'], y=np.log(df['Fatalities']+1),

                    mode='lines+markers',

                    name='Fatalities'))

fig.update_layout(

    title="Confirmed Cases and Fatalities in CA",

    xaxis_title="Date",

    yaxis_title="Count",

)



fig.show()
# Creating time series data

train_data = train[['Date','ConfirmedCases','Fatalities']]

#test_data = test[['Date','ConfirmedCases','Fatalities']]



train_data.index = pd.to_datetime(train_data['Date'])

train_data = train_data[['ConfirmedCases','Fatalities']]

#test_data.index = pd.to_datetime(test_data['Date'])
# Linear regression

import statsmodels.api as sm



fig = plt.figure(figsize=(12,8))

ax1 = fig.add_subplot(211)

fig = sm.graphics.tsa.plot_acf(train['ConfirmedCases'], lags=40, ax=ax1) # 

ax2 = fig.add_subplot(212)

fig = sm.graphics.tsa.plot_pacf(train['ConfirmedCases'], lags=40, ax=ax2)# , lags=40
from pmdarima.arima import auto_arima

stepwise_model_cc = auto_arima(train['ConfirmedCases'], start_p=1, start_q=1,

                           max_p=3, max_q=3, m=12,

                           start_P=0, seasonal=False,

                           d=1, trace=True,

                           error_action='ignore',  

                           suppress_warnings=True, 

                           stepwise=True)

print(stepwise_model_cc.aic())



stepwise_model_cc.fit(train['ConfirmedCases'])
arima_model = ARIMA(train_data['ConfirmedCases'], order = (1,1,0))

arima_result = arima_model.fit()

arima_result.summary()
arima_pred_conf = arima_result.predict(start = '2020-03-12', end = '2020-04-23', typ="levels").rename("ARIMA Predictions")
fig = go.Figure()



fig.add_trace(go.Scatter(x=arima_pred_conf.index, y=arima_pred_conf.values,

                    mode='lines+markers',

                    name='Prediction'))

fig.add_trace(go.Scatter(x=train['Date'], y=train['ConfirmedCases'],

                    mode='lines+markers',

                    name='Actual'))

fig.update_layout(

    title="Confirmed Cases in CA",

    xaxis_title="Date",

    yaxis_title="Count",

)



fig.show()
import statsmodels.api as sm



fig = plt.figure(figsize=(12,8))

ax1 = fig.add_subplot(211)

fig = sm.graphics.tsa.plot_acf(train['Fatalities'], lags=40, ax=ax1) # 

ax2 = fig.add_subplot(212)

fig = sm.graphics.tsa.plot_pacf(train['Fatalities'], lags=40, ax=ax2)# , lags=40
from pmdarima.arima import auto_arima

stepwise_model_f = auto_arima(train['Fatalities'], start_p=1, start_q=1,

                           max_p=3, max_q=3, m=12,

                           start_P=0, seasonal=False,

                           d=1, trace=True,

                           error_action='ignore',  

                           suppress_warnings=True, 

                           stepwise=True)

print(stepwise_model_f.aic())
arima_model = ARIMA(train_data['Fatalities'], order = (1,1,0))

arima_result = arima_model.fit()

arima_result.summary()
arima_pred_fatal = arima_result.predict(start = '2020-03-12', end = '2020-04-23', typ="levels").rename("ARIMA Predictions")
frame = { 'ConfirmedCases': arima_pred_conf, 'Fatalities': arima_pred_fatal } 

df = pd.DataFrame(frame) 



test.index = pd.to_datetime(test['Date'])

final = pd.merge(test,df,left_index=True,right_index=True,how='left')

final.head()
final[['ForecastId','ConfirmedCases','Fatalities']].to_csv('submission.csv',index=False)