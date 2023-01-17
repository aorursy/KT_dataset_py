# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
path = '/kaggle/input/usd.csv'
df = pd.read_csv(path)
df.head(5)
df.shape

time_series_data = df[['1/1/2020', '37.60']]
time_series_data.columns = ['ds', 'y']
time_series_data.ds = pd.to_datetime(time_series_data.ds,infer_datetime_format=True)
time_series_data.y = time_series_data.y.replace('\n','', regex=True)
time_series_data['y'] = pd.to_numeric(time_series_data['y'], errors='coerce')
time_series_data
train_range = np.random.rand(len(time_series_data)) < 0.8
train_ts = time_series_data[train_range]
test_ts = time_series_data[~train_range]
test_ts = test_ts.set_index('ds')
train_ts.shape
from fbprophet import Prophet

m = Prophet()
m.fit(time_series_data)
future = m.make_future_dataframe(periods=365)
prophet_pred = m.predict(future)
m.plot(prophet_pred)
#prophet_pred = pd.DataFrame({"Date" : prophet_pred[-12:]['ds'], "Pred" : prophet_pred[-12:]["yhat"]})
prophet_pred = prophet_pred.set_index("ds")
prophet_pred['ds']=prophet_pred.index
# Plot the components of the model
fig = m.plot_components(prophet_pred)
import plotly.graph_objects as go

test_fig = go.Figure() 
test_fig.add_trace(go.Scatter(
                x= test_ts.index,
                y= test_ts.y,
                name = "Actual Cases",
                line_color= "deepskyblue",
                mode = 'lines',
                opacity= 0.8))
test_fig.add_trace(go.Scatter(
                x= prophet_pred.index,
                y= prophet_pred.yhat,
                name= "Prediction",
                mode = 'lines',
                line_color = 'red',
                opacity= 0.8))
test_fig.add_trace(go.Scatter(
                x= prophet_pred.index,
                y= prophet_pred.yhat_lower,
                name= "Prediction Lower Bound",
                mode = 'lines',
                line = dict(color='gray', width=2, dash='dash'),
                opacity= 0.8))
test_fig.add_trace(go.Scatter(
                x= prophet_pred.index,
                y= prophet_pred.yhat_upper,
                name= "Prediction Upper Bound",
                mode = 'lines',
                line = dict(color='royalblue', width=2, dash='dash'),
                opacity = 0.8
                ))

test_fig.update_layout(title_text= "Prophet Model's Test Prediction",
                       xaxis_title="Date", yaxis_title="Cases",)

test_fig.show()
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
metric_df = prophet_pred.set_index('ds')[['yhat']].join(time_series_data.set_index('ds').y).reset_index()

metric_df.dropna(inplace=True)
pr=r2_score(metric_df.y, metric_df.yhat)
print("Le taux de précision de votre modéle est " + str(pr*100) + " %")
from fbprophet import Prophet
m = Prophet(
    changepoint_prior_scale=0.2, # increasing it will make the trend more flexible
    changepoint_range=0.95, # place potential changepoints in the first 98% of the time series
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=True,
    seasonality_mode='additive'
)

m.fit(train_ts)

future = m.make_future_dataframe(periods=len(test_ts))
forecast = m.predict(future)


forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7)
m.plot(forecast)
fig = m.plot_components(forecast)
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
metric_df = forecast.set_index('ds')[['yhat']].join(train_ts.set_index('ds').y).reset_index()

metric_df.dropna(inplace=True)
pr=r2_score(metric_df.y, metric_df.yhat)
print("Le taux de précision de votre modéle est " + str(pr*100) + " %")
metric_df.tail()
forecast.yhat_gr=(forecast.yhat/forecast.yhat.shift(+1)-1)*100
forecast['yhat_uppr_gr']=(forecast.yhat_upper/forecast.yhat_upper.shift(+1)-1)*100
forecast['yhat_lower_gr']=(forecast.yhat_lower/forecast.yhat_lower.shift(+1)-1)*100
forecast.yhat_uppr_gr


df_trend = forecast[['ds','yhat','yhat_uppr_gr','yhat_lower_gr']]
df_trend.dropna(inplace=True)
df_trend.columns = ['date', 'prédiction','best','worest']
df_trend.tail(30)
time_series_data = df[['1/1/2020', '37.60']]
time_series_data.columns = ['ds', 'y']
time_series_data.ds = pd.to_datetime(time_series_data.ds,infer_datetime_format=True)
time_series_data.y = time_series_data.y.replace('\n','', regex=True)
#time_series_data['y'] = time_series_data['y'].astype(float)
time_series_data['y'] = pd.to_numeric(time_series_data['y'], errors='coerce')
time_series_data