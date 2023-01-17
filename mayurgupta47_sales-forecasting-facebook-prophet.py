import numpy as np 
import pandas as pd 
import time
import os
import seaborn as sns
import matplotlib.pyplot as plt
from fbprophet import Prophet
import statsmodels.api as sm
from plotly.offline import init_notebook_mode, iplot
from plotly import graph_objs as go
init_notebook_mode(connected=True)
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df_raw= pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/train.csv')
df_test=pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/test.csv')
df_raw.head()
df_raw.describe().T
plt.figure(figsize=(16,6))
sns.barplot(data=df_raw,x='store',y='sales')
df_store=df_raw[(df_raw['store']==2) & (df_raw['date']>='2017-01-01')]
df_sc=df_store.copy()
df_sc.loc[:,'month'] = pd.DatetimeIndex(df_sc['date']).month
#df_sc['month'] = pd.DatetimeIndex(df_sc['date']).month
df_sc.head()
df_store1=pd.DataFrame(df_sc.groupby(['month','item']).sum()['sales'])
df_store1.reset_index(inplace=True)
import plotly.express as px
fig = px.line(df_store1, x='month', y='sales',color='item')
fig.show()
df1=pd.DataFrame(df_raw.groupby('date').sum()['sales'],columns=['sales'])
df2=df1.reset_index()
df2['date']=pd.to_datetime(df2['date'])
df2['sales']=df2['sales']*1.0
df2.head()
plt.figure(figsize=(16,6))
sns.lineplot(data=df2,x='date',y='sales')
df3=df2.set_index(pd.to_datetime(df2['date']))
df3.info()
y = df3['sales'].resample('MS').mean() 
decomposition = sm.tsa.seasonal_decompose(y)
plt.figure(figsize=(16,12))
decomposition.plot()
df4=df3.reset_index(drop=True)
df4.columns=['ds','y']
df4.head()
#df4['ds'].dt.strftime('%Y-%m')
df4['year'] = pd.DatetimeIndex(df4['ds']).year
df4['month'] = pd.DatetimeIndex(df4['ds']).month
df4['week'] = df4['ds'].dt.strftime('%A')
#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Period.strftime.html
df4.tail()
plt.figure(figsize=(10,6))
sns.lineplot(data=df4,x='year',y='y',ci=1)
plt.figure(figsize=(16,6))
sns.lineplot(data=df4,x='month',y='y', hue='year',ci=1)
plt.figure(figsize=(16,6))
sns.lineplot(data=df4,x='week',y='y',sort='True',hue='year',ci=1)
#df4.head()
df_raw.tail()
df_train = df_raw[(df_raw['item'] == 15) & (df_raw['store'] == 2) & (df_raw['date']<='2016-12-31')]
df_train.columns=['ds','store','item','y']
#Renaming is required since Facebook Prohet requires date column with name as ds and metric column name as y
df_train.tail()
m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
m.fit(df_train[['ds','y']])
future = m.make_future_dataframe(periods=365)
future.tail(n=3)
forecast = m.predict(future)
forecast.head(n=3)
m.plot(forecast)
m.plot_components(forecast)
df_orig = df_raw[(df_raw['item'] == 15) & (df_raw['store'] == 2)]
df_orig.columns=['ds','store','item','y']
df_orig.loc[:,('ds')]=pd.to_datetime(df_orig['ds'])
df_forecast=forecast[['ds','yhat_lower','yhat_upper','yhat']]
df_result= pd.merge(df_orig,df_forecast,on='ds')
df_result.tail()
from fbprophet.diagnostics import cross_validation
df_cv = cross_validation(m, initial='730 days', period='90 days', horizon = '365 days')
from fbprophet.diagnostics import performance_metrics
df_p = performance_metrics(df_cv)
df_p.tail()
from fbprophet.plot import plot_cross_validation_metric
fig = plot_cross_validation_metric(df_cv, metric='mape')
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
mean_absolute_percentage_error(df_result['y'],df_result['yhat'])
df_result_2017= df_result[df_result['ds']>='2017-01-01']
mean_absolute_percentage_error(df_result_2017['y'],df_result_2017['yhat'])
df_result['y - yhat']=df_result['y'] - df_result['yhat']
plt.figure(figsize=(16,6))
sns.lineplot(data=df_result,x='ds',y='y - yhat')
df_result.describe().T