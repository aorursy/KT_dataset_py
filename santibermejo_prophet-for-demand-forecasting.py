import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import os

import seaborn as sns

import datetime

import lightgbm as lgb

import time

import matplotlib.dates as mdates

import datetime as dt



from tqdm import tqdm

from random import choice

!pip uninstall --yes fbprophet

!pip install fbprophet --no-cache-dir --no-binary :all:

#!pip3 install numpy 1.14.5



from fbprophet import Prophet



from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import learning_curve



from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder



print(os.listdir("../input")) 
df = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df.head()
df.describe()
df.info()
df['date'] = pd.to_datetime(df['date'])

df_test['date'] = pd.to_datetime(df_test['date'])
df.info()
df['year'] = df['date'].dt.year

df['month'] = df['date'].dt.month

df['week'] = df['date'].dt.week

df['day'] = df['date'].dt.day

df['dayofweek'] = df['date'].dt.dayofweek

#df = df.drop('date', axis=1)



df_test['year'] = df_test['date'].dt.year

df_test['month'] = df_test['date'].dt.month

df_test['week'] = df_test['date'].dt.week

df_test['day'] = df_test['date'].dt.day

df_test['dayofweek'] = df_test['date'].dt.dayofweek

#df_test = df_test.drop('date', axis=1)



df.head()
corr = df.corr()

sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size':20})
f, ax = plt.subplots(figsize=(20,5))

df.pivot_table('sales', index=['year','month'], columns='store', aggfunc='sum').plot(ax=ax)

plt.xlabel('Year / Month')

plt.ylabel('Sales')

plt.show()



f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,5))

df.pivot_table('sales', index=['year'], columns='store', aggfunc='sum').plot(ax=ax1)

df.pivot_table('sales', index=['month'], columns='store', aggfunc='sum').plot(ax=ax2)
df_s1_i1 = df.loc[(df['store']==1) & (df['item']==1)]



m = Prophet()



# Drop the columns

ph_df = df_s1_i1.drop(['item', 'store', 'month', 'year', 'day', 'dayofweek', 'week'], axis=1)

ph_df.rename(columns={'sales': 'y', 'date': 'ds'}, inplace=True)



ph_df['y_orig'] = ph_df['y']

ph_df['y'] = np.log(ph_df['y'])



ph_df.head()
m = Prophet()

m.fit(ph_df)
future_data = m.make_future_dataframe(periods=90, freq = 'd')

forecast_data = m.predict(future_data)

forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
m.plot_components(forecast_data);
starting_date = dt.datetime(2018, 4, 1)

starting_date1 = mdates.date2num(starting_date)



pointing_arrow = dt.datetime(2018, 1, 1)

pointing_arrow1 = mdates.date2num(pointing_arrow)



fig = m.plot(forecast_data)

ax1 = fig.add_subplot(111)

ax1.set_title("Item 1/ Store 1 Stock Price Forecast", fontsize=16)

ax1.set_xlabel("Date", fontsize=12)

ax1.set_ylabel("Sales", fontsize=12)

ax1.annotate('Forecast \n Initialization', xy=(pointing_arrow1, 2.9), xytext=(starting_date1,3.5),

            arrowprops=dict(facecolor='#ff7f50', shrink=0.1),

            )



plt.show()
forecast_data_orig = forecast_data

forecast_data_orig['yhat'] = np.exp(forecast_data_orig['yhat'])

forecast_data_orig['yhat_lower'] = np.exp(forecast_data_orig['yhat_lower'])

forecast_data_orig['yhat_upper'] = np.exp(forecast_data_orig['yhat_upper'])



m.plot(forecast_data_orig);
output = forecast_data[['ds','yhat']].sort_values(by=['ds'])
output = output.loc[output['ds']>='2018-01-01'].reset_index(drop=True)
output = np.around(output['yhat']).astype(int)
df_s_i = df.groupby(['store', 'item'])
print(max(df.sales))

print(min(df.sales))
prophet_results = pd.DataFrame()



for i, d in tqdm(df_s_i):

    ph_df = d.drop(['item', 'store', 'month', 'year', 'day', 'dayofweek', 'week'], axis=1)

    ph_df = ph_df.rename(columns={'date': 'ds', 'sales': 'y'})

#    ph_df['y'] = np.log(ph_df['y'])



    m = Prophet()

    m.fit(ph_df)

    

    future_data = m.make_future_dataframe(periods=90, freq = 'd')

    forecast_data = m.predict(future_data)

       

    prophet_results=prophet_results.append(forecast_data)

    

#    prophet_results.append(forecast_data)
submission = prophet_results[['ds','yhat']]

submission = submission.loc[submission['ds'] >= '2018-01-01'].round().reset_index(drop=True)

submission['sales'] = submission['yhat'].astype(int)
submission['id'] = list(range(submission.shape[0]))

submission.drop(columns=['ds','yhat'], inplace=True)

submission = submission[['id','sales']]
submission.head()
submission.to_csv("Submission.csv", index=False)