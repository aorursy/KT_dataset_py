import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import fbprophet as Prophet

from sklearn.metrics import mean_squared_error,r2_score

import math

import time

%matplotlib inline
df = pd.read_csv('../input/traffic/DataSet/train27303.csv')
df.head()
df.tail()
df.shape
plt.figure(figsize=(20,10))

plt.plot(df['hourly_traffic_count'])

plt.show()
df = df.iloc[:9792,:]



plt.figure(figsize=(20,10))

plt.plot(df['hourly_traffic_count'])

plt.show()
df = df.rename(columns = {'timestamp':'ds','hourly_traffic_count':'y'})

df.dtypes
df['ds'] = pd.to_datetime(df['ds'])

df.dtypes
df
train_df = df.iloc[:8928,:]

test_df = df.iloc[8928:,:]
train_df
test_df
start = time.time()



m = Prophet.Prophet(changepoint_prior_scale=0.01)

m.fit(train_df)



end = time.time()

print('Time Required :', end-start,'seconds')
test = test_df.iloc[:,0:1]

fcst = m.predict(test)
fcst
fig = m.plot(fcst)
fig2 = m.plot_components(fcst)
plt.figure(figsize=(20,10))

plt.plot(test_df['ds'],test_df['y'], color = 'red', label = 'Real Traffic Count')

plt.plot(fcst['ds'],fcst['yhat'].astype(int), color = 'blue', label = 'Predicted Traffic Count')

plt.title('Traffic Prediction')

plt.legend()

plt.show()
output = test_df.copy()

output['y_pred'] = fcst['yhat'].values.astype(int)

output
output.to_csv('./output.csv',index=False)
rmse = math.sqrt(mean_squared_error(test_df['y'],fcst['yhat'].astype(int)))

print("Root Mean Squared Error :",rmse)

r_sq = r2_score(test_df['y'],fcst['yhat'].astype(int))

print("R Squared Score :",r_sq)

print('Time Required :', end-start,'seconds')